# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accelerator-agnostic dense weight transfer over `torch.distributed`.

The `nccl` backend reaches NCCL through `PyNcclCommunicator`, which loads
`libnccl` directly and so is limited to CUDA and ROCm. This engine performs the
same dense broadcast through `torch.distributed` collectives instead, so the
transport is whichever backend the platform already uses for its own
communication (`xccl` on XPU, `nccl` on CUDA, `gloo` on CPU). That makes trainer
-> inference weight sync available on accelerators that have no NCCL at all.

The group is created with `stateless_init_torch_distributed_process_group`, so
it is independent of the workers' existing tensor/pipeline-parallel groups:
the trainer is a separate process that is not a member of those. Broadcasts
therefore address the sender with `group_src`, not the global `src` rank, which
has no meaning in a stateless group.

Weights arrive in checkpoint (HF) format and are applied through the model's own
`load_weights` under the layerwise reload lifecycle, exactly as in `nccl_engine`.
Tensors are sent one at a time; the packed-buffer batching that the NCCL backend
uses is specific to `PyNcclCommunicator` and has no equivalent here yet.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.distributed as dist
from typing_extensions import Self

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.config import VllmConfig

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    SupportsWeightTransferClose,
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
    checked_iter,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "TorchDistWeightTransferInitInfo",
    "TorchDistTrainerInitInfo",
    "TorchDistWeightTransferUpdateInfo",
    "TorchDistWeightTransferEngine",
    "TorchDistTrainerWeightTransferEngine",
]

# The sender is always rank 0 of the transfer group, so workers sit at offset 1.
_SENDER_GROUP_RANK = 0


def resolve_dist_backend(backend: str | None) -> str:
    """Return the `torch.distributed` backend to use for the transfer group.

    `None` means "whatever this platform communicates with", which is the right
    default on both sides: the trainer and the workers run on the same kind of
    accelerator, so they resolve to the same string without having to agree on
    one explicitly.
    """
    if backend is not None:
        return backend
    from vllm.platforms import current_platform

    return current_platform.dist_backend


@dataclass
class TorchDistWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info for the `torch.distributed` backend.

    `rank_offset` places this worker after the trainer ranks in the transfer
    group; `world_size` is the full trainer+worker size. `dist_backend` is
    propagated by the trainer so the two sides cannot pick different backends.
    """

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    dist_backend: str | None = None


@dataclass
class TorchDistTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for the `torch.distributed` backend.

    The sender is rank 0 of the transfer group and also hosts its rendezvous
    store, so `master_address`/`master_port` must be reachable *from the
    workers*. `rank` (from `TrainerInitInfo`) identifies this trainer process;
    rank 0 is the sender.
    """

    backend: ClassVar[str] = "torch_dist"

    master_address: str
    master_port: int
    world_size: int
    dist_backend: str | None = None


@dataclass
class TorchDistWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round parameter metadata: what the worker is about to receive."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]

    def __post_init__(self):
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {num_params}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {num_params}"
            )


def _init_transfer_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    dist_backend: str | None,
) -> "ProcessGroup":
    """Join the trainer<->worker transfer group without touching global state.

    The caller must already have selected this process's accelerator device:
    the backend binds its communicator to the current device on first use.
    """
    from vllm.distributed.utils import stateless_init_torch_distributed_process_group

    backend = resolve_dist_backend(dist_backend)
    logger.info(
        "Joining weight transfer group %s:%s as rank %d/%d over %s",
        master_address,
        master_port,
        rank,
        world_size,
        backend,
    )
    return stateless_init_torch_distributed_process_group(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size,
        backend=backend,
    )


def _rendezvous(pg: "ProcessGroup") -> None:
    """Re-couple the two ends of the transfer after one tensor.

    A collective with `async_op=False` only orders the calling stream against
    the transport's; the host returns before the data has landed. Nothing else
    throttles the sender, so it enqueues its whole stream of tensors while the
    receiver is still loading the first one -- and some backends (oneCCL, used
    for `xccl`) then deadlock: once the two ends have drifted apart by more than
    a handful of collectives, none of them ever completes and both ranks spin in
    the next device synchronization. One barrier per tensor bounds the drift at a
    single collective. It costs well under a millisecond against a transfer
    measured in hundreds, so it is not worth making conditional.
    """
    dist.barrier(group=pg)


def _release_transfer_group(pg: "ProcessGroup | None") -> None:
    """Leave the transfer group without waiting for the other end.

    The control plane has no teardown call, so the two ends never leave
    together: whichever side goes first faces a peer that stays in the group.
    A graceful `shutdown()` is the wrong tool for that -- it finalizes the
    transport, which on oneCCL means a handshake that either waits on a peer
    that will never arrive or, if that peer has already died, fails hard enough
    to take this process down with it. `abort()` drops the local state only,
    and a transfer group has nothing in flight to lose: every round ends
    synchronized.
    """
    if pg is None:
        return
    from torch.distributed.distributed_c10d import _unregister_process_group

    pg.abort()
    _unregister_process_group(pg.group_name)


class TorchDistWeightTransferEngine(
    WeightTransferEngine[
        TorchDistWeightTransferInitInfo, TorchDistWeightTransferUpdateInfo
    ]
):
    """Worker-side dense weight receive over `torch.distributed` broadcasts."""

    init_info_cls = TorchDistWeightTransferInitInfo
    update_info_cls = TorchDistWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.model_update_group: ProcessGroup | None = None

    def init_transfer_engine(self, init_info: TorchDistWeightTransferInitInfo) -> None:
        """Join the transfer group, ranked uniquely across all DP groups.

        A group left over from an earlier trainer is dropped first: it keeps the
        old rendezvous port connected, so a trainer that reconnects on the same
        port would fail to bind its store.
        """
        self.shutdown()
        parallel_config = self.parallel_config
        worker_rank = (
            parallel_config.data_parallel_index * parallel_config.world_size
            + parallel_config.rank
        )
        self.model_update_group = _init_transfer_group(
            init_info.master_address,
            init_info.master_port,
            worker_rank + init_info.rank_offset,
            init_info.world_size,
            init_info.dist_backend,
        )

    def start_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

        initialize_layerwise_reload(self.model)

    def finish_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        finalize_layerwise_reload(self.model, self.model_config)

    def receive_weights(self, update_info: TorchDistWeightTransferUpdateInfo) -> None:
        if self.model_update_group is None:
            raise RuntimeError(
                "torch.distributed weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        with disable_mtp_completeness_check():
            for name, dtype_name, shape in zip(
                update_info.names, update_info.dtype_names, update_info.shapes
            ):
                dtype = getattr(torch, dtype_name)
                weight = torch.empty(shape, dtype=dtype, device=self.device)
                dist.broadcast(
                    weight,
                    group=self.model_update_group,
                    group_src=_SENDER_GROUP_RANK,
                )
                _rendezvous(self.model_update_group)
                self.model.load_weights([(name, weight)])
                del weight

    def shutdown(self) -> None:
        _release_transfer_group(self.model_update_group)
        self.model_update_group = None


class TorchDistTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[TorchDistTrainerInitInfo]
):
    """Trainer-side counterpart of `TorchDistWeightTransferEngine`.

    The sender holds the transfer group and drives the round trip: the workers'
    `update_weights` has to run concurrently with the trainer-side broadcast,
    since both sides rendezvous inside the same collective. Non-sender trainer
    ranks hold no group; they only replay the source iteration so that whatever
    collectives materializing a parameter needs (FSDP `full_tensor()`) stay
    aligned across the trainer.
    """

    init_info_cls = TorchDistTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
        is_sender: bool = True,
        dist_backend: str | None = None,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.dist_backend = dist_backend
        self.model_update_group: ProcessGroup | None = None

    @classmethod
    def trainer_init(
        cls,
        init_info: TorchDistTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            dist_backend=init_info.dist_backend,
        )
        if not engine.is_sender:
            return engine

        worker_init_info = TorchDistWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=1,
            world_size=init_info.world_size,
            dist_backend=init_info.dist_backend,
        )

        # The workers block inside init_weight_transfer_engine waiting for the
        # rendezvous, so that RPC has to be in flight while this process hosts
        # the store and joins as rank 0.
        with ThreadPoolExecutor(max_workers=1) as exe:
            future = exe.submit(
                engine.client.init_weight_transfer_engine, asdict(worker_init_info)
            )
            engine.model_update_group = _init_transfer_group(
                init_info.master_address,
                init_info.master_port,
                _SENDER_GROUP_RANK,
                init_info.world_size,
                init_info.dist_backend,
            )
            future.result()  # surface any inference-side init error

        return engine

    def send_weights(
        self,
        source: WeightSource | None = None,
        *,
        drive_lifecycle: bool = True,
    ) -> None:
        """Push one full set of weights to the inference workers.

        Args:
            source: Weights to send this round, overriding the source given at
                `trainer_init`. Lets a caller that produces a fresh stream per
                round hand it over at send time instead of holding a re-iterable
                source.
            drive_lifecycle: Whether to bracket the transfer with
                `start_weight_update`/`finish_weight_update`. Pass `False` when
                the caller already opened the update to group several sends into
                one reload.
        """
        source = source or self.source
        if source is None:
            raise ValueError(
                "torch.distributed trainer weight transfer requires a "
                "WeightSource, either at trainer_init() or at send_weights()."
            )

        # Declaring metadata is itself a collective for some sources, so every
        # rank runs it; only the sender ships it.
        meta = source.metadata()

        if not self.is_sender:
            self._broadcast(source, meta)
            self._post_send_sync()
            return

        update_info = TorchDistWeightTransferUpdateInfo(
            names=[m.name for m in meta],
            dtype_names=[str(m.dtype).split(".")[-1] for m in meta],
            shapes=[list(m.shape) for m in meta],
        )

        if drive_lifecycle:
            self.client.start_weight_update()
        exe = ThreadPoolExecutor(max_workers=1)
        try:
            future = exe.submit(self.client.update_weights, asdict(update_info))
            # If the request was already rejected outright, surface that instead
            # of blocking in a broadcast whose peer will never arrive.
            if future.done():
                future.result()
            self._broadcast(source, meta)
            future.result()
        finally:
            # Never join the RPC thread on the error path: a worker that failed
            # mid-transfer is still parked in the matching collective, so
            # waiting would turn the error into a hang. The group is unusable
            # either way and the caller has to tear it down.
            exe.shutdown(wait=False)
        if drive_lifecycle:
            self.client.finish_weight_update()
        self._post_send_sync()

    def _broadcast(self, source: WeightSource, meta: list[ParamMeta]) -> None:
        if not self.is_sender:
            for _ in source:
                pass
            return

        assert self.model_update_group is not None, (
            "trainer_init() must be called before _broadcast()."
        )
        for _name, tensor in checked_iter(source, meta):
            # A non-contiguous view would ship whatever follows its base
            # pointer, so linearize first.
            send = tensor if tensor.is_contiguous() else tensor.contiguous()
            dist.broadcast(
                send, group=self.model_update_group, group_src=_SENDER_GROUP_RANK
            )
            _rendezvous(self.model_update_group)

    def _post_send_sync(self) -> None:
        """Wait for this rank's transfer work to land before returning, so the
        caller may mutate its parameters as soon as `send_weights` returns."""
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()

    def shutdown(self) -> None:
        """Leave the group, taking the inference workers out of it first.

        The workers have to go first: oneCCL finalizes the local end only once
        the peers' communicators are gone, so a trainer that drops just its own
        side stays stuck in the transport's teardown for as long as the server
        keeps running. Closing is an optional client capability, so a client
        without it degrades to that behaviour rather than failing here.
        """
        if self.is_sender and isinstance(self.client, SupportsWeightTransferClose):
            self.client.close_weight_transfer_engine()
        _release_transfer_group(self.model_update_group)
        self.model_update_group = None
