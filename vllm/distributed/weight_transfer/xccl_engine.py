# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XCCL-based weight transfer engine for Intel XPU.

Implements TRL-compatible weight synchronization using ProcessGroupXCCL
for communication between the trainer (TRL client) and vLLM workers.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as c10d

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class XCCLWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for XCCL-based weight transfer (TRL-compatible).

    Matches the JSON payload from TRL's ``init_communicator()`` call:
    ``{"host": "0.0.0.0", "port": 12345, "world_size": 2, "client_device_uuid": "..."}``
    """

    host: str
    port: int
    world_size: int
    client_device_uuid: str = ""


@dataclass
class XCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for a single named parameter (TRL-compatible).

    Matches the JSON payload from TRL's ``update_named_param()`` call:
    ``{"name": "model.layers.0...", "dtype": "torch.bfloat16", "shape": [4096, 4096]}``
    """

    name: str
    dtype: str
    shape: list[int]


def worker_init_xccl_process_group(
    init_info: XCCLWeightTransferInitInfo,
    parallel_config: "ParallelConfig",
) -> c10d.ProcessGroup:
    """Create a ProcessGroupXCCL on the server side to join TRL's collective group.

    TRL (the client) creates its side with:
        rank = vllm_world_size (last rank)
        is_master = False

    The vLLM server joins with:
        rank = worker_rank (starting from 0)
        is_master = True (creates the TCPStore)
    """
    dp_rank = parallel_config.data_parallel_index
    world_size_per_dp = parallel_config.world_size  # TP * PP
    rank_within_dp = parallel_config.rank

    # Unique rank across all DP groups
    worker_rank = dp_rank * world_size_per_dp + rank_within_dp

    logger.info(
        "Initializing XCCL weight transfer: host=%s, port=%d, "
        "world_size=%d, worker_rank=%d",
        init_info.host,
        init_info.port,
        init_info.world_size,
        worker_rank,
    )

    store = c10d.TCPStore(
        host_name=init_info.host,
        port=init_info.port,
        world_size=init_info.world_size,
        is_master=(worker_rank == 0),
    )
    prefixed_store = c10d.PrefixStore("client2server", store)
    xccl_options = c10d.ProcessGroupXCCL.Options()
    pg = c10d.ProcessGroupXCCL(
        store=prefixed_store,
        rank=worker_rank,
        size=init_info.world_size,
        options=xccl_options,
    )
    return pg


class XCCLWeightTransferEngine(
    WeightTransferEngine[XCCLWeightTransferInitInfo, XCCLWeightTransferUpdateInfo]
):
    """
    Weight transfer engine using XCCL (Intel XPU) for TRL-compatible communication.

    TRL's GRPOTrainer in server mode broadcasts updated weights from the trainer
    process to the vLLM server using ProcessGroupXCCL. This engine handles the
    server side of that protocol.

    Protocol:
        1. TRL calls POST /init_communicator/ → server creates ProcessGroupXCCL
        2. For each param: TRL calls POST /update_named_param/ with {name, dtype, shape}
           → server allocates tensor, receives broadcast, loads into model
        3. TRL calls POST /close_communicator/ → server destroys process group
    """

    init_info_cls = XCCLWeightTransferInitInfo
    update_info_cls = XCCLWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        parallel_config: "ParallelConfig",
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, parallel_config, model)
        self.model_update_group: c10d.ProcessGroup | None = None
        self._client_rank: int = 0
        self._device: torch.device = torch.device("xpu", torch.xpu.current_device())

    def init_transfer_engine(self, init_info: XCCLWeightTransferInitInfo) -> None:
        """Initialize XCCL process group with the TRL trainer."""
        self.model_update_group = worker_init_xccl_process_group(
            init_info, self.parallel_config
        )
        # TRL client rank is the last rank in the group
        self._client_rank = init_info.world_size - 1
        logger.info(
            "XCCL weight transfer initialized. Client rank: %d",
            self._client_rank,
        )

    def receive_weights(
        self,
        update_info: XCCLWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Receive a single named parameter via XCCL broadcast from the TRL trainer.

        Args:
            update_info: Contains the parameter name, dtype string, and shape.
            load_weights: Callable that loads weights into the model.
        """
        if self.model_update_group is None:
            raise RuntimeError(
                "XCCL weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        dtype = getattr(torch, update_info.dtype.removeprefix("torch."))
        weight = torch.empty(update_info.shape, dtype=dtype, device=self._device)

        # Receive broadcast from TRL client (root = client rank)
        self.model_update_group.broadcast(weight, root=self._client_rank)
        self.model_update_group.barrier()

        # Load into model
        load_weights([(update_info.name, weight)])
        del weight

    def close(self) -> None:
        """Destroy the XCCL process group."""
        if self.model_update_group is not None:
            logger.info("Closing XCCL weight transfer communicator")
            self.model_update_group = None

    def shutdown(self) -> None:
        """Shutdown the XCCL weight transfer engine."""
        self.close()

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, Any]],
        trainer_args: "dict[str, Any] | Any",
    ) -> None:
        """Not used for XCCL engine -- TRL handles the client side."""
        raise NotImplementedError(
            "XCCLWeightTransferEngine does not implement trainer_send_weights. "
            "The TRL client handles broadcasting via ProcessGroupXCCL directly."
        )
