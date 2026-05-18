# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for batch-invariant matmul kernels on XPU.

Validates:
  1. Correctness against torch.matmul reference
  2. Batch invariance (same result regardless of batch composition)
  3. Performance relative to non-batch-invariant torch.matmul
"""

import time

import pytest
import torch

from vllm.model_executor.layers.batch_invariant import (
    matmul_descriptor_persistent,
    matmul_persistent,
)

# Test dimensions: (M, K, N)
MATMUL_SHAPES = [
    (128, 128, 128),
    (256, 512, 256),
    (1024, 1024, 1024),
    (2048, 4096, 2048),
]

DTYPES = [torch.bfloat16, torch.float16]

DEVICE = "xpu"

# Tolerance for correctness checks
RTOL = {torch.bfloat16: 0.02, torch.float16: 0.01, torch.float32: 1e-4}
ATOL = {torch.bfloat16: 0.05, torch.float16: 0.02, torch.float32: 1e-4}

# Number of iterations for batch-invariance checks
NUM_INVARIANCE_TRIALS = 5

# Number of warmup/benchmark iterations
WARMUP_ITERS = 10
BENCH_ITERS = 50


def _sync():
    """Synchronize XPU device."""
    torch.xpu.synchronize()


def _benchmark(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark a callable, returning median time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    _sync()

    times = []
    for _ in range(iters):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times.sort()
    return times[len(times) // 2]


def _check_batch_invariance(matmul_fn, M, K, N, dtype, num_trials):
    """Check that the same input row produces identical output regardless of batch.

    A batch-invariant kernel must produce the same result for a given row
    regardless of what other rows are in the batch. We embed a fixed "probe"
    row into batches of varying sizes and random content, then verify the
    output row for the probe is always bit-identical.
    """
    torch.manual_seed(42)
    # Fixed weight matrix and a single probe row
    w = torch.randn(K, N, device=DEVICE, dtype=dtype)
    probe_row = torch.randn(1, K, device=DEVICE, dtype=dtype)

    # Reference: compute probe_row @ w standalone (batch size = 1)
    ref = matmul_fn(probe_row, w)
    _sync()

    # Varying batch sizes to stress different tiling / scheduling paths
    batch_sizes = [2, 7, 16, 33, 64, 128, M] if M > 128 else [2, 7, 16, 33, 64, M]

    for trial, batch_size in enumerate(batch_sizes[:num_trials]):
        torch.manual_seed(trial * 1000 + 7)
        # Build a batch with random rows + the probe row at a random position
        filler = torch.randn(batch_size - 1, K, device=DEVICE, dtype=dtype)
        probe_idx = trial % batch_size  # vary where the probe sits
        if probe_idx == 0:
            batch = torch.cat([probe_row, filler], dim=0)
        elif probe_idx >= batch_size - 1:
            batch = torch.cat([filler, probe_row], dim=0)
            probe_idx = batch_size - 1
        else:
            batch = torch.cat(
                [filler[:probe_idx], probe_row, filler[probe_idx:]], dim=0
            )

        result = matmul_fn(batch, w)
        _sync()

        probe_result = result[probe_idx : probe_idx + 1]
        if not torch.equal(ref, probe_result):
            max_diff = (ref - probe_result).abs().max().item()
            return False, (
                f"Trial {trial} (batch_size={batch_size}, probe_idx={probe_idx}): "
                f"max diff = {max_diff}"
            )

    return True, "Probe row produced bit-identical results across all batch sizes"


# =============================================================================
# Test: matmul_descriptor_persistent (batch-invariant, XPU fast-path)
# =============================================================================


@pytest.mark.parametrize("M,K,N", MATMUL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
class TestMatmulDescriptorPersistent:
    """Tests for the tensor-descriptor-based persistent matmul kernel."""

    def test_correctness(self, M, K, N, dtype):
        """Result must be close to torch.matmul reference."""
        torch.manual_seed(0)
        a = torch.randn(M, K, device=DEVICE, dtype=dtype)
        b = torch.randn(K, N, device=DEVICE, dtype=dtype)

        result = matmul_descriptor_persistent(a, b)
        reference = torch.matmul(a, b)

        torch.testing.assert_close(
            result,
            reference,
            rtol=RTOL[dtype],
            atol=ATOL[dtype],
            msg=f"descriptor_persistent failed for shape ({M},{K},{N}) {dtype}",
        )

    def test_batch_invariance(self, M, K, N, dtype):
        """Kernel must produce identical results across repeated calls."""
        is_invariant, msg = _check_batch_invariance(
            matmul_descriptor_persistent, M, K, N, dtype, NUM_INVARIANCE_TRIALS
        )
        assert is_invariant, (
            f"matmul_descriptor_persistent NOT batch-invariant "
            f"for ({M},{K},{N}) {dtype}: {msg}"
        )

    def test_performance(self, M, K, N, dtype):
        """Measure speedup over standard torch.matmul."""
        torch.manual_seed(0)
        a = torch.randn(M, K, device=DEVICE, dtype=dtype)
        b = torch.randn(K, N, device=DEVICE, dtype=dtype)

        time_descriptor = _benchmark(lambda: matmul_descriptor_persistent(a, b))
        time_torch = _benchmark(lambda: torch.matmul(a, b))

        speedup = time_torch / time_descriptor
        print(
            f"\n  descriptor_persistent ({M}x{K}x{N}, {dtype}): "
            f"{time_descriptor:.3f} ms vs torch {time_torch:.3f} ms "
            f"(speedup: {speedup:.2f}x)"
        )
        # No hard assertion on speed — just report


# =============================================================================
# Test: matmul_persistent (batch-invariant, pointer-based)
# =============================================================================


@pytest.mark.parametrize("M,K,N", MATMUL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
class TestMatmulPersistent:
    """Tests for the pointer-based persistent matmul kernel."""

    def test_correctness(self, M, K, N, dtype):
        """Result must be close to torch.matmul reference."""
        torch.manual_seed(0)
        a = torch.randn(M, K, device=DEVICE, dtype=dtype)
        b = torch.randn(K, N, device=DEVICE, dtype=dtype)

        result = matmul_persistent(a, b)
        reference = torch.matmul(a, b)

        torch.testing.assert_close(
            result,
            reference,
            rtol=RTOL[dtype],
            atol=ATOL[dtype],
            msg=f"matmul_persistent failed for shape ({M},{K},{N}) {dtype}",
        )

    def test_batch_invariance(self, M, K, N, dtype):
        """Kernel must produce identical results across repeated calls."""
        is_invariant, msg = _check_batch_invariance(
            matmul_persistent, M, K, N, dtype, NUM_INVARIANCE_TRIALS
        )
        assert is_invariant, (
            f"matmul_persistent NOT batch-invariant for ({M},{K},{N}) {dtype}: {msg}"
        )

    def test_performance(self, M, K, N, dtype):
        """Measure speedup over standard torch.matmul."""
        torch.manual_seed(0)
        a = torch.randn(M, K, device=DEVICE, dtype=dtype)
        b = torch.randn(K, N, device=DEVICE, dtype=dtype)

        time_persistent = _benchmark(lambda: matmul_persistent(a, b))
        time_torch = _benchmark(lambda: torch.matmul(a, b))

        speedup = time_torch / time_persistent
        print(
            f"\n  matmul_persistent ({M}x{K}x{N}, {dtype}): "
            f"{time_persistent:.3f} ms vs torch {time_torch:.3f} ms "
            f"(speedup: {speedup:.2f}x)"
        )


# =============================================================================
# Test: torch.matmul (standard, non-batch-invariant baseline)
# =============================================================================


@pytest.mark.parametrize("M,K,N", MATMUL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
class TestTorchMatmulBaseline:
    """Tests verifying that standard torch.matmul is NOT batch-invariant.

    This serves as a control group: if torch.matmul IS batch-invariant on a
    given platform, there is no need for custom kernels.
    """

    def test_correctness(self, M, K, N, dtype):
        """Sanity: torch.matmul produces correct results."""
        torch.manual_seed(0)
        a = torch.randn(M, K, device=DEVICE, dtype=dtype)
        b = torch.randn(K, N, device=DEVICE, dtype=dtype)

        # Compare against fp32 reference
        ref = torch.matmul(a.float(), b.float()).to(dtype)
        result = torch.matmul(a, b)

        torch.testing.assert_close(
            result,
            ref,
            rtol=RTOL[dtype],
            atol=ATOL[dtype],
            msg=f"torch.matmul correctness check for ({M},{K},{N}) {dtype}",
        )

    def test_batch_invariance(self, M, K, N, dtype):
        """Check whether torch.matmul is batch-invariant (expected: often not).

        This test does NOT assert failure — it reports the result. If torch.matmul
        turns out to be invariant on the current hardware/driver, that's fine.
        """
        is_invariant, msg = _check_batch_invariance(
            torch.matmul, M, K, N, dtype, NUM_INVARIANCE_TRIALS
        )
        if not is_invariant:
            print(
                f"\n  [EXPECTED] torch.matmul NOT batch-invariant "
                f"for ({M},{K},{N}) {dtype}: {msg}"
            )
        else:
            print(
                f"\n  [INFO] torch.matmul IS batch-invariant for ({M},{K},{N}) {dtype}"
            )
        # Intentionally no assertion — this is informational

    def test_performance(self, M, K, N, dtype):
        """Baseline timing for torch.matmul."""
        torch.manual_seed(0)
        a = torch.randn(M, K, device=DEVICE, dtype=dtype)
        b = torch.randn(K, N, device=DEVICE, dtype=dtype)

        time_torch = _benchmark(lambda: torch.matmul(a, b))
        print(f"\n  torch.matmul ({M}x{K}x{N}, {dtype}): {time_torch:.3f} ms")


# =============================================================================
# Summary comparison test
# =============================================================================


@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024), (2048, 4096, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_comparative_summary(M, K, N, dtype):
    """Side-by-side comparison of all three kernels."""
    torch.manual_seed(123)
    a = torch.randn(M, K, device=DEVICE, dtype=dtype)
    b = torch.randn(K, N, device=DEVICE, dtype=dtype)

    # Correctness
    ref = torch.matmul(a, b)
    res_descriptor = matmul_descriptor_persistent(a, b)
    res_persistent = matmul_persistent(a, b)

    torch.testing.assert_close(res_descriptor, ref, rtol=RTOL[dtype], atol=ATOL[dtype])
    torch.testing.assert_close(res_persistent, ref, rtol=RTOL[dtype], atol=ATOL[dtype])

    # Batch invariance
    inv_descriptor, _ = _check_batch_invariance(
        matmul_descriptor_persistent, M, K, N, dtype, NUM_INVARIANCE_TRIALS
    )
    inv_persistent, _ = _check_batch_invariance(
        matmul_persistent, M, K, N, dtype, NUM_INVARIANCE_TRIALS
    )
    inv_torch, _ = _check_batch_invariance(
        torch.matmul, M, K, N, dtype, NUM_INVARIANCE_TRIALS
    )

    # Performance
    time_descriptor = _benchmark(lambda: matmul_descriptor_persistent(a, b))
    time_persistent = _benchmark(lambda: matmul_persistent(a, b))
    time_torch = _benchmark(lambda: torch.matmul(a, b))

    print(f"\n{'=' * 70}")
    print(f"  Comparative Summary: ({M}x{K}x{N}, {dtype})")
    print(f"{'=' * 70}")
    print(f"  {'Kernel':<30} {'Time (ms)':<12} {'Invariant':<12} {'vs torch'}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 10}")
    print(
        f"  {'descriptor_persistent':<30} {time_descriptor:<12.3f} "
        f"{'YES' if inv_descriptor else 'NO':<12} "
        f"{time_torch / time_descriptor:.2f}x"
    )
    print(
        f"  {'matmul_persistent':<30} {time_persistent:<12.3f} "
        f"{'YES' if inv_persistent else 'NO':<12} "
        f"{time_torch / time_persistent:.2f}x"
    )
    print(
        f"  {'torch.matmul (baseline)':<30} {time_torch:<12.3f} "
        f"{'YES' if inv_torch else 'NO':<12} "
        f"1.00x"
    )
    print(f"{'=' * 70}")

    # Assert batch-invariant kernels are indeed invariant
    assert inv_descriptor, "matmul_descriptor_persistent must be batch-invariant"
    assert inv_persistent, "matmul_persistent must be batch-invariant"


if __name__ == "__main__":
    print("Running matmul kernel tests on XPU...\n")

    # Track results across all shapes/dtypes
    results = {
        "descriptor_persistent": {"correct": [], "invariant": [], "times": []},
        "matmul_persistent": {"correct": [], "invariant": [], "times": []},
        "torch.matmul": {"correct": [], "invariant": [], "times": []},
    }

    for dtype in DTYPES:
        for M, K, N in MATMUL_SHAPES:
            print(f"--- Shape ({M}, {K}, {N}), dtype={dtype} ---")

            torch.manual_seed(0)
            a = torch.randn(M, K, device=DEVICE, dtype=dtype)
            b = torch.randn(K, N, device=DEVICE, dtype=dtype)
            ref = torch.matmul(a.float(), b.float()).to(dtype)

            # Correctness
            res_desc = matmul_descriptor_persistent(a, b)
            res_pers = matmul_persistent(a, b)
            res_torch = torch.matmul(a, b)

            desc_ok = torch.allclose(res_desc, ref, rtol=RTOL[dtype], atol=ATOL[dtype])
            pers_ok = torch.allclose(res_pers, ref, rtol=RTOL[dtype], atol=ATOL[dtype])
            torch_ok = torch.allclose(
                res_torch, ref, rtol=RTOL[dtype], atol=ATOL[dtype]
            )
            results["descriptor_persistent"]["correct"].append(desc_ok)
            results["matmul_persistent"]["correct"].append(pers_ok)
            results["torch.matmul"]["correct"].append(torch_ok)
            print(
                f"  Correctness: descriptor={desc_ok}, "
                f"persistent={pers_ok}, torch={torch_ok}"
            )

            # Batch invariance
            inv_desc, msg_desc = _check_batch_invariance(
                matmul_descriptor_persistent, M, K, N, dtype, NUM_INVARIANCE_TRIALS
            )
            inv_pers, msg_pers = _check_batch_invariance(
                matmul_persistent, M, K, N, dtype, NUM_INVARIANCE_TRIALS
            )
            inv_torch, msg_torch = _check_batch_invariance(
                torch.matmul, M, K, N, dtype, NUM_INVARIANCE_TRIALS
            )
            results["descriptor_persistent"]["invariant"].append(inv_desc)
            results["matmul_persistent"]["invariant"].append(inv_pers)
            results["torch.matmul"]["invariant"].append(inv_torch)
            print(
                f"  Batch-invariant: descriptor={inv_desc}, "
                f"persistent={inv_pers}, torch={inv_torch}"
            )
            if not inv_torch:
                print(f"    torch detail: {msg_torch}")

            # Performance
            time_desc = _benchmark(lambda: matmul_descriptor_persistent(a, b))
            time_pers = _benchmark(lambda: matmul_persistent(a, b))
            time_torch = _benchmark(lambda: torch.matmul(a, b))
            results["descriptor_persistent"]["times"].append(
                (time_desc, time_torch)
            )
            results["matmul_persistent"]["times"].append((time_pers, time_torch))
            results["torch.matmul"]["times"].append((time_torch, time_torch))
            print(
                f"  Time (ms): descriptor={time_desc:.3f}, "
                f"persistent={time_pers:.3f}, torch={time_torch:.3f}"
            )
            print(
                f"  Speedup vs torch: descriptor={time_torch / time_desc:.2f}x, "
                f"persistent={time_torch / time_pers:.2f}x"
            )
            print()

    # =========================================================================
    # Final summary
    # =========================================================================
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    all_pass = True
    for name, data in results.items():
        n_correct = sum(data["correct"])
        n_total = len(data["correct"])
        n_invariant = sum(data["invariant"])

        correct_all = n_correct == n_total
        invariant_all = n_invariant == n_total

        # Average slowdown vs torch baseline (only meaningful for custom kernels)
        if data["times"]:
            avg_ratio = sum(t / base for t, base in data["times"]) / len(
                data["times"]
            )
        else:
            avg_ratio = 1.0

        status = "PASS" if (correct_all and invariant_all) else "FAIL"
        if name == "torch.matmul" and not invariant_all:
            status = "EXPECTED (not invariant)"

        print(f"\n  {name}:")
        print(f"    Correct:         {n_correct}/{n_total} {'OK' if correct_all else 'FAIL'}")
        print(f"    Batch-invariant: {n_invariant}/{n_total} {'OK' if invariant_all else 'FAIL'}")
        if name != "torch.matmul":
            print(f"    Avg time ratio vs torch: {avg_ratio:.2f}x (< 1 = faster)")
        print(f"    Status: {status}")

        if name != "torch.matmul" and (not correct_all or not invariant_all):
            all_pass = False

    print(f"\n{'=' * 70}")
    if all_pass:
        print("  ALL BATCH-INVARIANT KERNELS: PASSED (correct + invariant)")
    else:
        print("  SOME BATCH-INVARIANT KERNELS: FAILED")
    print(f"{'=' * 70}")

    if not all_pass:
        raise SystemExit(1)
