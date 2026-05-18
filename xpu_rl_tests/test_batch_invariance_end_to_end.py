# Run with TRITON_INTEL_PREDICATED_LOAD=0 TRITON_INTEL_PREDICATED_STORE=0 VLLM_BATCH_INVARIANT=[1/0]
# should pass with batch invariant ON, may fail with it OFF (especially on larger M)
# remember to clone and install vllm-fork and intel-xpu-backend-for-triton

import os
import sys

# Must be set before importing vllm
BATCH_INVARIANT = os.environ.get('VLLM_BATCH_INVARIANT', '1')
os.environ['VLLM_BATCH_INVARIANT'] = BATCH_INVARIANT


def _sync(device='xpu'):
    """Synchronize device."""
    import torch
    if device == 'xpu':
        torch.xpu.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def _benchmark_mm(mm_fn, a, b, device='xpu', warmup=5, iters=20):
    """Benchmark a matmul call, returning median time in ms."""
    import time
    for _ in range(warmup):
        mm_fn(a, b)
    _sync(device)

    times = []
    for _ in range(iters):
        _sync(device)
        start = time.perf_counter()
        mm_fn(a, b)
        _sync(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times.sort()
    return times[len(times) // 2]


def kernel_level_test(device='xpu'):
    """Test whether matmul produces identical results for the same row
    when embedded in matrices of different sizes (the core batch invariance
    property).  This catches non-deterministic tiling/split-k in BLAS."""
    import torch
    print("=" * 60)
    print("KERNEL-LEVEL BATCH INVARIANCE TEST")
    print("=" * 60)

    bi_mode = BATCH_INVARIANT == '1'
    if bi_mode:
        from vllm.model_executor.layers.batch_invariant import (
            enable_batch_invariant_mode,
        )
        enable_batch_invariant_mode()
        mm_fn = torch.mm  # dispatches to Triton override
        print("Mode: batch invariant ON (Triton matmul override)")
    else:
        mm_fn = torch.mm  # dispatches to oneMKL
        print("Mode: batch invariant OFF (oneMKL)")

    all_pass = True

    # Test multiple hidden sizes including realistic LLM dimensions
    test_configs = [
        # (hidden, intermediate, label)
        (768, 3072, "opt-125m sized"),
        (4096, 11008, "llama-7b sized"),
        (4096, 4096, "square 4k"),
        (5120, 13824, "llama-13b sized"),
        (8192, 8192, "square 8k"),
    ]

    print(f"\n{'Config':<22} {'M':>5} {'Invariant':<10} {'Time (ms)':>10}")
    print(f"{'-'*22} {'-'*5} {'-'*10} {'-'*10}")

    perf_results = []  # (label, M, time_ms)

    for hidden, intermediate, label in test_configs:
        torch.manual_seed(42)
        weight = torch.randn(hidden, intermediate, dtype=torch.bfloat16,
                             device=device)
        row = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)

        # Reference: compute the single row alone
        ref = mm_fn(row, weight)

        batch_sizes = [2, 8, 32, 128, 512]
        for bs in batch_sizes:
            # Embed our target row at position 3 (or last if bs < 4)
            pos = min(3, bs - 1)
            batch = torch.randn(bs, hidden, dtype=torch.bfloat16,
                                device=device)
            batch[pos] = row[0]

            # Measure performance
            time_ms = _benchmark_mm(mm_fn, batch, weight, device)

            result = mm_fn(batch, weight)
            match = torch.equal(ref[0], result[pos])
            status = "PASS" if match else "FAIL"

            perf_results.append((label, bs, time_ms, match))

            if not match:
                diff = (ref[0].float() - result[pos].float()).abs()
                print(f"{label:<22} {bs:>5} {status:<10} {time_ms:>9.3f}  "
                      f"max_diff={diff.max().item():.6e}  "
                      f"num_diff={(diff > 0).sum().item()}/{diff.numel()}")
                all_pass = False
            else:
                print(f"{label:<22} {bs:>5} {status:<10} {time_ms:>9.3f}")

    # Performance summary
    print(f"\n{'=' * 60}")
    print("KERNEL PERFORMANCE SUMMARY")
    print(f"{'=' * 60}")
    # Group by config, show TFLOPS
    print(f"  {'Config':<22} {'M':>5} {'ms':>8} {'TFLOPS':>8}")
    print(f"  {'-'*22} {'-'*5} {'-'*8} {'-'*8}")
    for hidden, intermediate, label in test_configs:
        for (lbl, bs, time_ms, _) in perf_results:
            if lbl == label:
                flops = 2.0 * bs * hidden * intermediate
                tflops = flops / (time_ms * 1e-3) / 1e12
                print(f"  {label:<22} {bs:>5} {time_ms:>7.3f} {tflops:>7.2f}")

    return all_pass


def e2e_test():
    """End-to-end vLLM inference test: same prompt at different batch
    positions should produce identical output."""
    import time
    from vllm import LLM, SamplingParams
    import torch

    print("\n" + "=" * 60)
    print("END-TO-END vLLM INFERENCE TEST")
    print("=" * 60)

    model = os.environ.get('TEST_MODEL', 'facebook/opt-1.3b')
    llm = LLM(model=model, dtype='bfloat16', max_model_len=512,
              gpu_memory_utilization=0.7, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=64, seed=42)
    prompt = (
        'The meaning of life is a question that has been debated by '
        'philosophers for centuries. Some argue that the purpose of '
        'existence is to seek happiness, while others believe it is'
    )
    # Longer filler prompts to push prefill M dimension into non-deterministic
    # territory (M=32+ showed failures in kernel-level tests).
    filler = (
        'Explain the theory of general relativity in simple terms and '
        'describe how it affects our understanding of time and space in '
        'the modern era of physics research and scientific discovery'
    )
    # Test with increasing batch sizes — larger M during prefill increases
    # the chance that non-deterministic tiling in BLAS changes the result.
    batch_configs = [
        (8, [0, 3, 7]),
        (32, [0, 15, 31]),
        (64, [0, 31, 63]),
    ]

    bi_mode = BATCH_INVARIANT == '1'
    dev = (torch.xpu.get_device_name()
           if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cuda')
    print(f"Model: {model}")
    print(f"Batch invariant mode: {'ON' if bi_mode else 'OFF'}")
    print(f"Device: {dev}\n")

    # Reference: run prompt alone (with timing)
    start = time.perf_counter()
    out_alone = llm.generate([prompt], sp)[0].outputs[0].text
    time_alone = (time.perf_counter() - start) * 1000
    print(f"Reference (alone): {out_alone[:60]}")
    print(f"  Time: {time_alone:.1f} ms\n")

    all_pass = True
    perf_data = []  # (batch_size, pos, time_ms, match)

    # Check 1: same output regardless of position in batch
    print("Position invariance:")
    for batch_size, positions in batch_configs:
        print(f"\n  batch_size={batch_size}:")
        for pos in positions:
            batch = [filler] * batch_size
            batch[pos] = prompt
            start = time.perf_counter()
            out_n = llm.generate(batch, sp)[pos].outputs[0].text
            elapsed = (time.perf_counter() - start) * 1000
            match = out_alone == out_n
            if not match:
                all_pass = False
            perf_data.append((batch_size, pos, elapsed, match))
            print(f"    pos={pos:>2}: {'PASS' if match else 'FAIL'} | "
                  f"{elapsed:>7.1f} ms | {out_n[:50]}")

    # Check 2: repeated runs are identical (with timing)
    print("\nRepeatability (3 runs at pos=15 in batch of 32):")
    repeat_results = []
    repeat_times = []
    for i in range(3):
        batch = [filler] * 32
        batch[15] = prompt
        start = time.perf_counter()
        out = llm.generate(batch, sp)[15].outputs[0].text
        elapsed = (time.perf_counter() - start) * 1000
        repeat_results.append(out)
        repeat_times.append(elapsed)
        print(f"  run {i} ({len(out)} chars, {elapsed:.1f} ms): {out!r}")

    repeatable = all(r == repeat_results[0] for r in repeat_results)
    print(f"  All identical: {repeatable}")
    if not repeatable:
        all_pass = False
        for i, r in enumerate(repeat_results[1:], 1):
            if r != repeat_results[0]:
                diverge = next(
                    (j for j, (a, b) in enumerate(zip(repeat_results[0], r))
                     if a != b),
                    min(len(repeat_results[0]), len(r))
                )
                print(f"  run 0 vs run {i}: diverge at char {diverge}")
                print(f"    run 0: ...{repeat_results[0][max(0,diverge-10):diverge+30]!r}")
                print(f"    run {i}: ...{r[max(0,diverge-10):diverge+30]!r}")

    # Performance summary
    print(f"\n{'=' * 60}")
    print("E2E PERFORMANCE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Single prompt (batch=1): {time_alone:.1f} ms")
    print(f"  Repeatability (batch=32, 3 runs): "
          f"avg={sum(repeat_times)/len(repeat_times):.1f} ms, "
          f"std={((sum((t - sum(repeat_times)/len(repeat_times))**2 for t in repeat_times) / len(repeat_times))**0.5):.1f} ms")
    print(f"\n  {'Batch':>6} {'Pos':>4} {'Time (ms)':>10} {'tok/s':>8} {'Invariant'}")
    print(f"  {'-'*6} {'-'*4} {'-'*10} {'-'*8} {'-'*9}")
    for batch_size, pos, elapsed, match in perf_data:
        # Approximate: 64 tokens generated per prompt in batch
        tok_per_sec = (batch_size * 64) / (elapsed / 1000)
        print(f"  {batch_size:>6} {pos:>4} {elapsed:>9.1f} {tok_per_sec:>7.1f} "
              f"{'PASS' if match else 'FAIL'}")

    return all_pass


if __name__ == '__main__':
    kernel_pass = kernel_level_test()

    print()
    e2e_pass = e2e_test()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Kernel-level: {'PASS' if kernel_pass else 'FAIL'}")
    print(f"  End-to-end:   {'PASS' if e2e_pass else 'FAIL'}")

    if kernel_pass and e2e_pass:
        print("\nRESULT: ALL CHECKS PASSED")
    else:
        print("\nRESULT: SOME CHECKS FAILED")
        sys.exit(1)