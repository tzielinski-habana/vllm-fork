"""Test whether oneMKL CNR mode provides batch invariance on Intel XPU.

Run with: MKL_CBWR=AUTO python test_onemkl_cnr_batch_invariance.py
Run without CNR for comparison: python test_onemkl_cnr_batch_invariance.py

Tests whether the result for a fixed row changes when the total number of rows
(M dimension) changes. Batch invariance requires bitwise-identical results
regardless of M.
"""
import os
import torch

device = "xpu"
dtype = torch.bfloat16

cnr_mode = os.environ.get("MKL_CBWR", "not set")
print(f"MKL_CBWR = {cnr_mode}")
print(f"Device: {torch.xpu.get_device_name(0)}")
print(f"Dtype: {dtype}")
print("=" * 60)


def test_matmul_batch_invariance(K, N, M_sizes, num_trials=10):
    """Test if row 0 of (M, K) @ (K, N) is the same for different M values."""
    print(f"\n--- torch.mm batch invariance: K={K}, N={N} ---")
    W = torch.randn(K, N, device=device, dtype=dtype)

    all_pass = True
    for trial in range(num_trials):
        x = torch.randn(1, K, device=device, dtype=dtype)
        result_single = x @ W  # M=1

        for M in M_sizes:
            pad = torch.randn(M - 1, K, device=device, dtype=dtype)
            big_input = torch.cat([x, pad], dim=0)
            result_big = big_input @ W  # M=M
            row0 = result_big[0:1]

            diff = (result_single - row0).abs().max().item()
            match = diff == 0.0
            if not match:
                all_pass = False
                print(f"  trial={trial}, M={M}: FAIL (max diff = {diff:.6e})")

    if all_pass:
        print(f"  All {num_trials} trials PASS for M in {M_sizes}")
    return all_pass


def test_bmm_batch_invariance(B, K, N, M_sizes, num_trials=10):
    """Test if torch.bmm result for a fixed batch element is the same for different M."""
    print(f"\n--- torch.bmm batch invariance: B={B}, K={K}, N={N} ---")
    W = torch.randn(B, K, N, device=device, dtype=dtype)

    all_pass = True
    for trial in range(num_trials):
        x = torch.randn(B, 1, K, device=device, dtype=dtype)
        result_single = torch.bmm(x, W)  # M=1

        for M in M_sizes:
            pad = torch.randn(B, M - 1, K, device=device, dtype=dtype)
            big_input = torch.cat([x, pad], dim=1)
            result_big = torch.bmm(big_input, W)  # M=M
            row0 = result_big[:, 0:1, :]

            diff = (result_single - row0).abs().max().item()
            match = diff == 0.0
            if not match:
                all_pass = False
                print(f"  trial={trial}, M={M}: FAIL (max diff = {diff:.6e})")

    if all_pass:
        print(f"  All {num_trials} trials PASS for M in {M_sizes}")
    return all_pass


def test_softmax_batch_invariance(N, M_sizes, num_trials=10):
    """Test if softmax of row 0 changes with different M."""
    print(f"\n--- torch.softmax batch invariance: N={N} ---")

    all_pass = True
    for trial in range(num_trials):
        x = torch.randn(1, N, device=device, dtype=dtype)
        result_single = torch.softmax(x, dim=-1)

        for M in M_sizes:
            pad = torch.randn(M - 1, N, device=device, dtype=dtype)
            big_input = torch.cat([x, pad], dim=0)
            result_big = torch.softmax(big_input, dim=-1)
            row0 = result_big[0:1]

            diff = (result_single - row0).abs().max().item()
            match = diff == 0.0
            if not match:
                all_pass = False
                print(f"  trial={trial}, M={M}: FAIL (max diff = {diff:.6e})")

    if all_pass:
        print(f"  All {num_trials} trials PASS for M in {M_sizes}")
    return all_pass


def test_mean_batch_invariance(N, M_sizes, num_trials=10):
    """Test if mean of row 0 changes with different M (reduce along last dim)."""
    print(f"\n--- torch.mean batch invariance: N={N} ---")

    all_pass = True
    for trial in range(num_trials):
        x = torch.randn(1, N, device=device, dtype=torch.float32)
        result_single = torch.mean(x, dim=-1)

        for M in M_sizes:
            pad = torch.randn(M - 1, N, device=device, dtype=torch.float32)
            big_input = torch.cat([x, pad], dim=0)
            result_big = torch.mean(big_input, dim=-1)
            row0 = result_big[0:1]

            diff = (result_single - row0).abs().max().item()
            match = diff == 0.0
            if not match:
                all_pass = False
                print(f"  trial={trial}, M={M}: FAIL (max diff = {diff:.6e})")

    if all_pass:
        print(f"  All {num_trials} trials PASS for M in {M_sizes}")
    return all_pass


M_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
results = {}

results["mm (4096x4096)"] = test_matmul_batch_invariance(4096, 4096, M_sizes)
results["mm (2048x8192)"] = test_matmul_batch_invariance(2048, 8192, M_sizes)
results["mm (512x512)"] = test_matmul_batch_invariance(512, 512, M_sizes)
results["bmm (8xKxN)"] = test_bmm_batch_invariance(8, 1024, 512, M_sizes)
results["softmax (4096)"] = test_softmax_batch_invariance(4096, M_sizes)
results["mean (4096)"] = test_mean_batch_invariance(4096, M_sizes)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, passed in results.items():
    status = "PASS (batch invariant)" if passed else "FAIL (NOT batch invariant)"
    print(f"  {name}: {status}")
