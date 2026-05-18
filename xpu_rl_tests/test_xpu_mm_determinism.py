"""Test whether torch.mm on XPU is batch-invariant (bit-exact across batch sizes).

Batch invariance means: the result for a given input row is the same
regardless of what other rows are in the batch.
"""
import torch

device = "xpu"


def check_batch_invariance(name, K, N, dtype, batch_sizes=(1, 2, 4, 8, 16, 32, 64)):
    """Check if mm result for a fixed row is identical across different batch sizes."""
    # Fixed input row and weight matrix
    fixed_row = torch.randn(1, K, device=device, dtype=dtype)
    weight = torch.randn(K, N, device=device, dtype=dtype)

    # Reference: compute mm with just the fixed row
    ref = torch.mm(fixed_row, weight)

    mismatches = []
    for bs in batch_sizes:
        # Pad with random rows to simulate different batch compositions
        padding = torch.randn(bs - 1, K, device=device, dtype=dtype)
        batch = torch.cat([fixed_row, padding], dim=0)
        result = torch.mm(batch, weight)
        # Extract result for the first row (our fixed input)
        row_result = result[0:1]
        if not torch.equal(ref, row_result):
            diff = (ref - row_result).abs().max().item()
            mismatches.append((bs, diff))

    if not mismatches:
        print(f"  {name}: BATCH-INVARIANT (tested {len(batch_sizes)} batch sizes)")
        return True
    else:
        print(f"  {name}: NOT BATCH-INVARIANT")
        for bs, diff in mismatches:
            print(f"    batch_size={bs}: max_diff={diff:.6e}")
        return False


def check_bmm_batch_invariance(name, M, K, N, dtype,
                               batch_sizes=(1, 2, 4, 8, 16)):
    """Check if bmm result for a fixed batch element is identical across sizes."""
    # Fixed single batch element
    fixed_a = torch.randn(1, M, K, device=device, dtype=dtype)
    fixed_b = torch.randn(1, K, N, device=device, dtype=dtype)

    # Reference: bmm with batch=1
    ref = torch.bmm(fixed_a, fixed_b)

    mismatches = []
    for bs in batch_sizes:
        if bs == 1:
            continue
        pad_a = torch.randn(bs - 1, M, K, device=device, dtype=dtype)
        pad_b = torch.randn(bs - 1, K, N, device=device, dtype=dtype)
        batch_a = torch.cat([fixed_a, pad_a], dim=0)
        batch_b = torch.cat([fixed_b, pad_b], dim=0)
        result = torch.bmm(batch_a, batch_b)
        row_result = result[0:1]
        if not torch.equal(ref, row_result):
            diff = (ref - row_result).abs().max().item()
            mismatches.append((bs, diff))

    if not mismatches:
        print(f"  {name}: BATCH-INVARIANT (tested {len(batch_sizes)} batch sizes)")
        return True
    else:
        print(f"  {name}: NOT BATCH-INVARIANT")
        for bs, diff in mismatches:
            print(f"    batch_size={bs}: max_diff={diff:.6e}")
        return False


def check_linear_batch_invariance(name, K, N, dtype,
                                  batch_sizes=(1, 2, 4, 8, 16, 32, 64)):
    """Check if linear result for a fixed row is identical across batch sizes."""
    fixed_row = torch.randn(1, K, device=device, dtype=dtype)
    weight = torch.randn(N, K, device=device, dtype=dtype)

    ref = torch.nn.functional.linear(fixed_row, weight)

    mismatches = []
    for bs in batch_sizes:
        padding = torch.randn(bs - 1, K, device=device, dtype=dtype)
        batch = torch.cat([fixed_row, padding], dim=0)
        result = torch.nn.functional.linear(batch, weight)
        row_result = result[0:1]
        if not torch.equal(ref, row_result):
            diff = (ref - row_result).abs().max().item()
            mismatches.append((bs, diff))

    if not mismatches:
        print(f"  {name}: BATCH-INVARIANT (tested {len(batch_sizes)} batch sizes)")
        return True
    else:
        print(f"  {name}: NOT BATCH-INVARIANT")
        for bs, diff in mismatches:
            print(f"    batch_size={bs}: max_diff={diff:.6e}")
        return False


print("=" * 60)
print("Testing batch invariance of torch.mm/bmm/linear on XPU")
print("(Does the result for a fixed input change with batch size?)")
print("=" * 60)

all_pass = True

for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    print(f"\ndtype = {dtype}")

    # --- torch.mm batch invariance ---
    for K, N in [(4096, 4096), (1024, 4096), (4096, 1024)]:
        ok = check_batch_invariance(f"mm [?x{K}] x [{K}x{N}]", K, N, dtype)
        all_pass = all_pass and ok

    # --- torch.bmm batch invariance ---
    for M, K, N in [(128, 256, 128), (512, 1024, 512)]:
        ok = check_bmm_batch_invariance(f"bmm [?x{M}x{K}] x [?x{K}x{N}]",
                                        M, K, N, dtype)
        all_pass = all_pass and ok

    # --- torch.nn.functional.linear batch invariance ---
    for K, N in [(4096, 4096), (4096, 11008)]:
        ok = check_linear_batch_invariance(f"linear [?x{K}] x [{N}x{K}]^T",
                                           K, N, dtype)
        all_pass = all_pass and ok

print("\n" + "=" * 60)
if all_pass:
    print("RESULT: All ops are BATCH-INVARIANT on XPU")
    print("(Native torch.mm/linear do not need Triton overrides)")
else:
    print("RESULT: Some ops are NOT batch-invariant on XPU")
    print("(Triton persistent matmul overrides may be needed)")
print("=" * 60)
