"""Test batch-invariant matmul, bmm, and softmax kernels on XPU."""
import time
import torch
import triton

# Import the Triton kernels and wrappers directly
from vllm.model_executor.layers.batch_invariant import (
    matmul_kernel_persistent,
    bmm_batch_invariant,
    softmax_batch_invariant,
)

device = "xpu"
dtype = torch.bfloat16

# Import tutorial kernels from intel-xpu-backend-for-triton
import importlib.util
_tutorial_path = (
    "intel-xpu-backend-for-triton"
    "/python/tutorials/09-persistent-matmul.py"
)
_spec = importlib.util.spec_from_file_location("persistent_matmul", _tutorial_path)
tutorial = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tutorial)


# --- 1. softmax_batch_invariant (pure PyTorch, should trivially work) ---
print("=" * 60)
print("Testing softmax_batch_invariant...")
x = torch.randn(4, 1024, device=device, dtype=dtype)
result = softmax_batch_invariant(x, dim=-1)
ref = torch.nn.functional.softmax(x.float(), dim=-1).to(dtype)
diff = (result - ref).abs().max().item()
print(f"  max diff = {diff:.6e}")
assert diff < 1e-3, f"softmax diff too large: {diff}"
print("  PASS")


# --- 2. matmul_persistent (patched for XPU) ---
print("=" * 60)
print("Testing matmul_kernel_persistent...")

def _get_num_sms(dev):
    """Get EU count for device (cached, prints once)."""
    if not hasattr(_get_num_sms, "_cache"):
        try:
            props = torch.xpu.get_device_properties(dev)
            n = getattr(props, 'gpu_eu_count', None)
            if n is None:
                n = getattr(props, 'multi_processor_count', None)
            if n is None:
                n = 512
            print(f"  NUM_SMS (from device props): {n}")
        except Exception as e:
            n = 512
            print(f"  NUM_SMS (fallback): {n}, reason: {e}")
        _get_num_sms._cache = n
    return _get_num_sms._cache


def matmul_persistent_xpu(a, b, bias=None):
    """XPU-patched version of matmul_persistent."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    NUM_SMS = _get_num_sms(a.device)

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }

    matmul_kernel_persistent[grid](
        a, b, c,
        bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[a.dtype],
    )
    return c


# Test matmul correctness
M, K, N = 256, 512, 256
a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(K, N, device=device, dtype=dtype)

try:
    result = matmul_persistent_xpu(a, b)
    ref = torch.mm(a.float(), b.float()).to(dtype)
    diff = (result - ref).abs().max().item()
    print(f"  max diff = {diff:.6e}")
    # bf16 matmul can have larger diffs due to accumulation order
    assert diff < 0.2, f"diff too large: {diff}"
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")



# --- 2b. Tutorial: matmul_descriptor_persistent (2D block I/O, fast on XPU) ---
print("=" * 60)
print("Testing tutorial matmul_descriptor_persistent...")

M, K, N = 256, 512, 256
a = torch.randn(M, K, device=device, dtype=dtype)
# descriptor persistent expects b as [N, K] (transposed layout)
b_T = torch.randn(N, K, device=device, dtype=dtype)

try:
    result = tutorial.matmul_descriptor_persistent(a, b_T, warp_specialize=False)
    ref = torch.mm(a.float(), b_T.T.float()).to(dtype)
    diff = (result - ref).abs().max().item()
    print(f"  max diff = {diff:.6e}")
    assert diff < 0.2, f"matmul_descriptor_persistent diff too large: {diff}"
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")


# --- 2c. Tutorial: matmul_persistent (pointer-based, from tutorial) ---
# NOTE: may need TRITON_INTEL_PREDICATED_STORE=0 for pointer-based variants
print("=" * 60)
print("Testing tutorial matmul_persistent (pointer-based)...")

M, K, N = 256, 512, 256
a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(K, N, device=device, dtype=dtype)

try:
    result = tutorial.matmul_persistent(a, b)
    ref = torch.mm(a.float(), b.float()).to(dtype)
    diff = (result - ref).abs().max().item()
    print(f"  max diff = {diff:.6e}")
    assert diff < 0.2, f"tutorial matmul_persistent diff too large: {diff}"
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")


# --- 2d. Tutorial: matmul (naive tiled) ---
# NOTE: may need TRITON_INTEL_PREDICATED_STORE=0 for pointer-based variants
print("=" * 60)
print("Testing tutorial matmul (naive tiled)...")

M, K, N = 256, 512, 256
a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(K, N, device=device, dtype=dtype)

try:
    result = tutorial.matmul(a, b)
    ref = torch.mm(a.float(), b.float()).to(dtype)
    diff = (result - ref).abs().max().item()
    print(f"  max diff = {diff:.6e}")
    assert diff < 0.2, f"tutorial matmul diff too large: {diff}"
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")


# --- 3. bmm_batch_invariant ---
print("=" * 60)
print("Testing bmm_batch_invariant...")

B_size, M, K, N = 4, 128, 256, 128
a = torch.randn(B_size, M, K, device=device, dtype=dtype)
b = torch.randn(B_size, K, N, device=device, dtype=dtype)

try:
    result = bmm_batch_invariant(a, b)
    ref = torch.bmm(a.float(), b.float()).to(dtype)
    diff = (result - ref).abs().max().item()
    print(f"  max diff = {diff:.6e}")
    assert diff < 0.2, f"diff too large: {diff}"
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")


# --- 4. Performance comparison ---
print("=" * 60)
print("Performance comparison (10 iterations, warmup=3)")

def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms

M, K, N = 2048, 4096, 2048
a2d = torch.randn(M, K, device=device, dtype=dtype)
b2d = torch.randn(K, N, device=device, dtype=dtype)
# Matmul — compare all variants against torch.mm
t_torch = bench(lambda: torch.mm(a2d, b2d))
b2d_T = b2d.T.contiguous()  # [N, K] layout for descriptor persistent
print(f"matmul [{M}x{K}] x [{K}x{N}]:")
print(f"  torch.mm (baseline):            {t_torch:.2f} ms")

try:
    t = bench(lambda: matmul_persistent_xpu(a2d, b2d))
    print(f"  vllm persistent (ptr):          {t:.2f} ms  ({(t/t_torch - 1)*100:+.0f}%)")
except Exception as e:
    print(f"  vllm persistent (ptr):          SKIPPED ({e})")

try:
    t = bench(lambda: tutorial.matmul_descriptor_persistent(a2d, b2d_T, warp_specialize=False))
    print(f"  tutorial descriptor persistent:  {t:.2f} ms  ({(t/t_torch - 1)*100:+.0f}%)")
except Exception as e:
    print(f"  tutorial descriptor persistent:  SKIPPED ({e})")

try:
    t = bench(lambda: tutorial.matmul_persistent(a2d, b2d))
    print(f"  tutorial persistent (ptr):       {t:.2f} ms  ({(t/t_torch - 1)*100:+.0f}%)")
except Exception as e:
    print(f"  tutorial persistent (ptr):       SKIPPED ({e})")

try:
    t = bench(lambda: tutorial.matmul(a2d, b2d))
    print(f"  tutorial naive tiled:            {t:.2f} ms  ({(t/t_torch - 1)*100:+.0f}%)")
except Exception as e:
    print(f"  tutorial naive tiled:            SKIPPED ({e})")

# BMM
B_size = 8
a3d = torch.randn(B_size, 512, 1024, device=device, dtype=dtype)
b3d = torch.randn(B_size, 1024, 512, device=device, dtype=dtype)

try:
    t_triton = bench(lambda: bmm_batch_invariant(a3d, b3d))
    t_torch = bench(lambda: torch.bmm(a3d, b3d))
    print(f"bmm [{B_size}x512x1024] x [{B_size}x1024x512]:")
    print(f"  Triton bmm:  {t_triton:.2f} ms")
    print(f"  torch.bmm:   {t_torch:.2f} ms")
    print(f"  overhead:    {(t_triton/t_torch - 1)*100:+.1f}%")
except Exception as e:
    print(f"bmm perf: SKIPPED ({e})")

# Softmax
x_soft = torch.randn(4096, 4096, device=device, dtype=dtype)

t_triton = bench(lambda: softmax_batch_invariant(x_soft, dim=-1))
t_torch = bench(lambda: torch.softmax(x_soft, dim=-1))
print(f"softmax [4096x4096]:")
print(f"  batch_invariant: {t_triton:.2f} ms")
print(f"  torch.softmax:   {t_torch:.2f} ms")
print(f"  overhead:        {(t_triton/t_torch - 1)*100:+.1f}%")

print("=" * 60)
print("Done.")
