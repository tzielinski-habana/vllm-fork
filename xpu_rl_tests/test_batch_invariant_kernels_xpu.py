import torch
# Import the kernels directly from the file
from vllm.model_executor.layers.batch_invariant import (
    rms_norm, mean_dim, log_softmax
)

device = "xpu"
dtype = torch.bfloat16

# --- rms_norm ---
x = torch.randn(4, 1024, device=device, dtype=dtype)
w = torch.randn(1024, device=device, dtype=dtype)
result = rms_norm(x, w, eps=1e-6)
# Reference: manual PyTorch computation
x_f32 = x.float()
rms = (x_f32 ** 2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
ref = (x_f32 / rms * w.float()).to(dtype)
print(f"rms_norm: max diff = {(result - ref).abs().max().item():.6e}")

# --- log_softmax ---
x = torch.randn(4, 1024, device=device, dtype=dtype)
result = log_softmax(x, dim=-1)
ref = torch.nn.functional.log_softmax(x.float(), dim=-1).to(dtype)
print(f"log_softmax: max diff = {(result - ref).abs().max().item():.6e}")

# --- mean_dim ---
x = torch.randn(4, 8, 1024, device=device, dtype=torch.float32)
result = mean_dim(x, dim=1)
ref = x.mean(dim=1)
print(f"mean_dim: max diff = {(result - ref).abs().max().item():.6e}")

print("All tests passed!" if True else "")
