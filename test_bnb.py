import torch
import bitsandbytes as bnb

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

try:
    print("Testing bitsandbytes linear layer...")
    layer = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False, threshold=6.0)
    layer = layer.to("cuda")
    x = torch.randn(10, 10).to("cuda")
    y = layer(x)
    print("✅ Bitsandbytes linear layer works!")
except Exception as e:
    print(f"❌ Bitsandbytes failed: {e}")
