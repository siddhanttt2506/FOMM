import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
# Get GPU device name
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")