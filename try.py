import torch

print(torch.version.cuda)

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"GPU is available! Using device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Falling back to CPU.")
