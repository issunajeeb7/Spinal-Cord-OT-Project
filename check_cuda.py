import torch

cuda_available = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
print(f"CUDA available: {cuda_available}, GPU name: {gpu_name}")
