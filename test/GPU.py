import time
import torch

# large tensor on CPU
cpu_tensor = torch.rand(10000, 10000)
start = time.time()
cpu_result = cpu_tensor @ cpu_tensor  # matrix multiplication
print("CPU time:", time.time() - start)

# same operation on GPU
gpu_tensor = torch.rand(10000, 10000, device="cuda")
torch.cuda.synchronize()  # wait for any previous GPU work to finish
start = time.time()
gpu_result = gpu_tensor @ gpu_tensor
torch.cuda.synchronize()  # wait until operation finishes
print("GPU time:", time.time() - start)

print(f"Current GPU: {torch.cuda.current_device()}")       # prints GPU index (usually 0)
print(f"GPU name: {torch.cuda.get_device_name(0)}")     # prints GPU name
