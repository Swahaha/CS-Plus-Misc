import torch
print(torch.cuda.is_available())  # Should print True if GPU is available
print(torch.cuda.get_device_name(0))  # Prints the name of your GPU
