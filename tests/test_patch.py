import torch

import TorchDevice

print("TorchDevice loaded as:", TorchDevice.__file__)
print("torch.device is:", torch.device)
print(torch.device('cpu:-1'))
