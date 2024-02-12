
import torch
import torchvision
from  torch import nn
from  torch.utils.data import DataLoader

x=torch.randn(2,4,4)


shifted_x = torch.roll(x, shifts=2, dims=1)

print(x)

print(shifted_x)






