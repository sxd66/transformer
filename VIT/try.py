import torch
import torchvision
from  torch import nn




a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([1, 2, 6, 7, 8])

common_elements = torch.sum(a == b)
print(common_elements)


