import numpy as np          # 导入NumPy库，用于进行矩阵运算和数据处理
import torch                # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn       # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time          # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt



class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a=nn.Parameter( torch.ones(d_model))
        self.b=nn.Parameter(torch.zeros(d_model))
        self.eps=eps
    def forward(self,x):

        mean=x.mean(-1,keepdim=True)
        std=x.std(-1, keepdim=True)
        return self.a*(x-mean)/(std+self.eps)+self.b

kk=LayerNorm(20)

x=torch.ones(4,5,20)

w=kk(x)

print(w)
print(w.shape)
