import numpy as np          # 导入NumPy库，用于进行矩阵运算和数据处理
import torch                # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn       # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time          # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt
from layernorm import LayerNorm
from multhead import Muilt_head


class Subcon(nn.Module):
    def __init__(self,d_model,dropout=0.1):
        super(Subcon, self).__init__()

        self.norm=LayerNorm(d_model)
        self.drop=nn.Dropout(dropout)
        self.d_model=d_model

    def forward(self,sublayer,x):


        return x+self.drop(sublayer(self.norm(x)))

sxd=torch.ones(4,5,20)

s_layer=Muilt_head(5,20,0.1)
sublayer= lambda x: s_layer.forward(x,x,x)

kx=Subcon(20,0.1)
final=kx.forward(sublayer,sxd)

print(final)
print(final.shape)




