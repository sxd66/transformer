import numpy as np          # 导入NumPy库，用于进行矩阵运算和数据处理
import torch                # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn       # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time          # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt


class Link(nn.Module):
    def __init__(self,d_model,fft,dropout=0.1):
        super(Link, self).__init__()

        self.liner1=nn.Linear(d_model,fft)
        self.liner2=nn.Linear(fft,d_model)
        self.drop=nn.Dropout(dropout)

    def forward(self,x):
        return  self.liner2(self.drop(F.relu(self.liner1(x))))

query=torch.randn(2,5,20)

kk=Link(20,10,0.1)

q_new=kk(query)
print(q_new)
print(q_new.shape)

