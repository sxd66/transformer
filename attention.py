import numpy as np          # 导入NumPy库，用于进行矩阵运算和数据处理
import torch                # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn       # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time          # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt
from mask import mask
def attention(query,key,value,mask=None,drop=None):
    d_model=query.size(-1)
    key_ver=key.transpose(-2,-1)
    temp=torch.matmul( query,key_ver/math.sqrt(d_model))

    if mask is not None:
        temp=temp.masked_fill(mask==0,1e-9)

    patten=F.softmax(temp,dim=-1 )
    if drop is not None:
        patten=drop(patten)

    return torch.matmul(patten,value),patten

query=torch.randn(2,5,4,3)
mas=mask(4)
x,y=attention(query,query,query,mas)
print(x.shape)
print(y.shape)






