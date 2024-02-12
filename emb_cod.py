import numpy as np          # 导入NumPy库，用于进行矩阵运算和数据处理
import torch                # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn       # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time          # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt      # 导入Matplotlib的pyplot模块，用于绘制图表和可视化
from embeding import Embeddings
class Position(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(Position, self).__init__()
        self.dropout=nn.Dropout(dropout)
        posit=torch.arange(1,max_len+1).unsqueeze(1)
        pe=torch.zeros(max_len,d_model)
        div=torch.exp(torch.arange(1,d_model,2)*-(math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(posit*div)
        pe[:, 1::2] = torch.cos(posit * div)
        pe=pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self,x):

        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)

emb=Embeddings(50,100)

var=torch.LongTensor([[2,4,6],[2,7,9,]])

k=emb.forward(var)
print(k.shape)
print(k)
y=Position(50,0.1,400)
w=y.forward(k)
print(w.shape)
print(w)

