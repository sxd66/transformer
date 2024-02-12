import numpy as np          # 导入NumPy库，用于进行矩阵运算和数据处理
import torch                # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn       # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time          # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt      # 导入Matplotlib的pyplot模块，用于绘制图表和可视化


# 定义一个名为Embeddings的类，继承自PyTorch的nn.Module类
class Embeddings(nn.Module):
    def __init__(self,d_model,vocal):
        super(Embeddings,self).__init__()

        self.lut=nn.Embedding(vocal,d_model)
        self.d_model=d_model
    def forward(self,x):

        return self.lut(x)*math.sqrt(self.d_model)


