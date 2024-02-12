import numpy as np          # 导入NumPy库，用于进行矩阵运算和数据处理
import torch                # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn       # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time          # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt



def mask(size):

    k=np.triu(np.ones((1,size,size)),k=1).astype("uint8")
    return torch.from_numpy(1-k)

x=mask(4)
print(x)





