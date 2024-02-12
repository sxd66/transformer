import numpy as np          # 导入NumPy库，用于进行矩阵运算和数据处理
import torch                # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn       # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time          # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt
from  attention import attention
def clones(module,N):


    return nn.ModuleList([ copy.deepcopy(module)  for _ in range(N)  ])


class Muilt_head(nn.Module):
    def __init__(self,head,emb_dim,dropout=0.1):
        super(Muilt_head, self).__init__()
        assert emb_dim/head
        self.head=head
        self.emb_dim=emb_dim
        self.d_k=emb_dim//head
        self.drop=nn.Dropout(dropout)
        x=nn.Linear(emb_dim,emb_dim)
        self.Liner=clones(x,4)
        self.atten=None

    def forward(self,Q,K,V,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(1)

        batch_size=Q.size(0)
        Q,K,V=\
        [    model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2)
            for model,x in zip( self.Liner,(Q,K,V) )
        ]

        x,atten=attention(Q,K,V,mask,self.drop)
        x=x.transpose(1,2).contiguous().view(batch_size,-1,self.emb_dim)
        return self.Liner[-1](x)

query=torch.randn(2,5,20)


Mu_head=Muilt_head(5,20,0.1)

ww=Mu_head.forward(query,query,query)

print(ww.shape)





























