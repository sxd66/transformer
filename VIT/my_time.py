from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn



class attention(nn.Module):
    def __init__(self,
                 dim=768,
                 head=8,

                dropout=0.


                 ):
        super(attention, self).__init__()
        self.qkv=nn.Linear(dim,dim*3)
        self.head=head
        self.head_dim=dim//head
        self.scale=self.head_dim**-0.5
        self.proj=nn.Linear(dim,dim)

    def forward(self,x):
        B,N,C=x.shape
        temp=self.qkv(x).reshape(B,N,3,self.head,self.head_dim).permute(2,0,3,1,4)
        q,k,v=temp[0],temp[1],temp[2]
        k=k.transpose(-1,-2)
        atten= ( q@k)*self.scale
        atten=atten.softmax(dim=-1)
        x=atten@(v)
        xx=x.transpose(1,2).reshape(B,N,C)
        xx2=self.proj(xx)
        return xx2



















