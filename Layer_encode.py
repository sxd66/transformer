import numpy as np          # 导入NumPy库，用于进行矩阵运算和数据处理
import torch                # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn       # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time          # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt
from subcon import Subcon
from  multhead import  Muilt_head,clones
from  link import  Link
from layernorm import LayerNorm
from embeding import  Embeddings
from output import Output
from emb_cod import Position

class Layer_encode(nn.Module):
    def __init__(self,d_model,atten,feedforward):
        super(Layer_encode, self).__init__()
        self.d_model=d_model
        self.atten=atten
        self.link=feedforward
        self.layer1=clones(Subcon(d_model,0.1),2)
    def forward(self,x,mask=None):

        x=self.layer1[0]( lambda x: self.atten(x,x,x,mask),x)
        x=self.layer1[1](lambda  x: self.link(x),x)

        return x



class Encoder(nn.Module):
    def __init__(self,layer,N):

        super(Encoder, self).__init__()

        self.layers=clones(layer,N)
        self.d_model=layer.d_model
        self.norm=LayerNorm(self.d_model)


    def forward(self,x,mask=None):

        for layer in self.layers:
            x=layer.forward(x,mask)

        return self.norm(x)


"""
class Decoderlayer(nn.Module):
    def __init__(self,d_model,self_atten,sim_atten,feed):
        super(Decoderlayer, self).__init__()
        self.self_atten=self_atten
        self.feed=feed
        self.d_model=d_model
        self.sim_atten=sim_atten
        self.layers=clones(Subcon(self.d_model),3)

    def forward(self,x,memory,mask,sim_mask):
        x=self.layers[0](  lambda x:self.self_atten(x,x,x,mask),x   )
        x=self.layers[1](  lambda x:self.sim_atten(x,memory,memory,sim_mask),x   )
        x=self.layers[2](  self.feed,x  )

        return x


class Decoder(nn.Module):
    def __init__(self,layer,N):

        super(Decoder, self).__init__()

        self.layers=clones(layer,N)
        self.d_model=layer.d_model
        self.norm=LayerNorm(self.d_model)


    def forward(self,x,memery,mask=None,sim_mask=None):

        for layer in self.layers:
            x=layer.forward(x,memery,mask,sim_mask)

        return self.norm(x)


class Encoder_decoder(nn.Module):
    def __init__(self,encoder,decoder,tar_emb,sour_emb,output):
        super(Encoder_decoder, self).__init__()

        self.encoder=encoder
        self.decoder=decoder
        self.tar_emb=tar_emb
        self.sour_emb=sour_emb
        self.output=output


    def forward(self,sour,sour_msak,target,tar_msak):

        sour=self.encode(sour,sour_msak)
        target=self.decode(sour,sour_msak,target,tar_msak)

        return target

    def encode(self,sour,sour_msak):
        x=self.sour_emb(sour)
        return    self.encoder( x, sour_msak  )

    def decode(self,sour,sour_msak,target,tar_msak):



        return   self.decoder(  self.tar_emb( target),sour,tar_msak,sour_msak )


class Make_model(nn.Module):
    def __init__(self,sour_vocal,target_vocal,d_model,fft=5,N=3,head=5,dropout=0.1):
        super(Make_model, self).__init__()
        c=copy.deepcopy
        atten=Muilt_head(head,d_model,dropout)
        sim_atten=Muilt_head(head,d_model,dropout)
        feed=Link(d_model,fft,dropout)
        position=Position(d_model,dropout)
        self.model=Encoder_decoder(
            Encoder(Layer_encode(d_model,atten,feed),N),
            Decoder(Decoderlayer(d_model,atten,sim_atten,feed),N),
            Embeddings(d_model,target_vocal),
            Embeddings(d_model,sour_vocal),
            Output(d_model,target_vocal)

        )

















d_model=20
head=5
dropout=0.1
fft=5
sxd=Variable(torch.LongTensor([[10,10,11,1,2],[10,10,11,1,2],[10,10,11,1,2],[10,10,11,1,2]]))
N=3
mask=Variable(torch.ones(4,5,5))
vocal=50

embed=Embeddings(d_model,vocal)
self_atten=Muilt_head(head,d_model,dropout)
sim_atten=Muilt_head(head,d_model,dropout)
feed=Link(d_model,fft,dropout)
output=Output(d_model,vocal)

decoder_layer=Decoderlayer(d_model,self_atten,sim_atten,feed)
decoder=Decoder(decoder_layer,N)
encoder_layer=Layer_encode(d_model,self_atten,feed)
encoder=Encoder(encoder_layer,N)

kk=Encoder_decoder(encoder,decoder,embed,embed,output)
final=kk.forward(sxd,mask,sxd,mask)

print("final")

print(final.shape)

new=Make_model(50,50,20)

print(new)"""

class Dmodel(nn.Module):
    def __init__(self,d_model=11,fft=5,N=3,head=3,dropout=0.1):
        super(Dmodel, self).__init__()
        atten = Muilt_head(head, d_model, dropout)
        sim_atten = Muilt_head(head, d_model, dropout)
        feed = Link(d_model, fft, dropout)


        self.modu=nn.ModuleList(Encoder(Layer_encode(d_model,atten,feed),N),nn.Linear(d_model,2) )

    def forward(self,x):

        return self.modu(x)



