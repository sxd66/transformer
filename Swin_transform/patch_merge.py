
import torch
import torchvision
from  torch import nn
from  torch.utils.data import DataLoader

class Patch_merge(nn.Module):
    def __init__(self,dim,resoult,norm=1):
        super(Patch_merge, self).__init__()
        self.dim=dim
        h,w=resoult
        self.H=h
        self.W=w
        self.norm= nn.LayerNorm(4*dim) if norm else  None
        self.reduction=nn.Linear(4*dim,2*dim)
    def forward(self,x):
        B,hw,C=x.shape
        assert hw==(self.H*self.W),  "图像长宽不匹配"
        assert ((self.H%2)==0 )and ((self.W%2)==0),"无法继续分片"

        x=x.reshape(B,self.H,self.W,C)
        x0=x[:,0::2,0::2,:]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x=torch.cat((x0,x1,x2,x3),dim=-1)
        x=x.reshape( B,-1,4*C  )
        x = self.norm(x)
        x=self.reduction(x)

        return x













patch=Patch_merge(10,(16,16),1)

x=torch.randn(12,256,10)
y=patch(x)
print(y.shape)

























