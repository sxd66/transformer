from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

class Pachembed(nn.Module):
    def __init__(self,patch_size=16,image_size=224,in_c=3,embding=768,layernorm=None):
        super(Pachembed, self).__init__()

        self.patch_size=patch_size
        self.image_size=image_size
        self.embding=embding
        self.grid_size=image_size/patch_size
        self.grid_num=self.grid_size*self.grid_size
        self.conv=nn.Conv2d(in_c,embding,(patch_size,patch_size),patch_size)
        self.layer_norm= nn.LayerNorm(embding) if layernorm else nn.Identity()
    def forward(self,x):
        B,C,H,W=x.shape
        assert H==self.image_size and W==self.image_size,f"图像宽和高{H}和模型{self.image_size}不匹配"
        x=self.conv(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        x=self.layer_norm(x)
        return x

class Muilt_head(nn.Module):
    def __init__(self, dim=768,head=12,atten_drop=0.1,proj_drop=0.1):
        super(Muilt_head, self).__init__()
        self.dim=dim
        self.head=head
        self.head_dim=dim/head
        self.atten_drop=nn.Dropout(atten_drop)
        self.proj_drop=nn.Dropout(proj_drop)
        self.qkv=nn.Linear(dim,3*dim)
        self.proj=nn.Linear(dim,dim)
        self.dk=self.head_dim ** -0.5
    def forward(self,x):
        B,C,H=x.shape
        x=self.qkv(x).reshape(B,C,3,self.head,-1).permute(2,0,3,1,4)

        q,k,v=x[0],x[1],x[2]

        result=(q@k.transpose(-1,-2))*self.dk
        result=result.softmax(dim=-1)
        result=self.atten_drop(result)

        result=(result@v).permute(0,2,1,3).reshape(B,C,-1)

        result=self.proj(result)
        result = self.proj_drop(result)

        return result


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features  # 768
        hidden_features = hidden_features or in_features  # 3072
        self.fc1 = nn.Linear(in_features, hidden_features)  # 768 --> 3072
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  # 3072 --> 768
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Block(nn.Module):
    def __init__(self,dim=768,head=12,atten_drop=0.1,proj_drop=0.1,in_features=768, hidden_features=3000, out_features=768, act_layer=nn.GELU, mlp_drop=0.1,drop=0.1):
        super(Block, self).__init__()
        self.mlp=Mlp(in_features,hidden_features,out_features,act_layer,mlp_drop)
        self.muilt_head=Muilt_head(dim,head,atten_drop,proj_drop)
        self.norm=nn.LayerNorm(dim)
        self.dropout=nn.Dropout(drop)
        self.Mutil_atten=nn.Sequential(
            self.norm,
            self.muilt_head,
            self.dropout
        )
        self.Mlp_layer=nn.Sequential(
            self.norm,
            self.mlp,
            self.dropout
        )

    def forward(self,x):

        x1=self.Mutil_atten(x)+x
        x2=self.Mlp_layer(x1)+x1
        return x2










model=Block()
x=torch.randn(5,196,768)
y=model(x)
print(y.shape)





