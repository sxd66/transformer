import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

from example import  Mlp,DropPath,window_partition,window_reverse

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads,  attn_drop=0., proj_drop=0.):

        super().__init__()

        self.dim=dim
        self.window_size=window_size
        self.num_haed=num_heads
        head_dim=dim//num_heads
        self.scale=head_dim**-0.5
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj_drop=nn.Dropout(proj_drop)
        self.proj=nn.Linear(dim,dim)
        self.qkv=nn.Linear(dim,3*dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # meshgrid生成网格，再通过stack方法拼接
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(
            1)  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        # 整个训练当中，window_size大小不变，因此这个索引也不会改变
        self.register_buffer("relative_position_index", relative_position_index)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x,mask=None):
        B,N,C=x.shape
        x=self.qkv(x).reshape(B,N,3,self.num_haed,-1).permute(2,0,3,1,4)
        q,k,v=x.unbind(0)

        attn=q@k.transpose(-1,-2)*self.scale
        temp=self.relative_position_index.view(-1)
        relative_position_bias = self.relative_position_bias_table[temp].reshape(N,N,self.num_haed)
        relative_position_bias=relative_position_bias.permute(2,0,1).contiguous().unsqueeze(0)
        attn=attn+relative_position_bias
        attn=self.softmax(attn)

        attn=self.attn_drop(attn)
        attn=(attn@v).permute(0,2,1,3).reshape(B,N,C)
        attn=self.proj(attn)
        attn=self.proj_drop(attn)

        return attn

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    # 与Vit的block结构是相同的
    def __init__(self, dim, num_heads,reso, window_size=7, shift_size=0,
                 mlp_ratio=4.,  drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.resoultion=reso
    def forward(self, x, attn_mask=None):
        B,N,C=x.shape
        H,W=self.resoultion
        assert H*W==N, "wrong with size"
        shortcut=x
        x=self.norm1(x)
        x=x.view(B,H,W,C)

        if self.shift_size>0:
            x=torch.roll( x,(-self.shift_size,-self.shift_size),(-1,-2)  )
        else:
            x=x

        x=window_partition(x,self.window_size)
        x=x.view(-1,self.window_size*self.window_size,C)
        x=self.attn(x)

        x=x.view(-1,self.window_size,self.window_size,C)
        x=window_reverse(x,self.window_size,)


        if self.shift_size>0:
            x=torch.roll( x,(self.shift_size,self.shift_size),(-1,-2)  )
        else:
            x=x

        x=x+shortcut

        shortcut=x
        x=self.mlp()






model=WindowAttention(16,(5,5),4)
x=torch.randn(10,25,16)
y=model(x)
print(y.shape)








