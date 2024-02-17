from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from VIT.example import Block as Blk2

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)  # [224,224]
        patch_size = (patch_size, patch_size)  # [16,16]
        self.img_size = img_size  # [224,224]
        self.patch_size = patch_size  # [16,16]
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # [14,14]
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 196
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)  # 3，768,（16,16），16
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # proj:[B,3,224,224] -> [B,768,14,14]
        # flatten: [B, 768, 14, 14] -> [B, 768, 196]
        # transpose: [B, 768, 196] -> [B, 196, 768]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Pyrmid(nn.Module):
    def __init__(self,in_ch,out_ch,kernel=3,image_size=32):
        super(Pyrmid, self).__init__()
        self.norm=nn.LayerNorm(image_size//2)
        self.norm2=nn.LayerNorm(image_size//4)
        self.proj=nn.Conv2d(in_ch,out_ch*2,kernel,stride=2,padding=1)
        self.proj2=nn.Conv2d(in_ch,out_ch*2,1,2)
        self.proj3=nn.Conv2d(out_ch*2,out_ch*4,kernel,stride=2,padding=1)
        self.proj4=nn.Conv2d(out_ch*2,out_ch*4,1,2)
    def forward(self,x):
        x1=self.proj(x)+self.proj2(x)
        x1=self.norm(x1)
        x2=self.proj3(x1)+self.proj4(x1)
        x2=self.norm2(x2)
        return x,x1,x2

class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim 768
                 num_heads=8,  # multi-head 12
                 qkv_bias=False,  # True
                 qk_scale=None,  # 和根号dimk作用相同
                 attn_drop_ratio=0.,  # dropout率
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 768 // 12 = 64
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv经过一个linear得到。 768 --> 2304
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # 一次新的映射
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self,x):
        C,B,N,D=x.shape
        x=x.reshape(C,B,N,self.num_heads,D//self.num_heads).permute(0,1,3,2,4)

        q,k,v=x[0],x[1],x[2]
        atten=(q@k.transpose(-1,-2))*self.scale
        atten=atten.softmax(dim=-1)
        atten=self.attn_drop(atten)
        result=(atten@v).transpose(1,2).reshape(B,N,D)
        result=self.proj(result)
        result=self.proj_drop(result)

        return result

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)  # 与Transformer不同，这里先norm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 隐藏层神经元数
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)  # MLP

    def forward(self, x):
        # 残差连接
        x = x[0] + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Pyr_transform(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_c=3, num_classes=100,
                 embed_dim=48, depth=1, num_heads=8, mlp_ratio=2, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(Pyr_transform, self).__init__()
        self.num_classes = num_classes  # 1000
        self.num_features = self.embed_dim = embed_dim  # 768
        self.num_tokens =  1  # 不管
        self.patch=embed_layer(img_size,patch_size,in_c,embed_dim)
        self.blk=Block(embed_dim,num_heads,mlp_ratio,drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio)
        self.pyrmid=Pyrmid(3,3,3)
        self.head=nn.Linear(embed_dim,num_classes)

        num_patches = self.patch.num_patches
        self.tok_emb=nn.Parameter(torch.zeros(3,1,embed_dim))
        self.pos_emb=nn.Parameter(torch.zeros(1,num_patches+1,embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)  # 位置编码
        nn.init.trunc_normal_(self.tok_emb,std=0.02)
    def forward(self,x):

        y1, y2, y3 = self.pyrmid(x)
        y1, y2, y3 =self.patch(y1).unsqueeze(0),self.patch(y2).unsqueeze(0),self.patch(y3).unsqueeze(0)
        y = torch.cat((y1, y2, y3), dim=0)
        B=y.shape[1]
        tok_emb=self.tok_emb.unsqueeze(1).expand(3,B,-1,-1)
        y=torch.cat((tok_emb,y),dim=2)
        pos_emb=self.pos_emb.unsqueeze(0).expand(3,B,-1,-1)

        y=self.blk(y+pos_emb)
        y=self.head(y[:,0])
        return y

class Attention2(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim 768
                 num_heads=8,  # multi-head 12
                 qkv_bias=False,  # True
                 qk_scale=None,  # 和根号dimk作用相同
                 attn_drop_ratio=0.,  # dropout率
                 proj_drop_ratio=0.):
        super(Attention2, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 768 // 12 = 64
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv经过一个linear得到。 768 --> 2304
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # 一次新的映射
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 在Vit中qkv维度相同，都是[B,12,197,64]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 按行进行softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        # C就是total_embed_dim
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Blk(nn.Module):
    def __init__(self,inc=3,img_size=32, patch_size=4, embed_dim=48):
        super(Blk, self).__init__()
        self.pymid=Pyrmid(inc,inc,3)
        self.patch1 = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.patch2 = PatchEmbed(img_size=img_size//2, in_c=inc*2,patch_size= patch_size,embed_dim= embed_dim)
        self.patch3 = PatchEmbed(img_size=img_size//4, in_c=inc*4, patch_size=patch_size//2, embed_dim=embed_dim)

    def forward(self,x):
        y1, y2, y3 = self.pymid(x)
        y1, y2, y3 = self.patch1(y1), self.patch2(y2), self.patch3(y3)
        y = torch.cat((y1, y2, y3), dim=1)
        return y

class Rich_trans(nn.Module):
    def __init__(self,img_size=32, patch_size=4, in_c=3, num_classes=100,
                 embed_dim=48, depth=1, num_heads=8, mlp_ratio=2, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(Rich_trans, self).__init__()
        self.blk=Blk(inc=in_c,img_size=img_size,patch_size=patch_size,embed_dim=embed_dim)
        self.blk2=Blk2(embed_dim,num_heads=num_heads,mlp_ratio=2)
        dim=img_size//patch_size
        self.num=dim**2+(dim//2)**2+(dim//2)**2+1
        self.dim_pos=[ dim**2,(dim//2)**2,(dim//2)**2   ]
        self.tok=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_tok=nn.Parameter(torch.zeros(3,1,1,embed_dim))
        self.all_pos_tok=nn.Parameter(torch.zeros(1,self.num,embed_dim))
        self.head=nn.Linear(embed_dim,num_classes)

        nn.init.trunc_normal_(self.pos_tok, std=0.02)
        nn.init.trunc_normal_(self.all_pos_tok, std=0.02)
        nn.init.trunc_normal_(self.tok, std=0.02)

    def forward(self,x):
        x=self.blk(x)
        tok=self.tok.expand(x.shape[0],-1,-1)
        x=torch.cat((tok,x),dim=1)
        pos0=self.pos_tok[0].expand(x.shape[0],self.dim_pos[0]+1,-1)
        pos1=self.pos_tok[1].expand(x.shape[0],self.dim_pos[1],-1)
        pos2=self.pos_tok[2].expand(x.shape[0],self.dim_pos[2],-1)
        pos=torch.cat((pos0,pos1,pos2),dim=1)
        x=x+pos+self.all_pos_tok
        x=self.blk2(x)
        x=self.head(x[:,0])
        return x

if __name__ == "__main__":
    x=torch.randn(64,3,32,32)
    model=Rich_trans()
    y=model(x)
    print(y.shape)


    """
    pymid=Pyrmid(3,3,3)
    y1,y2,y3=pymid(x)

    patch1=PatchEmbed(img_size=32,patch_size=4,embed_dim=48)
    patch2=PatchEmbed(img_size=16,in_c=6,patch_size=4,embed_dim=48)
    patch3=PatchEmbed(img_size=8,in_c=12,patch_size=2,embed_dim=48)
    y1, y2, y3 =patch1(y1),patch2(y2),patch3(y3)
    y=torch.cat((y1,y2,y3),dim=1)
    att=Attention2(48,8)
    y=att(y)
    print(y.shape)
    """
    """
    pyr=Pyr_transform()
    kk=pyr(x)
    print(kk.shape)
    """
    """
    patch=PatchEmbed(img_size=32,
                              patch_size=4,
                              embed_dim=48)
    pymid=Pyrmid(3,3,3)
    y1,y2,y3=pymid(x)
    y1, y2, y3=patch(y1).unsqueeze(0),patch(y2).unsqueeze(0),patch(y3).unsqueeze(0)
    y=torch.cat((y1,y2,y3),dim=0)
    blk=Block(48,8,2)
    kk=blk(y)
    print(kk.shape)
    """
    """
    kk=torch.cat((y1,y2,y3),dim=0)
    print(kk.shape)

    att=Attention(48,8)
    kk=att(kk)
    print(kk.shape)
    mlp=Mlp(48,96,48)
    kk=mlp(kk)
    print(kk.shape)
    """








