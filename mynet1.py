import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
import numpy as np
import argparse
from einops.layers.torch import Rearrange

def _weights_init(m):
    if isinstance(m, nn.Linear): 
        init.xavier_uniform_(m.weight) #torch.nn.init.xavier_uniform_(tensor, gain=1) 均匀分布 ~  U(−a,a)
        torch.nn.init.normal_(m.bias, std=1e-6)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.up_proj = Rearrange(' b (h w) (c r)->b c r h w',c=8, r=8, h=3, w=3)
        
        self.conv3d_spatial_features = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(8),
            nn.Dropout(0.1)
        )

        self.conv3d_spectral_features = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True),
            nn.BatchNorm3d(8),
            nn.Dropout(0.1)
        )

        self.conv3d_spatial_features1 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(8),
            nn.Dropout(0.1)
        )

        self.conv3d_spectral_features1 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True),
            nn.BatchNorm3d(8),
            nn.Dropout(0.1)
        )
        self.gelu = nn.GELU()
        self.down_proj = Rearrange('b c r h w -> b (h w) (c r)',c=8, r=8, h=3, w=3)


    def forward(self, x):
        x = self.up_proj(x)
        x_spatial = self.conv3d_spatial_features(x)  # x1 torch.Size([64, 16, 7, 36])
        x_spatial = x + x_spatial
        x_spatial = self.gelu(x_spatial)

        x_spectral = self.conv3d_spectral_features(x_spatial)  # torch.Size([64, 16, 42, 6])
        x_spectral = x_spatial + x_spectral
        x_spectral = self.gelu(x_spectral)

        x_spatial1 = self.conv3d_spatial_features1(x_spectral)  # x1 torch.Size([64, 16, 7, 36])
        x_spatial1 = x_spectral + x_spatial1
        x_spatial1 = self.gelu(x_spatial1)

        x_spectral1 = self.conv3d_spectral_features(x_spatial1)  # torch.Size([64, 16, 42, 6])
        x_spectral1 = x_spatial1 + x_spectral1
        x_spectral1 = self.gelu(x_spectral1)
        
        x = self.down_proj(x_spectral1)
        return x



class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)    缩放操作
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x):

        b, n, _, h = *x.shape, self.heads  #获得输入x的维度和多头注意力头的个数，x: [batch_size,patch_num,dim]
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3,.chunk功能：将数据拆分为特定数量的块
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions，对qkv的维度进行调整。 #map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  ## 爱因斯坦求和约定（einsum）,隐含语义：当bhi,bhj固定时，得到两个长度为d的向量，元素相乘，并对d长度求和
        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block #out torch.Size([64, 8, 5, 8])
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):  #nn.Module是在pytorch使用非常广泛的类，搭建网络基本都需要用到这个。
    def __init__(self, dim, depth, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])  #专门用于存储module的list。
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block()))
            ]))

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)  # go to attention
            cls_tokens = x[:, 0]  #cls_tokens torch.Size([64, 64])
            x = mlp(x[:, 1:])  #x1 torch.Size([64, 4, 64])
            x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        return x


class Sstokenizer(nn.Module):  # nn.Module是在pytorch使用非常广泛的类，搭建网络基本都需要用到这个。
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv3d_features1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=6, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=True),
            nn.BatchNorm3d(6),
            nn.ReLU(),
            nn.Dropout(0.1)
        )  # nn.Sequential这是一个有顺序的容器,将特定神经网络模块按照在传入构造器的顺序依次被添加到计算图中执行。

        self.conv3d_features2 = nn.Sequential(
            nn.Conv3d(in_channels=6, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Dropout(0.1)

        )  # nn.Sequential这是一个有顺序的容器,将特定神经网络模块按照在传入构造器的顺序依次被添加到计算图中执行。
    
        self.conv3d_spatial_features1 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(1,3, 3), padding=(0,1,1), bias=True),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(1,3, 3), padding=(0,1,1), bias=True),
            nn.BatchNorm3d(8),
            nn.Dropout(0.1)
        )

        self.conv3d_spectral_features1 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3,1, 1), padding=(1,0,0), bias=True),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 1,1), padding=(1,0,0), bias=True),
            nn.BatchNorm3d(8),
            nn.Dropout(0.1)
        )

        self.conv3d_spatial_features2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(8),
            nn.Dropout(0.1)
        )
        self.conv3d_spectral_features2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True),
            nn.BatchNorm3d(8),
            nn.Dropout(0.1)
        )
       

        self.globel_pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool3d((8,3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
           )

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
      
    def forward(self, x):
        x = self.conv3d_features1(x)
        x = self.conv3d_features2(x)
        
         #######################################spatial_branch1###########################
        x_spatial = self.conv3d_spatial_features1(x)  # x1 torch.Size([64, 16, 7, 36])
        x_spatial = x + x_spatial
        x_spatial= self.gelu ( x_spatial)
        ######################################spectral_branch1###########################
        x_spectral = self.conv3d_spectral_features1(x_spatial)  # torch.Size([64, 16, 42, 6])
        x_spectral = x_spatial + x_spectral
        x_spectral= self.gelu (x_spectral)
 
        ##########################################avg_pool################################
        #x_pool = self.avg_pool_layer(x_spectral)


        #######################################spatial_branch2###########################
        x_spatial1 = self.conv3d_spatial_features2(x_spectral)  # x1 torch.Size([64, 16, 7, 36])
        x_spatial1 = x_spectral + x_spatial1
        x_spatial1 = self.gelu(x_spatial1)
#        ######################################spectral_branch2###########################
        x_spectral1 = self.conv3d_spectral_features2(x_spatial1)  # torch.Size([64, 16, 42, 6])
        x_spectral1 = x_spatial1 + x_spectral1
        x_spectral1 = self.gelu(x_spectral1)
    
        x = self.globel_pool_layer(x_spectral1)

        x = rearrange(x, 'b c r h w -> b  (h w) (c r)')

        return x

class mynet(nn.Module):
    def __init__(self, num_classes=16, num_tokens=4, dim=64, depth=1, heads=8,  dropout=0.1, emb_dropout=0.1):
        super(mynet, self).__init__()
        # self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        # torch.nn.init.normal_(self.pos_embedding, std=.02)   #服从~ N(mean,std)


        self.sstokenizer = Sstokenizer(in_channels=1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dropout)

        self.to_cls_token = nn.Identity() #不区分参数的占位符标识运算符,输入是啥，直接给输出，不做任何的改变

        self.nn1 = nn.Linear(dim, num_classes)

    def forward(self, x):
        x =  self.sstokenizer(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  #[64,1,64]其将单个维度扩大成更大维度，返回一个新的tensor,在expand中的-1表示取当前所在维度的尺寸，也就是表示当前维度不变。
        x = torch.cat((cls_tokens, x), dim=1)   #[64,5,64]
        #x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)  # main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x


if __name__ == '__main__':
    model = mynet()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 15, 13, 13)
    y = model(input).apply(_weights_init)
    print(y.size())

