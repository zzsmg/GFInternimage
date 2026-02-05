import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import numpy as np
import torch.nn as nn

import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
from .modules.EFF import EFF
import math
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import math
from timm.models.layers import DropPath, trunc_normal_

from IPython import embed


from .modules.EMCAD import EUCB
from .modules.MFAM import MFAM
from .modules.LKS import LKS


up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x





class Mlp(nn.Module):
    """Mlp implemented by with 1*1 convolutions.
    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 #  act_cfg=dict(type='GELU'),
                 act_layer=nn.GELU,
                 drop_path=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        #self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        # self.act = build_activation_layer(act_cfg)
        self.act = act_layer()
        self.dwconv = DWConv(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop_path)
        # self.sconv = SeparableConv2d(hidden_features,hidden_features)
    
    
   
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.sconv(x)
        # print('以应用')
        x = self.fc2(x)
        x = self.drop(x)

        x = x.flatten(2).transpose(1, 2)
        return x


class ChannelReductionAttention(nn.Module):
    def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        # self.dim2 = dim2
        self.pool_ratio = pool_ratio
        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.pool1 = nn.AvgPool2d(1, 1)
        self.pool2 = nn.AvgPool2d(5, 5)
        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim1)
        self.act = nn.GELU()
        self.apply(self._init_weights)
        # self.glsa = GLSA(dim1, dim1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, h, w):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        
        x_ = self.norm(x_)
        x_ = self.act(x_)

        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes*2, out_planes*2,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes*2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




from einops import rearrange, repeat


class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = x_1 * x_2

        x = self.linear2(x)
        # x = self.eca(x)

        return x

class global_meta_block(nn.Module):

    def __init__(self, dim1, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm3 = norm_layer(dim1)

        self.attn = ChannelReductionAttention(dim1=dim1, num_heads=num_heads, pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_path=drop_path)

        self.apply(self._init_weights)
        self.frfn = FRFN(dim1)
       
        # self.dcv  = DCNv4(dim1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, h, w):


        x = x + self.drop_path(self.attn(self.norm1(x), h, w))


        x = x + self.drop_path(self.mlp(self.norm3(x), h, w))
        #x = x + self.drop_path(self.frfn(self.norm3(x)))
        
        
        return x

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=True):
        super(DSConv, self).__init__()
        self.body = nn.Sequential(
                        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, groups=in_channels, bias=bias),
                        # 1x3
                        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, groups=in_channels, bias=bias),
                        # PointWise Conv
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
                    )

    def forward(self, x):
        return self.body(x)
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function
class _DenseLayer(nn.Module):
    def __init__(self, layer_i, num_input_features, growth_rate, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, growth_rate,
                        kernel_size=1, stride=1, bias=False)),

        # self.add_module('norm2', nn.BatchNorm2d(growth_rate)),
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        # self.add_module('conv2', nn.Conv2d(growth_rate, growth_rate,
        #                 kernel_size=3, stride=1, padding=1, bias=False)),

        self.add_module('norm2', nn.BatchNorm2d(growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', DSConv(growth_rate, growth_rate,
                        kernel_size=3, stride=1, dilation=1+2*layer_i, padding=1+2*layer_i, bias=False)),

        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)  # 输入BCL
        if self.efficient:
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features
class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate, efficient=False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                layer_i=i,
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):  # 输入BCHW 输出B(num_layers*C)HW
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)
class DSPP1(nn.Module):  # BLC BCHW
    def __init__(self, channels):
        super(DSPP1, self).__init__()
        # conv1:BCHW conv2:BLC conv3:BCHW
        self.channels = channels
        self.block1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels, 1, 1),
            nn.BatchNorm2d(channels),  # 用batch时,epoch>1
            nn.ReLU()
        )
        self.block3 = DenseBlock(num_layers=5, num_input_features=112, growth_rate=112, drop_rate=0, efficient=False)
        self.block4 = nn.Sequential(
            nn.Conv2d(channels * 7, channels, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):  # BCHW
        x1 = self.block1(x)
        x3 = self.block3(x)
        x3 = x3[:, self.channels:, :, :]

        size = x.shape[2:]
        x2 = self.block2(x)
        x2 = F.upsample(x2, size=size, mode='bilinear')

        y = torch.cat([x1, x2, x3], dim=1)  # B(7*C)HW
        y = self.block4(y)
        return y
import torch.nn.functional as F





@HEADS.register_module()
class AYHead(BaseDecodeHead):

    def __init__(self, feature_strides, **kwargs):
        super(AYHead, self).__init__(input_transform='multiple_select', **kwargs)
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = dict(embed_dim=(768))
        embedding_dim = decoder_params['embed_dim']
        pool_ratio = 2
        mlp_ratio = 2
        mutil_channel = [64, 128, 320, 512]
        # self.attn4 = global_meta_block(dim1=c4_in_channels//2, num_heads=8, mlp_ratio=mlp_ratio,
        #                                drop_path=0.1, pool_ratio=pool_ratio)
        #
        # self.attn3 = global_meta_block(dim1=c3_in_channels//2, num_heads=4, mlp_ratio=mlp_ratio, drop_path=0.1,
        #                                pool_ratio=pool_ratio * 2)
        #
        # self.attn2 = global_meta_block(dim1=c2_in_channels//2, num_heads=2, mlp_ratio=mlp_ratio, drop_path=0.1,
        #                                pool_ratio=pool_ratio * 4)
        # self.attn1 = global_meta_block(dim1=c1_in_channels//2, num_heads=1, mlp_ratio=mlp_ratio, drop_path=0.1,
        #                                pool_ratio=pool_ratio * 8)
        self.attn4 = global_meta_block(dim1=896, num_heads=8, mlp_ratio=mlp_ratio,
                                       drop_path=0.1, pool_ratio=pool_ratio)

        self.attn3 = global_meta_block(dim1=448, num_heads=4, mlp_ratio=mlp_ratio, drop_path=0.1,
                                       pool_ratio=pool_ratio * 2)

        self.attn2 = global_meta_block(dim1=224, num_heads=2, mlp_ratio=mlp_ratio, drop_path=0.1,
                                       pool_ratio=pool_ratio * 4)
        self.attn1 = global_meta_block(dim1=112, num_heads=1, mlp_ratio=mlp_ratio, drop_path=0.1,
                                       pool_ratio=pool_ratio * 8)

        self.linear_fuse = ConvModule(
            #in_channels=(c1_in_channels + c2_in_channels + c3_in_channels + c4_in_channels),
            #in_channels=(c2_in_channels + c3_in_channels + c4_in_channels),
            in_channels=(112),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.eucb3 = EUCB(in_channels=896, out_channels=448, kernel_size=3, stride=3 // 2)
        self.eucb2 = EUCB(in_channels=448, out_channels=224, kernel_size=3, stride=3 // 2)
        self.eucb1 = EUCB(in_channels=224, out_channels=112, kernel_size=3, stride=3 // 2)

        
  
        
      
        
        



        self.mfam3 = MFAM(896,448)
        self.mfam2 = MFAM(448, 224)
        self.mfam1 = MFAM(224, 112)



        self.lks4 = LKS(896,16,16)
        self.lks3 = LKS(448,32,32)
        self.lks2 = LKS(224,64,64)
        self.lks1 = LKS(112,128,128)

        


    def forward(self, inputs):
        x = self._transform_inputs(inputs)

        c1  = inputs[0]
        c2 = inputs[1]
        c3 = inputs[2]
        c4 = inputs[3]


        
        
        
       
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        

        c4 = self.lks4(c4) + c4
        _c4 = c4.flatten(2).transpose(1, 2)

        a4 = self.attn4(_c4, h4, w4)
        
      
        #
        c4 = a4.permute(0, 2, 1).reshape(n, -1, h4, w4)
       
       
        d4 = self.eucb3(c4)

        

       
        c3 = self.lks3(c3) + c3
        c3 = c3.flatten(2).transpose(1, 2)
        #
        a3 = self.attn3(c3,h3,w3)
        c3 = a3.permute(0, 2, 1).reshape(n, -1, h3, w3)

        
        c3 = self.mfam3(c3,d4)

        d3 = self.eucb2(c3)

        c2 = self.lks2(c2) + c2
        c2 = c2.flatten(2).transpose(1, 2)
        a2 = self.attn2(c2,h2,w2)
        c2 = a2.permute(0, 2, 1).reshape(n, -1, h2, w2)


        c2 = self.mfam2(c2,d3)

        d2 = self.eucb1(c2)


        c1 = self.lks1(c1) + c1
        c1 = c1.flatten(2).transpose(1, 2)
        c1 = self.attn1(c1,h1,w1)
        c1 = c1.permute(0, 2, 1).reshape(n, -1, h1, w1)
        # c1 = self.mla1(c1)

        c1 = self.mfam1(c1, d2)

        
        _c = self.linear_fuse(c1)





        x = self.dropout(_c)
        x = self.linear_pred(x)


        return x












