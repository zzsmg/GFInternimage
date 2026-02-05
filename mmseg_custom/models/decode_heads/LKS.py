import torch
from torch import nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from timm.models.layers import trunc_normal_




class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)



class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x





class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations,
                      bias=False),  ###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ImprovedLKS(nn.Module):
    def __init__(self, dim, H, W, s_kernel_size=3):
        super().__init__()

        # 多尺度上下文模块（空洞卷积金字塔）→ 4个不同膨胀率的卷积
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, padding=d, dilation=d, groups=dim // 4)
            for d in [1, 2, 3, 5]  # 保持4个尺度
        ])

        # 通道-空间注意力（保持原设计）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        # 动态融合权重→ 调整为4个权重参数
        self.fusion_weights = nn.Parameter(torch.ones(4))  # 关键修改点

        # 锐化模块（保持原设计）
        self.h_ctx = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.GELU(),
            DepthWiseSeparableConv(dim, dim, 3)
        )

        # 添加缺失的动态投影层
        self.dynamic_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )

        # 残差连接
        self.shortcut = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x

        # 生成多尺度上下文特征
        ctx_features = []
        for conv in self.dilated_convs:
            ctx_features.append(F.gelu(conv(x)))
        ctx_features = torch.stack(ctx_features, dim=1)  # [B,4,C,H,W]

        # 动态融合（修正权重维度）
        fusion_weights = F.softmax(self.fusion_weights, 0).view(1, -1, 1, 1, 1)
        fused_ctx = torch.sum(ctx_features * fusion_weights, dim=1)  # [B,C,H,W]

        # 锐化模块
        sharp_feat = self.h_ctx(x)
        sharp_feat = sharp_feat + F.hardtanh(sharp_feat - x.mean(dim=1, keepdim=True))

        # 特征融合（保持通道维度一致）
        fused = torch.cat([fused_ctx, sharp_feat], dim=1)

        # 动态投影 + 残差连接
        out = self.dynamic_proj(fused) + self.shortcut(identity)
        return F.gelu(out)


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size,
                                   padding=kernel_size // 2, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DLK_2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 将 3D 卷积替换为 2D 卷积
        self.att_conv1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        #

        # self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)

        # 将 3D 卷积替换为 2D 卷积
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(att1)

        att = torch.cat([att1, att2], dim=1)
        # 调整为 2D 维度的平均池化
        avg_att = torch.mean(att, dim=1, keepdim=True)
        # 调整为 2D 维度的最大池化
        max_att, _ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:, 0, :, :].unsqueeze(1) + att2 * att[:, 1, :, :].unsqueeze(1)
        output = output + x
        return output

class LKS(nn.Module):
    def __init__(self, dim,H,W, s_kernel_size=3):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1, padding=0)
        self.proj_2 = nn.Conv2d(dim * 2, dim, 1, padding=0)
        self.proj_3 = nn.Conv2d(dim * 3, dim, 1, padding=0)
        self.norm_proj = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        # multiscale spatial context layers
        self.s_ctx_1 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, padding=s_kernel_size // 2, groups=dim // 4)
        self.s_ctx_2 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, dilation=2, padding=(s_kernel_size // 2) * 2,
                                 groups=dim // 4)


        self.norm_s = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # sharpening module layers
   
        self.h_ctx = nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=False, groups=dim)
       

        self.norm_h = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.act = nn.GELU()

        self.simam = simam_module()
        # self.laplace_sharpening = LaplaceSharpening(dim)
        
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, padding=0),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )
      

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_1 = nn.Sigmoid()


        self.dlk = DLK_2D(dim)
     

        # 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid

       
        
        
        
    def forward(self, x):


        x = self.norm_proj(self.act(self.proj_1(x)))
        B, C, H, W = x.shape


        sx = self.dlk(x)




        
        
        hx = self.act(self.h_ctx(x))

      



      
        
        
        
        hx_t = x - hx.mean(dim=1).unsqueeze(1)
        
        hx_t = torch.softmax(hx.mean(dim=[-2, -1]).unsqueeze(-1).unsqueeze(-1), dim=1) * hx_t

        hx = self.norm_h(hx + hx_t)
        

        # combine the multiscale contetxual features with the sharpened features
        x = self.act(self.proj_2(torch.cat([sx, hx], dim=1)))
        
        
        
        return x






