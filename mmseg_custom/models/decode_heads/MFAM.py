#论文：B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection
#论文地址：https://ieeexplore.ieee.org/document/10547405
import torch
import torch.nn as nn
from .EMA import EMA

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv = nn.Conv2d(dim, dim, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.conv0 = DSConv_pro(dim, dim, 5)
        # self.cot1 = cot = CoTAttention(dim=dim,kernel_size=7)
        # self.cot2 = cot = CoTAttention(dim=dim,kernel_size=11)
        # self.cot3 = cot = CoTAttention(dim=dim,kernel_size=21)
        #
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        #
        #
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        #
        #
        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        # self.convc1 = AdaptiveDilatedConv(dim,dim,3)
        # self.convc2 = AdaptiveDilatedConv(dim,dim,3)
        # self.convc3 = AdaptiveDilatedConv(dim,dim,3)
        # self.convc1 = AdaptiveDilatedConv(dim,dim,3)
        # self.convc2 = AdaptiveDilatedConv(dim,dim,3)
        # self.convc3 = AdaptiveDilatedConv(dim,dim,3)
        self.convc1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=6, dilation=6)
        self.convc2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=12, dilation=12)
        self.convc3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=18, dilation=18)
        self.convd = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        # 蛇形卷积
        # self.dsc_conv0_1 = DSConv_pro(dim, dim, kernel_size=7, extend_scope=1, morph=0, if_offset=True)
        # self.dsc_conv0_2 = DSConv_pro(dim, dim, kernel_size=7, extend_scope=1, morph=1, if_offset=True)

        # self.dsc_conv1_1 = DSConv_pro(dim, dim, kernel_size=11, extend_scope=1, morph=0, if_offset=True)
        # self.dsc_conv1_2 = DSConv_pro(dim, dim, kernel_size=11, extend_scope=1, morph=1, if_offset=True)

        # self.dsc_conv2_1 = DSConv_pro(dim, dim, kernel_size=21, extend_scope=1, morph=0, if_offset=True)
        # self.dsc_conv2_2 = DSConv_pro(dim, dim, kernel_size=21, extend_scope=1, morph=1, if_offset=True)
        # 使用 DCNv4 替代 conv0

        # self.conv0_1 = DSCNX(dim, kernel_size=7, dw_kernel_size=7, stride=1, pad=3, dilation=1, group=dim)
        # # 替换conv0_2为DSCNY
        # # 这里假设原来的(7, 1)卷积核，设置DSCNY的kernel_size和dw_kernel_size为7
        # self.conv0_2 = DSCNY(dim, kernel_size=7, dw_kernel_size=7, stride=1, pad=3, dilation=1, group=dim)
        #
        # # 替换conv1_1为DSCNX
        # self.conv1_1 = DSCNX(dim, kernel_size=11, dw_kernel_size=11, stride=1, pad=5, dilation=1, group=dim)
        # # 替换conv1_2为DSCNY
        # self.conv1_2 = DSCNY(dim, kernel_size=11, dw_kernel_size=11, stride=1, pad=5, dilation=1, group=dim)
        #
        # # 替换conv2_1为DSCNX
        # self.conv2_1 = DSCNX(dim, kernel_size=21, dw_kernel_size=21, stride=1, pad=10, dilation=1, group=dim)
        # # 替换conv2_2为DSCNY
        # self.conv2_2 = DSCNY(dim, kernel_size=21, dw_kernel_size=21, stride=1, pad=10, dilation=1, group=dim)

        # self.conv0 = DCNv4(channels=dim, kernel_size=3, stride=1, pad=1, group=4)

        # 使用 DCNv4 替代 conv0_1 和 conv0_2
        # self.conv0_1 = DSConv_pro(in_channels=dim, out_channels=dim, kernel_size=(1, 7), extend_scope=1, morph=0,if_offset=True,device=device)
        # self.conv0_2 = DSConv_pro(in_channels=dim, out_channels=dim, kernel_size=(1, 7), extend_scope=1, morph=0,if_offset=True,device=device)
        #
        # # 使用 DCNv4 替代 conv1_1 和 conv1_2
        # self.conv1_1 = DSConv_pro(in_channels=dim, out_channels=dim, kernel_size=(1, 11), extend_scope=1, morph=0,if_offset=True,device=device)
        # self.conv1_2 = DSConv_pro(in_channels=dim, out_channels=dim, kernel_size=(11, 1), extend_scope=1, morph=0,if_offset=True,device=device)
        #
        # 使用 DCNv4 替代 conv2_1 和 conv2_2
        # self.conv2_1 = DSConv_pro(in_channels=dim, out_channels=dim, kernel_size=(1, 21), extend_scope=1, morph=0,if_offset=True,device=device)
        # self.conv2_2 = DSConv_pro(in_channels=dim, out_channels=dim, kernel_size=(21, 1), extend_scope=1, morph=0,if_offset=True,device=device)

        self.conv3 = nn.Conv2d(dim, dim, 1)


        self.co1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU())
        # self.c = CBAM(dim)

        self.conv5 = nn.Conv2d(dim * 2, dim, 1)
        # self.dcv = DCNv4(dim)

    def forward(self, x):
        u = x.clone()

        # b,c,h,w = x.size()
        attn = self.conv0(x)
        # attnc1 = self.convc1(x)
        # attnc2 = self.convc2(x)
        # attnc3 = self.convc3(x)

        # x = self.conv0(x)
        # attn = x.permute(0, 2, 3, 1)
        attn_0 = self.conv0_1(attn)
        # # attn_0 = self.conv0_2(attn_0)
        attn_0_0 = self.conv0_2(attn_0)
        # attn_0 = self.conv0_1(attn,x)
        # attn_0_0 = self.conv0_2(attn_0,x)
        # combined1 = torch.cat([attnc1, attn_0_0], dim=1)
        # pooled = self.global_avg_pool(combined1)
        # pooled = torch.flatten(pooled, 1)
        # sigm = self.fc(pooled)
        #
        # a = sigm.view(-1, sigm.size(1), 1, 1)
        # a1 = 1 - sigm
        # a1 = a1.view(-1, a1.size(1), 1, 1)
        #
        # y = attnc1 * a
        # y1 = attn_0_0 * a1
        #
        # combined1 = torch.cat([y, y1], dim=1)
        # attn_0_0 = self.convd(combined1)
        #
        attn_1 = self.conv1_1(attn)
        attn_1_1 = self.conv1_2(attn_1)
        #
        # combined2 = torch.cat([attnc2, attn_1_1], dim=1)
        # pooled = self.global_avg_pool(combined2)
        # pooled = torch.flatten(pooled, 1)
        # sigm = self.fc(pooled)
        #
        # a = sigm.view(-1, sigm.size(1), 1, 1)
        # a1 = 1 - sigm
        # a1 = a1.view(-1, a1.size(1), 1, 1)
        #
        # y = attnc2 * a
        # y1 = attn_1_1 * a1

        # combined2 = torch.cat([y, y1], dim=1)
        # attn_1_1 = self.convd(combined2)

        # attn_1 = self.conv1_2(attn)

        attn_2 = self.conv2_1(attn)
        attn_2_2 = self.conv2_2(attn_2)
        # attn_2 = self.conv2_2(attn)
        # combined3 = torch.cat([attnc3, attn_2_2], dim=1)
        # pooled = self.global_avg_pool(combined3)
        # pooled = torch.flatten(pooled, 1)
        # sigm = self.fc(pooled)
        #
        # a = sigm.view(-1, sigm.size(1), 1, 1)
        # a1 = 1 - sigm
        # a1 = a1.view(-1, a1.size(1), 1, 1)
        #
        # y = attnc3 * a
        # y1 = attn_2_2 * a1
        #
        # combined3 = torch.cat([y, y1], dim=1)
        # attn_2_2 = self.convd(combined3)

        # attn_1 = self.conv1_1(attn)
        # attn_1_1 = self.conv1_2(attn_1)
        # attn_1 = self.conv1_1(attn,x)
        # attn_1_1 = self.conv1_2(attn_1,x)

        # attn_1 = self.conv1_2(attn)

        # attn_2 = self.conv2_1(attn)
        # attn_2_2 = self.conv2_2(attn_2)
        # attn_2 = self.conv2_1(attn,x)
        # attn_2_2 = self.conv2_2(attn_2,x)
        # attn_2 = self.conv2_2(attn)

        # attn_0_0 = attn_0_0.permute(0, 3, 1, 2)
        # attn_1_1 = attn_1_1.permute(0, 3, 1, 2)
        # attn_2_2 = attn_2_2.permute(0, 3, 1, 2)

        # attn = attn.permute(0, 3, 1, 2)

        attn_0_1 = self.conv1_1(attn_0_0)
        attn_1_0 = self.conv1_2(attn_0_1)

        attn_0_2 = self.conv2_1(attn_1_0)
        attn_2_0 = self.conv2_2(attn_0_2)

        # attn_2_1 = self.conv1_1(attn_2_2)
        # attn_1_2 = self.conv1_2(attn_2_1)
        #
        # attn_0_3 = self.conv0_1(attn_1_2)
        # attn_0_4 = self.conv0_2(attn_0_3)

        # attn = attn + attn_0_0 + attn_1_1 + attn_2_2
        # attn = attn + attn_0 + attn_1 + attn_2
        # attn = attn.permute(0, 3, 1, 2)
        attn = attn + attn_0_0 + attn_1_1 + attn_2_2 + attn_2_0
        attn = self.conv3(attn)

        # cc = self.cc(attn)
        # ss = self.ss(attn)
        # attn = torch.concat((cc,ss),dim = 1)
        # attn = self.conv5(attn)

        return attn * u
    
#Simam: A simple, parameter-free attention module for convolutional neural networks (ICML 2021)
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

#bitemporal feature aggregation module (BFAM)
class BFAM(nn.Module):
    def __init__(self,inp,out):
        super(BFAM, self).__init__()

        # self.pre_siam = simam_module()
        self.pre_siam = simam_module()
        self.lat_siam = simam_module()
        self.ema = EMA(inp//2)
        self.attn = AttentionModule(inp)

        out_1 = int(inp/2)

        self.conv_1 = nn.Conv2d(inp, out_1 , padding=1, kernel_size=3,groups=out_1,
                                   dilation=1)
        self.conv_2 = nn.Conv2d(inp, out_1, padding=2, kernel_size=3,groups=out_1,
                                   dilation=2)
        self.conv_3 = nn.Conv2d(inp, out_1, padding=3, kernel_size=3,groups=out_1,
                                   dilation=3)
        self.conv_4 = nn.Conv2d(inp, out_1, padding=4, kernel_size=3,groups=out_1,
                                   dilation=4)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_1 * 4, out_1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_1),
            nn.ReLU(inplace=True)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(out_1 * 2, out_1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_1),
            nn.ReLU(inplace=True)
        )

        self.fuse_siam = simam_module()

        self.out = nn.Sequential(
            nn.Conv2d(out_1, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    def forward(self,inp1,inp2,last_feature=None):
        x = torch.cat([inp1,inp2],dim=1)
        # c1 = self.conv_1(x)
        # c2 = self.conv_2(x)
        # c3 = self.conv_3(x)
        # c4 = self.conv_4(x)
        # cat = torch.cat([c1,c2,c3,c4],dim=1)
        # fuse = self.fuse(cat)
        fuse = self.attn(x)
        fuse = self.fuse1(fuse)
       
        # inp1_siam = self.pre_siam(inp1)
        inp1_siam = self.ema(inp1)
        # inp2_siam = self.lat_siam(inp2)
        inp2_siam = self.ema(inp2)

        
        inp1_mul = torch.mul(inp1_siam,fuse)
        inp2_mul = torch.mul(inp2_siam,fuse)
        fuse = self.fuse_siam(fuse)
        if last_feature is None:
            out = self.out(fuse + inp1 + inp2 + inp2_mul + inp1_mul)
        else:
            out = self.out(fuse+inp2_mul+inp1_mul+last_feature+inp1+inp2)
        out = self.fuse_siam(out)
        

        return out


if __name__ == '__main__':

    block = BFAM(inp=128, out=256)

    inp1 = torch.rand(1, 128 // 2, 16, 16) # B C H W
    inp2 = torch.rand(1, 128 // 2, 16, 16)# B C H W
    last_feature = torch.rand(1, 128 // 2, 16, 16)# B C H W

    # 通过BFAM模块，这里没有提供last_feature的话，可以为None
    output = block(inp1, inp2, last_feature)
    # output = bfam(inp1, inp2)

    # 打印输入和输出的shape
    print(inp1.size())
    print(inp2.size())
    print(output.size())
