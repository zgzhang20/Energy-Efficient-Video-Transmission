### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import numpy
import math
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from models.pytorch_gdn import GDN
# from .gmflow import GMFlow
import argparse
# import models.High_semantic_atten as High_semantic_atten
# from functools import reduce
import torch.nn.functional as F
from torchvision import models
import torchvision
from models.mbt import mbt_ga,mbt_gs,wz_ga,wz_gs,si_ga,FasterNet
from models.densenet import DenseNet
from models.IFNet import IFNet
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



class mi(nn.Module):
    def __init__(self):
        super(mi, self).__init__()

        self.mbt_ga=FasterNet(192)
        self.wz_ga = FasterNet(192)

        self.mbt_gs = mbt_gs()
        # self.wz_gs = wz_gs()
        self.recons_mid_forw = recons_mid_forw()
        self.IFNet=IFNet()
        self.si_ga=si_ga()
        self.STPNet=STPNet(64, 64, 64)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


class AF_Module(nn.Module):  # 合成图降维
    def __init__(self, inchannel):
        super(AF_Module, self).__init__()
        self.ave = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(inchannel + 1, int(inchannel / 8), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(inchannel / 8), inchannel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        y = SNR_to_noise(y)
        y = y.tolist()
        x1 = self.ave(x)
        ba = x1.shape[0]
        y = torch.tensor(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        y = y.repeat(ba, 1, 1, 1)
        x1 = torch.cat((y, x1), dim=1)
        x1 = self.se(x1)
        x2 = x * x1

        return x2

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.AF_Module1 = AF_Module(out_planes)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x, n_var):
        x = self.conv1(x)
        x = self.AF_Module1(x, n_var)
        x = self.conv2(x)
        return x


class deconv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(deconv, self).__init__()
        self.Tconv = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1,
                                     bias=True),
            nn.PReLU(out_planes)
        )
        self.AF_Module1 = AF_Module(out_planes)

    def forward(self, x, n_var):
        x = self.Tconv(x)
        x = self.AF_Module1(x, n_var)
        return x


class deconv2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(deconv2, self).__init__()
        self.Tconv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.PReLU(out_planes)
        )
        self.AF_Module1 = AF_Module(out_planes)

    def forward(self, x, n_var):
        x = self.Tconv(x)
        x = self.AF_Module1(x, n_var)
        return x


class Unet(nn.Module):
    def __init__(self, c=16):
        super(Unet, self).__init__()
        self.down0 = Conv2(3, 2 * c)
        self.down1 = Conv2(2 * c, 4 * c)
        self.down2 = Conv2(4 * c, 8 * c)
        self.down3 = Conv2(8 * c, 16 * c)
        self.up0 = deconv(16 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv2(8 * c, 2 * c)
        self.up3 = deconv2(4 * c, 4 * c)
        self.conv = nn.Conv2d(2 * c, 4 * c, 3, 1, 1)

    def forward(self, x, n_var):
        s0 = self.down0(x, n_var)
        s1 = self.down1(s0, n_var)
        s2 = self.down2(s1, n_var)
        s3 = self.down3(s2, n_var)
        x = self.up0(s3, n_var)
        x = self.up1(torch.cat((x, s2), 1), n_var)
        x = self.up2(torch.cat((x, s1), 1), n_var)
        # print(x.shape)
        # x = self.up3(torch.cat((x, s0), 1), n_var)
        x = self.conv(x)
        # return torch.sigmoid(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)



# 右侧的 residual block 结构（50-layer、101-layer、152-layer）
class Bottleneck(nn.Module):
    # expansion = 4

    def __init__(self, channel, stride=1):
        super(Bottleneck, self).__init__()
        self.SE = SE_Block(channel)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        SE_out = self.SE(x)
        out = x * SE_out
        out +=x
        out = F.relu(out)
        SE_out = self.SE(x)
        out = out * SE_out
        out += x
        out = F.relu(out)
        return out


class STPNet(nn.Module):
    def __init__(self, channel_in, gc, channel_out):
        super(STPNet, self).__init__()
        self.dense_ = DenseNet()
        self.Unet = Unet()
        self.lrelu = nn.PReLU()

        self.conv1_1 = nn.Conv2d(channel_out * 2, channel_out, 3, 1, 1)

        self.attention = Bottleneck(channel_out)

        self.soft = nn.Softmax()

        self.convmlp = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2
                      ),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=24, kernel_size=3, padding=1, stride=2
                      ),
            nn.PReLU(),

        )
        # self.con_output = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1),

                                        # )

    def forward(self, x, n_var):
        x_f = self.dense_(x)

        x_s = self.Unet(x, n_var)

        x_mid = self.conv1_1(torch.cat((x_f, x_s), dim=1))

        x_att = self.lrelu((self.attention(x_mid)))

        return self.convmlp(x_att)


class Deform(nn.Module):
    def __init__(self, channel):
        super(Deform, self).__init__()
        self.conv = nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1)  # 原卷积
        self.conv_offset = nn.Conv2d(channel, 18, kernel_size=3, stride=1, padding=1)
        init_offset = torch.Tensor(np.zeros([18, channel, 3, 3]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset)  # 初始化为0
        self.conv_mask = nn.Conv2d(channel, 9, kernel_size=3, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([9, channel, 3, 3]) + np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask)  # 初始化为0.5

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))
        out = torchvision.ops.deform_conv2d(input=x, offset=offset,
                                            weight=self.conv.weight,
                                            mask=mask, padding=(1, 1))
        return out


def deconvC(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class recons_mid_forw(nn.Module):  # 原图降维
    def __init__(self, ):
        super(recons_mid_forw, self).__init__()

        self.deform_i = Deform(128)

        self.convcc0 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, padding=1, stride=1))
        self.convcc1 = nn.Sequential(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride=1))

        self.upsample1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            GDN(192, device),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=48, out_channels=192, kernel_size=3, padding=1),
            GDN(192, device),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=48, out_channels=192, kernel_size=3, padding=1),
            GDN(192, device),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=48, out_channels=192, kernel_size=3, padding=1),
            GDN(192, device),
            nn.PixelShuffle(2),
        )
        # self.upsample2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.PixelShuffle(2)
        # )

        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=3, kernel_size=3, padding=1, stride=1
                      ))
        # self.relu = nn.PReLU()

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
        self.AF_Module0 = AF_Module(256)
        self.AF_Module1 = AF_Module(192)
        # self.AF_Module2 = AF_Module(256)

        self.conv1_1 = nn.Conv2d(in_channels=24, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv1_2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, padding=1, stride=1)

        self.attention = Bottleneck(24)

        self.relu = nn.PReLU()
        self.dgn1=GDN(192,device)

        self.g_s = nn.Sequential(
            deconvC(256, 256, kernel_size=5, stride=2),
            GDN(256,device, inverse=True),
            deconvC(256, 256, kernel_size=5, stride=2),
            GDN(256,device, inverse=True),
            deconvC(256, 256, kernel_size=5, stride=2),
            GDN(256,device, inverse=True),
            deconvC(256, 3, kernel_size=5, stride=2),
        )
    def forward(self, crc, si_info, n_var):
        crc_r = self.attention(crc)
        crc_r = crc+crc_r
        crc_r = self.conv1_1(crc_r)

        si_info = self.conv1_2(si_info)
        si_r = self.deform_i(si_info)
        si_r=si_r+si_info

        crc = torch.cat((crc_r, si_r), dim=1)

        x=self.g_s(crc)
        # x = self.AF_Module0(crc, n_var)
        # x = self.convcc0(x)
        # x = self.dgn1(x)
        # x = self.relu(x)
        #
        # residual = x
        # x = self.AF_Module1(x, n_var)
        # x = self.convcc1(x) + residual
        # x = self.dgn1(x)
        # x = self.relu(x)
        # residual = x
        # x = self.AF_Module1(x, n_var)
        # x = self.convcc1(x) + residual
        # x = self.dgn1(x)
        # x = self.relu(x)
        #
        # residual = x
        # x = self.AF_Module1(x, n_var)
        # x = self.convcc1(x) + residual
        # x = self.dgn1(x)
        # x = self.relu(x)
        #
        # x = self.AF_Module1(x, n_var)
        # x = self.convcc1(x)
        # x = self.dgn1(x)
        # x = self.relu(x)
        #
        # x = self.upsample1(x)
        #
        # x = self.conv1_7(x)
        # x = self.sigmoid(x)

        return x


def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)

    return x


class Channels():

    def AWGN(self, Tx_sig, snr):
        snr = 10 ** (snr / 10)
        noise_std = 1 / np.sqrt(2 * snr)
        Rx_sig = Tx_sig + torch.normal(0, noise_std, size=Tx_sig.shape).to(device)
        return Rx_sig

    def awgn_acc(self, x, snr):
        '''
        加入高斯白噪声 Additive White Gaussian Noise
        :param x: 原始信号
        :param snr: 信噪比
        :return: 加入噪声后的信号
        '''
        # np.random.seed(seed)  # 设置随机种子
        t_snr = 10 ** (snr / 10.0)
        y = x.view(-1)
        xpower = torch.sum(y ** 2) / len(y)
        npower = xpower / t_snr
        noise = torch.rand(x.shape).to(device) * torch.sqrt(npower)

        return x + noise

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        # Rx_sig = Tx_sig
        Rx_sig = self.awgn_acc(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = Tx_sig
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig


def psnr(img1, img2):
    # img1 = np.float64(img1)
    # img2 = np.float64(img2)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def CalcuPSNR(target, ref):
    diff = ref - target
    diff = diff.flatten(0)
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / (rmse))

ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)
# noinspection DuplicatedCode

def key_value(model, bimage, channel, n_var):

    model.eval()
    channels = Channels()
    image = bimage[[2 * i - 2 for i in range(1, 5)]]
    # image_pre = bimage[[2 * i - 1 for i in range(1, 4)]]
    #
    # print(image.shape)
    key_frame = model.mbt_ga(image)
    # CRC_Value = model.wz_ga(image_pre)

    Key_send = PowerNormalize(key_frame)
    # WZ_send = PowerNormalize(CRC_Value)

    if channel == 'awgn_acc':
        Key_send = channels.awgn_acc(Key_send, n_var)
        # WZ_send = channels.awgn_acc(WZ_send, n_var)
        # bid_flow = channels.AWGN(bid_flow, n_var)
    elif channel == 'Rayleigh':
        Key_send = channels.awgn_acc(Key_send, n_var)
        # WZ_send = channels.awgn_acc(WZ_send, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh")
    Key_send = model.mbt_gs(Key_send)
    return Key_send

def train_pyconv(Key_M,model, opt, bimage, channel, n_var):

    model.train()
    opt.zero_grad()
    channels = Channels()
    # image = bimage[[2 * i - 2 for i in range(1, 5)]]
    image_pre = bimage[[2 * i - 1 for i in range(1, 4)]]
    #
    # print(image.shape)
    # key_frame = model.mbt_ga(image)
    CRC_Value = model.wz_ga(image_pre)

    # Key_send = PowerNormalize(key_frame)
    WZ_send = PowerNormalize(CRC_Value)

    if channel == 'awgn_acc':
        # Key_send = channels.awgn_acc(Key_send, n_var)
        WZ_send = channels.awgn_acc(WZ_send, n_var)
        # bid_flow = channels.AWGN(bid_flow, n_var)
    elif channel == 'Rayleigh':
        # Key_send = channels.awgn_acc(Key_send, n_var)
        WZ_send = channels.awgn_acc(WZ_send, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh")
    Key_send = key_value(Key_M,bimage, channel, n_var)

    key = Key_send
    a = torch.cat((key[0], key[1], key[2],key[1],key[2],key[3]), dim=0).unsqueeze(0)

    flow, mask, merged, flow_teacher, merged_teacher, loss_distill =model.IFNet(a)

    c1 = model.si_ga(merged[0])
    c2 = model.si_ga(merged[1])
    c3 = model.si_ga(merged[2])
    c=torch.cat((c1,c2,c3),dim=0)

    b_b = torch.cat((merged[0], merged[1], merged[2]), 0).to(device)

    si_info = model.STPNet(b_b, n_var)

    # print(c.shape,si_info.shape,WZ_send.shape)
    cat_si_cr=torch.cat((c,si_info),dim=1)
    # print(WZ_send.shape,cat_si_cr.shape)

    WZ_send=model.recons_mid_forw(WZ_send,cat_si_cr,n_var)
    # WZ_send = model.wz_gs(cat_si_cr)

    # mse1 = torch.mean(torch.square(Key_send * 255 - image * 255))
    mse2 = torch.mean(torch.square(WZ_send * 255 - image_pre * 255))
    loss = mse2
    
    ssim_value = loss.data.item()
    loss.backward()
    opt.step()

    return ssim_value


def mse(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return mse


def test_epoch(Key_M, model, bimage, channel, n_var):

    model.eval()
    channels = Channels()
    image = bimage[[2 * i - 2 for i in range(1, 5)]]
    image_pre = bimage[[2 * i - 1 for i in range(1, 4)]]
    #
    # print(image.shape)
    # key_frame = model.mbt_ga(image)
    CRC_Value = model.wz_ga(image_pre)

    # Key_send = PowerNormalize(key_frame)
    WZ_send = PowerNormalize(CRC_Value)

    if channel == 'awgn_acc':
        # Key_send = channels.awgn_acc(Key_send, n_var)
        WZ_send = channels.awgn_acc(WZ_send, n_var)
        # bid_flow = channels.AWGN(bid_flow, n_var)
    elif channel == 'Rayleigh':
        # Key_send = channels.awgn_acc(Key_send, n_var)
        WZ_send = channels.awgn_acc(WZ_send, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh")
    Key_send = key_value(Key_M, bimage, channel, n_var)

    key = Key_send
    a = torch.cat((key[0], key[1], key[2], key[1], key[2], key[3]), dim=0).unsqueeze(0)

    flow, mask, merged, flow_teacher, merged_teacher, loss_distill = model.IFNet(a)

    c1 = model.si_ga(merged[0])
    c2 = model.si_ga(merged[1])
    c3 = model.si_ga(merged[2])
    c = torch.cat((c1, c2, c3), dim=0)

    b_b = torch.cat((merged[0], merged[1], merged[2]), 0).to(device)

    si_info = model.STPNet(b_b, n_var)

    # print(c.shape,si_info.shape,WZ_send.shape)
    cat_si_cr = torch.cat((c, si_info), dim=1)
    # print(WZ_send.shape,cat_si_cr.shape)

    WZ_send = model.recons_mid_forw(WZ_send, cat_si_cr, n_var)

    # loss1 = ms_ssim_module(Key_send * 255, image * 255)
    # loss2 = ms_ssim_module(WZ_send * 255, image_pre * 255)

    mse1 = torch.mean(torch.square(Key_send[0:3] * 255 - image[0:3] * 255))
    mse2 = torch.mean(torch.square(WZ_send * 255 - image_pre * 255))
    mse = (mse1+mse2)/2
    ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)
    ssim1 = ms_ssim_module(Key_send[0:3] * 255, image[0:3] * 255)
    ssim2 = ms_ssim_module(WZ_send * 255, image_pre * 255)
    ssim_s = (ssim1+ssim2)/2
    ssim_value = ssim_s.data.item()
    psn1 = psnr(Key_send[0:3] * 255, image[0:3] * 255)
    psn2 = psnr(WZ_send * 255, image_pre * 255)
    psn = (psn1+psn2)/2
    return WZ_send, ssim_value, psn, mse
