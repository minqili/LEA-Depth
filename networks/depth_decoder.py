# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda(0).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
    #构造注意力矩阵中用于 mask 的负无穷对角矩阵，用于强制避免某些无效位置参与注意力计算。常用于自注意力中排除自身或特定位置。

class eca_layer(nn.Module):#通道注意力模块（Efficient Channel Attention）
    """Constructs a ECA module.构建一个 ECA 模块。

    Args:
        channel: Number of channels of the input feature map输入特征图的通道数
        k_size: Adaptive selection of kernel size卷积核大小，决定了跨通道的局部感受野范围
    """
    def __init__(self, channel, k_size=3):#channel：输入特征的通道数；k_size：卷积核大小，决定通道注意力的范围。
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)# 全局平均池化（对 H x W 压缩，得到每个通道的全局特征） 输出形状 [B, C, 1, 1]
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)# 1D卷积，用于建模通道之间的关系（避免了复杂的全连接操作），输入形状 [B, C] 转换为 [B, 1, C]。
        self.sigmoid = nn.Sigmoid() # sigmoid 激活函数，用于生成通道注意力权重（0~1之间）

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module；squeeze + transpose 是为了把 [B, C, 1, 1] 变为 [B, 1, C] 送入 1D 卷积。
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)#用通道注意力权重乘以原始特征，实现加权。

        return x * y.expand_as(x)

class CCAttention(nn.Module):#CCAttention 模块：跨空间 + 通道混合注意力
    def __init__(self, in_channels):#in_channels: 输入特征图的通道数（通常是最后一层 encoder 的输出）。
        super(CCAttention, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 8#降维的通道数，用于减少计算量。
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)#构造注意力机制的 Q、K、V。

        self.SoftMax = nn.Softmax(dim=3)#SoftMax：做注意力归一化；
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))#gamma：一个可学习参数；
        self.ema = eca_layer(288)#ema：调用通道注意力模块。

    def forward(self, x):
        b, _, h, w = x.size()#输入为 [B, C, H, W]；
        # [b, c', h, w]
        query = self.ConvQuery(x)#query shape 为 [B, C', H, W]。
        # [b, w, c', h] -> [b*w, c', h] -> [b*w, h, c']
        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)#重排后形状变成 [B*W, H, C']。对 H 向进行注意力处理
        # [b, h, c', w] -> [b*h, c', w] -> [b*h, w, c']
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        # [b, c', h, w]
        key = self.ConvKey(x)
        # [b, w, c', h] -> [b*w, c', h]
        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)#对应的 Key 为 [B*W, C', H]。
        # [b, h, c', w] -> [b*h, c', w]
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # [b, c, h, w]
        value = self.ConvValue(x)
        # [b, w, c, h] -> [b*w, c, h]
        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        # [b, h, c, w] -> [b*h, c, w]
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # [b*w, h, c']* [b*w, c', h] -> [b*w, h, h] -> [b, h, w, h]
        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)#点积后得到 [B, H, W, H]，加入负无穷 mask。
        # [b*h, w, c']*[b*h, c', w] -> [b*h, w, w] -> [b, h, w, w]
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)
        # [b, h, w, h+w]  concate channels in axis=3
        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))#合并注意力权重，拼接后 shape 为 [B, H, W, H + W]。

        # [b, h, w, h] -> [b, w, h, h] -> [b*w, h, h]
        attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)#分离并计算输出值
        # [b*h, w, w]
        attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)

        # [b*w, h, c]*[b*w, h, h] -> [b, w, c, h] error [b,c,h,w]
        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        # [b,c,h,w]
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)
        chanel = self.ema(x)
        return self.gamma * (out_H + out_W + chanel) + x#输出 = 注意力增强 + 残差连接。

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, decoder_channel_scale=100):
        super(DepthDecoder, self).__init__()

        # 深度图每个像素代表的是物体到相机xy平面的距离，单位为mm。所以通道数默认为1
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        # 这里Decoder上采样的模型默认采用nearest
        self.upsample_mode = 'nearest'
        self.scales = scales

        # encoder各通道数
        self.num_ch_enc = num_ch_enc
        # decoder 通道数
        if decoder_channel_scale == 200:
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])      # 原始的
        elif decoder_channel_scale == 100:
            self.num_ch_dec = np.array([8, 16, 32, 64, 128])      # 缩小两倍
        elif decoder_channel_scale == 50:
            self.num_ch_dec = np.array([4, 8, 16, 32, 64])        # 缩小四倍

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.cca = CCAttention(self.num_ch_enc[-1])

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        #x = input_features[-1]
        x = self.cca(input_features[-1])

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
