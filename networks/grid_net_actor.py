from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# deeplabv3自编码器
# 参考代码https: // blog.csdn.net / chenfang0529 / article / details / 108133672

class AutoEncoder(nn.Module, ABC):
    def __init__(self, n_actions: int, grid_input_shape: list):
        super(AutoEncoder, self).__init__()
        self.grid_input_shape = grid_input_shape
        self.auto_encoder_output_shape = grid_input_shape[2] * grid_input_shape[3] * 64
        self.conv_block_1x1_1 = self.conv_block(input_channel=grid_input_shape[1], output_channel=64, kernel_size=1)
        self.conv_block_3x3_1 = self.conv_block(input_channel=grid_input_shape[1], output_channel=64,
                                                kernel_size=3, padding=6, dilation=6)
        self.conv_block_3x3_2 = self.conv_block(input_channel=grid_input_shape[1], output_channel=64,
                                                kernel_size=3, padding=12, dilation=12)

        self.avg_pool = nn.AdaptiveAvgPool2d(grid_input_shape[3] // 4)
        self.conv_block_1x1_2 = nn.Conv2d(grid_input_shape[1], 64, kernel_size=1)
        self.conv_block_1x1_3 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv_block_1x1_4 = self.conv_block(input_channel=grid_input_shape[1], output_channel=64, kernel_size=1)
        self.conv_block_1x1_5 = self.conv_block(input_channel=128, output_channel=256, kernel_size=1)
        self.conv_block_1x1_6 = nn.Conv2d(256, n_actions, kernel_size=1)

    # 定义一个卷积块的静态方法，增加泛用性
    @staticmethod
    def conv_block(input_channel: int, output_channel: int, kernel_size: int,
                   stride: int = 1, padding: int = 0, dilation: int = 1) -> nn.Sequential:
        one_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )
        return one_conv_block

    def encoder(self, grid_input: Tensor):
        grid_input_w = self.grid_input_shape[2]
        grid_input_h = self.grid_input_shape[3]
        out_1x1_1 = self.conv_block_1x1_1(grid_input)  # 对应图中 E
        out_3x3_1 = self.conv_block_3x3_1(grid_input)  # 对应图中 D
        out_3x3_2 = self.conv_block_3x3_2(grid_input)  # 对应图中 C
        grid_input_avg = self.avg_pool(grid_input)  # 对应图中 ImagePooling
        out_1x1_2 = self.conv_block_1x1_2(grid_input_avg)
        out_1x1_2_up = F.interpolate(out_1x1_2, size=(grid_input_h, grid_input_w), mode="bilinear", align_corners=False)
        out_cat = torch.cat([out_1x1_1, out_3x3_1, out_3x3_2, out_1x1_2_up], 1)
        encoder_out = self.conv_block_1x1_3(out_cat)  # 对应图中 H  out 对应图中I
        return encoder_out

    def decoder(self, grid_input: Tensor, encoder_output: Tensor):
        grid_input_w = self.grid_input_shape[2]
        grid_input_h = self.grid_input_shape[3]
        out_1x1_4 = self.conv_block_1x1_4(grid_input)
        encoder_output_up = F.interpolate(encoder_output, size=(grid_input_h, grid_input_w), mode="bilinear",
                                          align_corners=False)
        out_cat = torch.cat([out_1x1_4, encoder_output_up], 1)
        out_1x1_5 = self.conv_block_1x1_5(out_cat)
        decoder_out = self.conv_block_1x1_6(out_1x1_5)
        return decoder_out

    # 前向传播
    def forward(self, grid_input):
        encoder_output = self.encoder(grid_input)
        decoder_output = self.decoder(grid_input, encoder_output)
        decoder_output_softmax = F.softmax(decoder_output, dim=1)
        encoder_output_clone = encoder_output.clone().detach().view(-1, self.auto_encoder_output_shape)
        return decoder_output_softmax, encoder_output_clone


class AutoEncoderContinuousActions(AutoEncoder, ABC):
    def __init__(self, grid_input_shape: list, action_dim: int = 5):
        super(AutoEncoderContinuousActions, self).__init__(action_dim, grid_input_shape)

    def forward(self, grid_input):
        encoder_output = self.encoder(grid_input)
        decoder_output = self.decoder(grid_input, encoder_output)
        # 对action进行放缩，实际上a in [0,1]
        decoder_output_sigmoid = torch.sigmoid(decoder_output)
        encoder_output_clone = encoder_output.clone().detach().view(-1, self.auto_encoder_output_shape)
        return decoder_output_sigmoid, encoder_output_clone
