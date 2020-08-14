import torch
import torch.nn as nn
import torch.nn.functional as F


class Depth_Separable_Convolution(nn.Module):
    """
    Depthwise Separable Convolution layer within encoder block
    Incoporate Layernorm
    """
    def __init__(self, input_dim, output_dim, kernel_size):
        super().__init__()

        # depthwise separable convolutions: memory efficient and better generalization
        # https://arxiv.org/pdf/1610.02357.pdf
        # https://arxiv.org/pdf/1706.03059.pdf

        self.depth = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=
            input_dim,  # 按照官方文档，each input channel is convolved with its own set of filters
            bias=False)
        self.sep = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def forward(self, x):
        return self.sep(self.depth(x.permute(0,2,1))).permute(0,2,1)
