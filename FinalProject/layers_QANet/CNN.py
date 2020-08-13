import torch
import torch.nn as nn
import torch.nn.functional as F

from layers_QANet.LayerNorm import LayerNorm

class depthwise_separable_convolution(nn.Module):
    """
    Depthwise Separable Convolution layer within encoder block
    Incoporate Layernorm
    """
    def __init__(self, input_dim, output_dim, num_convs, drop_prob=0.1):
        super().__init__()

        # depthwise separable convolutions: memory efficient and better generalization
        # https://arxiv.org/pdf/1610.02357.pdf
        # https://arxiv.org/pdf/1706.03059.pdf

        self.num_convs = num_convs
        self.layernorm = LayerNorm(input_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.dsconv = nn.Conv1d(in_channels=input_dim,
                              out_channels=input_dim,
                              kernel_size=7,
                              stride=1,
                              padding=3,
                              groups=output_dim, # 按照官方文档，each input channel is convolved with its own set of filters
                              bias=True)
        
    def forward(self, x):
        x = self.dropout(x)
        x_out = x
        for i in range(self.num_convs):
            x_out = self.layernorm(x_out)
            x_out = self.dsconv(x_out.permute(0,2,1)).permute(0,2,1)
        return x_out + x
        
