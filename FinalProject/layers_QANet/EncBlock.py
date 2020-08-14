import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_QANet.PE import Positional_Encoding
from layers_QANet.DSConv import Depth_Separable_Convolution
from layers_QANet.SelfAtt import Self_Attention
from layers_QANet.FFN import Feed_Forward


def residual_block(x:torch.Tensor, layernorm:nn.Module, layer:nn.Module, mask=None):
    if mask:
        return x + layer(layernorm(x), mask)
    else:
        return x + layer(layernorm(x))

class Encoder_Block(nn.Module):
    def __init__(self, dim, maximum_context_length, num_conv, kernel_size,
                 num_heads):
        super().__init__()
        self.num_conv = num_conv

        # Positional Encoding
        self.pe = Positional_Encoding(dim, maximum_context_length)

        # Depth Separable Convolution layers * 4(EmbEnc) or 2 (ModEnc)
        self.conv_layers = nn.ModuleList([
            Depth_Separable_Convolution(dim, dim, kernel_size)
            for _ in range(num_conv)
        ])

        # Corresponding layernorm layers for each convolutional layer
        self.conv_lns = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(num_conv)])

        # Self-Attention (Multi-head)
        self.att = Self_Attention(num_heads, dim)
        self.att_ln = nn.LayerNorm(dim)

        # Feed Forward layer
        self.ffn = Feed_Forward(dim, dim)
        self.ffn_ln = nn.LayerNorm(dim)
    
    
    def forward(self, x, mask):
        # Positional Encoding
        x = self.pe(x)

        # multiple convolutions
        for i, conv, norm in zip(range(self.num_conv), self.conv_layers, self.conv_lns):
            x = residual_block(x, norm, conv)
        
        # Self Attention
        x = residual_block(x, self.att_ln, self.att, mask)

        # FFN
        x = residual_block(x, self.ffn_ln, self.ffn)

        return x


