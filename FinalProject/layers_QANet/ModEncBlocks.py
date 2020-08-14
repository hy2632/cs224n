import torch.nn as nn

from layers_QANet.PE import Positional_Encoding
from layers_QANet.DSConv import Depth_Separable_Convolution
from layers_QANet.SelfAtt import Self_Attention
from layers_QANet.FFN import Feed_Forward
from layers_QANet.EncBlock import Encoder_Block


class Model_Encoder(nn.Module):
    def __init__(self, dim, maximum_context_length, num_conv, kernel_size,
                 num_heads, num_blocks):
        super().__init__()

        self.num_blocks = num_blocks
        self.maximum_context_length = maximum_context_length
        
        self.enc = nn.ModuleList([
            Encoder_Block(dim, maximum_context_length, num_conv, kernel_size,
                          num_heads) for _ in range(num_blocks)
        ])

    def forward(self, x, mask):
        # x_c : (batch_size, seq_len, dim)
        # c_mask: (batch_size, seq_len, 1)

        if x.size(1) > self.maximum_context_length:
            raise ValueError("con_len > maximum_context_length")

        for i, block in zip(range(self.num_blocks), self.enc):
            x = block(x, mask)
        m0 = x
        for i, block in zip(range(self.num_blocks), self.enc):
            x = block(x, mask)
        m1 = x
        for i, block in zip(range(self.num_blocks), self.enc):
            x = block(x, mask)
        m2 = x
        return m0, m1, m2