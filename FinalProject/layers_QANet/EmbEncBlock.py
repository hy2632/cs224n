import torch.nn as nn

from layers_QANet.PE import Positional_Encoding
from layers_QANet.DSConv import Depth_Separable_Convolution
from layers_QANet.SelfAtt import Self_Attention
from layers_QANet.FFN import Feed_Forward
from layers_QANet.EncBlock import Encoder_Block


class Embedding_Encoder(nn.Module):
    def __init__(self, dim, maximum_context_length, num_conv, kernel_size,
                 num_heads):
        super().__init__()
        self.enc = Encoder_Block(dim, maximum_context_length, num_conv,
                                 kernel_size, num_heads)

    # Deal with Context Embedding & Question Embedding together, share the same encoder
    def forward(self, x_c, x_q, c_mask, q_mask):
        # x_c : (batch_size, seq_len, dim)
        # c_mask: (batch_size, seq_len, 1)

        if x_c.size(1) > self.maximum_context_length:
            raise ValueError("con_len > maximum_context_length")

        x_c_out = self.enc(x_c, c_mask)
        x_q_out = self.enc(x_q, q_mask)

        return x_c_out, x_q_out
