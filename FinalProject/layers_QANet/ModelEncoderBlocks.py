import torch.nn as nn

from layers_QANet.PositionalEncoding import PositionalEncoding
from layers_QANet.CNN import depthwise_separable_convolution
from layers_QANet.SelfAttention import SelfAttention
from layers_QANet.FFN import FFN

class ModelEncoderBlocks(nn.Module):
    def __init__(self,
                 d_model=128,
                 num_blocks=7,
                 maximum_context_length=400):

        super().__init__()

        self.maximum_context_length = maximum_context_length
        self.num_blocks = num_blocks

        self.posenc = PositionalEncoding(input_dim=d_model,
                                         max_len=maximum_context_length)
        
        self.cnn = depthwise_separable_convolution(input_dim=d_model, output_dim=d_model, num_convs=2, drop_prob=0.1)
        self.att = SelfAttention(n_heads=8, d_model=d_model)
        self.ffn = FFN(input_dim=d_model,
                       output_dim=d_model,
                       drop_prob=0.1)

    def forward(self, x):
        if x.size(1) > self.maximum_context_length:
            raise ValueError("seq_len > maximum_context_length")
        
        x_out = x
        for i in range(self.num_blocks):
            x_out = self.posenc(x_out)
            x_out = self.cnn(x_out)
            x_out = self.att(x_out)
            x_out = self.ffn(x_out)
        return x_out