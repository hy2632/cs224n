import torch.nn as nn

from layers_QANet.PositionalEncoding import PositionalEncoding
from layers_QANet.CNN import depthwise_separable_convolution
from layers_QANet.SelfAttention import SelfAttention
from layers_QANet.FFN import FFN

class EmbeddingEncoderBlock(nn.Module):
    def __init__(self,
                 word_dim=300,
                 char_dim=200,
                 d_model=128,
                 maximum_context_length=400):

        super().__init__()

        self.maximum_context_length = maximum_context_length

        # init只在最开始将 p1+p2=500 转换到 d_model=128 时使用
        # 在posenc前，不用residual_block
        self.cnn_init = nn.Conv1d(word_dim+char_dim, d_model, 5, 1, 2, bias=True)
        self.posenc = PositionalEncoding(input_dim=d_model,
                                         max_len=maximum_context_length)
        
        self.cnn = depthwise_separable_convolution(input_dim=d_model, output_dim=d_model, num_convs=4, drop_prob=0.1)
        self.att = SelfAttention(n_heads=8, d_model=d_model)
        self.ffn = FFN(input_dim=d_model,
                       output_dim=d_model,
                       drop_prob=0.1)

    def forward(self, x):
        # x: (batch_size, seq_len, word_dim+char_dim)
        x_out = self.cnn_init(x.permute(0,2,1)).permute(0,2,1)
        if x_out.size(1) > self.maximum_context_length:
            raise ValueError("seq_len > maximum_context_length")
        x_out = self.posenc(x_out)
        x_out = self.cnn(x_out)
        x_out = self.att(x_out)
        x_out = self.ffn(x_out)
        return x_out