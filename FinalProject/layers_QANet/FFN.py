import torch
import torch.nn as nn
import torch.nn.functional as F

from layers_QANet.LayerNorm import LayerNorm

class FFN(nn.Module):
    """ Position-wise Feed-Forward layer in the Encoder Block
    
    Based on the paper:
    "Attention is all you need"
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al.
    (https://arxiv.org/abs/1706.03762)

    """
    def __init__(self, input_dim, output_dim, drop_prob=0.):
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.w1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w2 = nn.Linear(input_dim, output_dim, bias=True)
        self.layernorm = LayerNorm(input_dim)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.dropout(x)
        x_out = self.layernorm(x)
        x_out = F.relu(self.w1(x_out),x_out)
        x_out = self.w2(x_out)
        
        return x_out + x
