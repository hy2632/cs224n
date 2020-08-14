import torch.nn as nn
import torch.nn.functional as F


class Feed_Forward(nn.Module):
    """ Position-wise Feed-Forward layer in the Encoder Block
    
    Based on the paper:
    "Attention is all you need"
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al.
    (https://arxiv.org/abs/1706.03762)

    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.w1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w2 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))
