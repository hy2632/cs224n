import torch
import torch.nn as nn
import torch.nn.functional as F

from util import masked_softmax


class C2QAttention(nn.Module):
    """ Context-Query Attention.
    Take 
    c: (batch_size, c_len, d_model)
    q: (batch_size, q_len, d_model)

    return:
    A: (batch_size, c_len, d_model)

    Ref:
        (https://github.com/marquezo/qanet-impl)
    """
    
    def __init__(self, dim = 128):
        super().__init__()
        self.dim = dim
        self.c_weight = nn.Parameter(torch.zeros(dim, 1), requires_grad=True)
        self.q_weight = nn.Parameter(torch.zeros(dim, 1), requires_grad=True)
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def get_similarity_matrix(self, c, q, c_len, q_len):
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        # c:(bs, c_len, h); self.c_weight: (h, 1); mm: (bs, c_len, 1); after expand: (bs, c_len, q_len)
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2).expand([-1, c_len, -1])
        # (bs, q_len, 1) => (bs, 1, q_len) => (bs, c_len, q_len)
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        # (bs, c_len, h) * (1,1,h) => (bs, c_len, h), * (bs, h, q_len) => (bs, c_len, q_len)
        s = s0 + s1 + s2 + self.bias
        return s

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = tuple(c.size())
        _, q_len, _ = tuple(q.size())
        S = self.get_similarity_matrix(c, q, c_len, q_len) # (bs, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1).contiguous()  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len).contiguous()   # (batch_size, 1, q_len)

        # softmax each row (dim 2: q_len, need q_mask)
        S_ = masked_softmax(S, q_mask, dim=2) # (batch_size, c_len, q_len)
        S__ = masked_softmax(S_, c_mask, dim=1) # (batch_size, c_len, q_len)
        A = torch.bmm(S_, q) # (bs, c_len, q_len)*(bs, q_len, d_model) = (bs, c_len, d_model)
        B = torch.bmm(torch.bmm(S_, S__.permute(0,2,1)), c) # (bs, c_len, q_len)*(bs, q_len, c_len)*(bs, c_len, d_model) = (bs, c_len, d_model)
        x = torch.cat([c, A, c * A, c * B], dim=2)  # (bs, c_len, 4 * d_model)
    
        return x


