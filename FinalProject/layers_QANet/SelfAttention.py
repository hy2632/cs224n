import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import masked_softmax

# from layers_QANet.LayerNorm import LayerNorm

import math

class SelfAttention(nn.Module):
    """
    Multi-Head Attention described in "Attention is all you need"
    """
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        self.d_v = self.d_k = d_model // n_heads

        self.fc_query = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k) for i in range(n_heads)])

        self.fc_key = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k) for i in range(n_heads)])

        self.fc_value = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_v) for i in range(n_heads)])
        
        self.fc_out = nn.Linear(d_model, d_model) # last FC layer after concatnation

        # self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = tuple(x.size())

        heads = torch.cuda.FloatTensor(batch_size, seq_len, self.n_heads, self.d_v).fill_(0)    
        # x_out = self.layernorm(x)
        x_out = F.layer_norm(x, [x.size(-1)])

        for i in range(self.n_heads):
            Q = self.fc_query[i](x_out)
            K = self.fc_key[i](x_out)
            V = self.fc_value[i](x_out)
            # multi-head不同head相互独立
            # Q, K, V: (batch_size, seq_len, d_k=d_v)

            # A = softmax(Q*K.T/sqrt(d_k))*V
            tmp = torch.bmm(Q, K.permute(0,2,1)) # (batch_size, seq_len, seq_len)
            tmp = tmp / math.sqrt(self.d_k)
            tmp = F.softmax(tmp, dim=1)
            heads[:,:,i,:] = torch.bmm(tmp,V) # (batch_size, seq_len, d_v)
        
        x_out = heads.view(batch_size, seq_len, -1).contiguous()
        x_out = self.fc_out(x_out)
        return x_out + x

        

        


    