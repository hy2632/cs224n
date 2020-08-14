import math
from util import get_available_devices
import torch
import torch.nn as nn
import torch.nn.functional as F

class Positional_Encoding(nn.Module):
    """Implement the PE function.
    Reference:
        (http://nlp.seas.harvard.edu/2018/04/03/attention.html)
    """
    def __init__(self, input_dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, input_dim)
        # https://stackoverflow.com/questions/52922445/runtimeerror-exp-not-implemented-for-torch-longtensor?rq=1 
        # "exp" not implemented for 'torch.LongTensor'
        position = torch.arange(0., max_len).unsqueeze(1) # 0 to 0. (max_len, 1)
        # 10000^(-2i/d_model)
        div_term = torch.exp(torch.arange(0., input_dim, 2) * -(math.log(10000.0) / input_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) #(1, seq_len, d_model)
        # self.register_buffer('pe', pe)
        device, _ = get_available_devices()
        self.pe = self.pe.to(device)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # pe: (1, max_seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
        