import torch
import torch.nn as nn
import torch.nn.functional as F
from util import masked_softmax


class Output(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(2 * d_model, 1, bias=False)
        self.w2 = nn.Linear(2 * d_model, 1, bias=False)

    def forward(self, M0, M1, M2, mask):
        # 08/10: mod: (batch_size, seq_len, h)
        # mask: (batch_size, seq_len, 1)

        logits_1 = self.w1(torch.cat([M0, M1], dim=-1))
        logits_2 = self.w1(torch.cat([M0, M2], dim=-1))

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2