import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.dropout import Dropout
from layers_QANet.Emb import Embedding
from layers_QANet.EmbEncBlock import Embedding_Encoder
from layers_QANet.C2QAttention import C2QAttention
from layers_QANet.ModEncBlocks import Model_Encoder
from layers_QANet.Output import Output


class QANet(nn.Module):
    """ QANet model for SQuAD.

    Based on the paper:
    "QANET: COMBINING LOCAL CONVOLUTION WITH GLOBAL SELF-ATTENTION FOR READING COMPREHENSION"
    by Adams Wei Yu , David Dohan, Minh-Thang Luong
    (https://arxiv.org/pdf/1804.09541).

    """
    def __init__(self,
                 word_vectors,
                 char_vocab_size,
                 char_dim,
                 d_model,
                 drop_prob=0.,
                 num_mod_blocks=7,
                 maximum_context_length=600):
        super().__init__()

        word_dim = word_vectors.size(1)
        self.drop_prob = drop_prob

        self.emb = Embedding(word_vectors, char_vocab_size, word_dim, char_dim,
                             0.1, 0.05)
        # 300+200 -> 128
        self.emb_proj = nn.Linear(word_dim + char_dim, d_model)

        self.emb_enc = Embedding_Encoder(
            dim=d_model,
            maximum_context_length=maximum_context_length,
            num_conv=4,
            kernel_size=7,
            num_heads=4) # 节省内存=============================
        # x_c_out & x_q_out

        self.c2q_att = C2QAttention(dim=d_model)

        # 4*128 -> 128
        self.att_proj = nn.Linear(4*d_model, d_model)

        self.mod_enc = Model_Encoder(
            dim=d_model,
            maximum_context_length=maximum_context_length,
            num_conv=2,
            kernel_size=5,
            num_heads=4,
            num_blocks=num_mod_blocks)# 节省内存=============================
        # m0, m1, m2
        
        self.out = Output(d_model=d_model)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        # cw_idxs: (batch_size, c_len)
        # cc_idxs: (batch_size, c_len, char_dim)
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.emb(cw_idxs, cc_idxs) #(batch_size, c_len, word_dim+char_dim)
        q_emb = self.emb(qw_idxs, qc_idxs) 

        # Dropout
        c_emb = F.dropout(c_emb, self.drop_prob, self.training)
        q_emb = F.dropout(q_emb, self.drop_prob, self.training)

        # Linear Projection
        c_emb = F.relu(self.emb_proj(c_emb))
        q_emb = F.relu(self.emb_proj(q_emb))

        c_enc, q_enc = self.emb_enc(c_emb, q_emb, c_mask, q_mask)
        c2q_att = self.c2q_att(c_enc, q_enc, c_mask, q_mask)

        # Dropout
        c2q_att = F.dropout(c2q_att, self.drop_prob, self.training)

        # Linear Projection
        c2q_att = F.relu(self.att_proj(c2q_att))
        
        # Model Encoder Blocks
        m0, m1, m2 = self.mod_enc(c2q_att, c_mask)

        # Dropout
        m0 = F.dropout(m0, self.drop_prob, self.training)
        m1 = F.dropout(m1, self.drop_prob, self.training)
        m2 = F.dropout(m2, self.drop_prob, self.training)

        out = self.out(m0, m1, m2, c_mask)  # logp1, logp2
        return out
