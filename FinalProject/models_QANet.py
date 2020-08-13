import torch
import torch.nn as nn
from layers_QANet.Embedding import *
from layers_QANet.EmbeddingEncoderBlock import EmbeddingEncoderBlock
from layers_QANet.C2QAttention import C2QAttention
from layers_QANet.ModelEncoderBlocks import ModelEncoderBlocks
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
                maximum_context_length=400):
        super(QANet, self).__init__()
        word_dim = word_vectors.size(1)

        self.word_emb = Word_Embedding(word_vectors, drop_prob)
        self.char_emb = Char_Embedding(char_vocab_size, char_dim, drop_prob)
        self.hwy = HighwayEncoder(num_layers=2, word_dim=word_vectors.size(1), char_dim=char_dim)

        self.emb_enc_block = EmbeddingEncoderBlock(
                word_dim=word_dim,
                char_dim=char_dim,
                d_model=d_model,
                maximum_context_length=maximum_context_length)
        
        self.c2q_att = C2QAttention(dim=d_model)

        self.mod_enc_blocks = ModelEncoderBlocks(
                d_model = d_model,
                num_blocks = num_mod_blocks,
                maximum_context_length=maximum_context_length)

        self.out = Output(d_model=d_model, drop_prob=drop_prob)
        

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        # cw_idxs: (batch_size, c_len, 1)
        # cc_idxs: (batch_size, c_len, char_dim, 1) 不存在字符 --OOV-- 的情况。
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        cw_emb = self.word_emb(cw_idxs)  # (batch_size, c_len, word_dim)
        qw_emb = self.word_emb(qw_idxs)  # (batch_size, q_len, word_dim)
        cc_emb = self.char_emb(cc_idxs)  # (batch_size, c_len, char_dim)
        qc_emb = self.char_emb(qc_idxs)  # (batch_size, q_len, char_dim)
        c_emb = self.hwy(cw_emb, cc_emb)  # (batch_size, c_len, )
        q_emb = self.hwy(qw_emb, qc_emb)  # (batch_size, q_len, word_dim+char_dim)

        c_enc = self.emb_enc_block(c_emb, c_mask)
        q_enc = self.emb_enc_block(q_emb, q_mask)
        c2q_att = self.c2q_att(c_enc, q_enc, c_mask, q_mask)

        m0 = self.mod_enc_blocks(c2q_att) ## 取消添加 mask
        m1 = self.mod_enc_blocks(m0)
        m2 = self.mod_enc_blocks(m1)

        out = self.out(m0, m1, m2, c_mask) # logp1, logp2

        return out
