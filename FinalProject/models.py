"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

from torch.nn.functional import pad
import layers
import torch
import torch.nn as nn

# import layers_QANet


# class BiDAF(nn.Module):
#     """Baseline BiDAF model for SQuAD.

#     Based on the paper:
#     "Bidirectional Attention Flow for Machine Comprehension"
#     by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
#     (https://arxiv.org/abs/1611.01603).

#     Follows a high-level structure commonly found in SQuAD models:
#         - Embedding layer: Embed word indices to get word vectors.
#         - Encoder layer: Encode the embedded sequence.
#         - Attention layer: Apply an attention mechanism to the encoded sequence.
#         - Model encoder layer: Encode the sequence again.
#         - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

#     Args:
#         word_vectors (torch.Tensor): Pre-trained word vectors.
#         hidden_size (int): Number of features in the hidden state at each layer.
#         drop_prob (float): Dropout probability.
#     """
#     def __init__(self, word_vectors, hidden_size, drop_prob=0.):
#         super(BiDAF, self).__init__()
#         self.emb = layers.Embedding(word_vectors=word_vectors,
#                                     hidden_size=hidden_size,
#                                     drop_prob=drop_prob)

#         self.enc = layers.RNNEncoder(input_size=hidden_size,
#                                      hidden_size=hidden_size,
#                                      num_layers=1,
#                                      drop_prob=drop_prob)

#         self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
#                                          drop_prob=drop_prob)

#         self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
#                                      hidden_size=hidden_size,
#                                      num_layers=2,
#                                      drop_prob=drop_prob)

#         self.out = layers.BiDAFOutput(hidden_size=hidden_size,
#                                       drop_prob=drop_prob)

#     def forward(self, cw_idxs, qw_idxs):
#         c_mask = torch.zeros_like(cw_idxs) != cw_idxs
#         q_mask = torch.zeros_like(qw_idxs) != qw_idxs
#         c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

#         c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
#         q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

#         c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
#         q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

#         att = self.att(c_enc, q_enc, c_mask,
#                        q_mask)  # (batch_size, c_len, 8 * hidden_size)

#         mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

#         out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

#         return out


class BiDAF_Char(nn.Module):
    """Char-level BiDAF model for SQuAD. 

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    high-level structure of 6 layers:
        - Character Embedding layer: maps each word to a vector space using character-level CNNs.
        - Word Embedding layer: maps each word to a vector space using a pre-trained word embedding model.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors. Shape: (word_vocab_size=88714, word_dim=300)
        char_vectors (torch.Tensor): Pre-trained char vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self,
                 word_vectors,
                 char_vocab_size,
                 char_dim,
                 hidden_size,
                 drop_prob=0.,
                 kernel_size=5,
                 padding=1):
        super(BiDAF_Char, self).__init__()

        self.char_emb = layers.Char_Embedding(char_vocab_size=char_vocab_size,
                                              char_dim=char_dim,
                                              drop_prob=drop_prob,
                                              hidden_size=hidden_size,
                                              kernel_size=kernel_size,
                                              padding=padding)

        self.word_emb = layers.Word_Embedding(word_vectors=word_vectors,
                                              hidden_size=hidden_size,
                                              drop_prob=drop_prob)

        self.hwy = layers.HighwayEncoder(num_layers=2, hidden_size=hidden_size)

        self.enc = layers.RNNEncoder(input_size=2 * hidden_size, # 08/09 注意这里改了input_size。因为经过highway后char_emb+word_emb的concatenation (bs, seq_len, 2*h)
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        # cw_idxs: (batch_size, c_len, word_dim)
        # cc_idxs: (batch_size, c_len, word_len, char_dim)
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        cw_emb = self.word_emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        qw_emb = self.word_emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        cc_emb = self.char_emb(cc_idxs) # (batch_size, c_len, hidden_size)
        qc_emb = self.char_emb(qc_idxs) # (batch_size, q_len, hidden_size)

        c_emb = self.hwy(cw_emb, cc_emb) # (batch_size, c_len, 2 * hidden_size)
        q_emb = self.hwy(qw_emb, qc_emb) # (batch_size, q_len, 2 * hidden_size)

        c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc, c_mask,
                       q_mask)  # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out






# # 增加 QANet 类
# class QANet (nn.Module):
#     """ QANet model for SQuAD.

#     Based on the paper:
#     "QANET: COMBINING LOCAL CONVOLUTION WITH GLOBAL SELF-ATTENTION FOR READING COMPREHENSION"
#     by Adams Wei Yu , David Dohan, Minh-Thang Luong
#     (https://arxiv.org/pdf/1804.09541).

#     """
#     def __init__(self, word_vectors, char_vectors, hidden_size=128, f=128, k=5, e_char=200,
#                 e_word = 300, m_word = 16,):
#         super(QANet, self).__init__()

#         self.char_emb = torch.randn()
