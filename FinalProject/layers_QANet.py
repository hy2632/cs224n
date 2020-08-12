"""Assortment of layers for use in QANet.

Author:
    Hua Yao (hy2632@columbia.edu)
"""

import math
import torch
from torch import dropout
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import layer_norm

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

# QANet包含：
# 1. Input Embedding Layer
# 2. Embedding Encoder Layer
# 3. C2Q Attention Layer
# 4. Model Encoder Layer
# 5. Output Layer


# 1. Input Embedding Layer
# word_emb 和 char_emb， 从BiDAF_Char粘贴并微调。
# char_emb需要做到：每个word只保留一个char_dim的向量，不再用CNN(及其torch.max)而是直接用torch.max
# 也包含一个两层highway
class Word_Embedding(nn.Module):
    """Word Embedding layer used by BiDAF

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors. Shape: (word_vocab_size=88714, word_dim=300)
        drop_prob (float): Probability of zero-ing out activations
        
    return:
        word_emb (torch.Tensor): (batch_size, seq_len, word_dim)
    """
    def __init__(self, word_vectors, drop_prob):
        super(Word_Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.embed(x)  # (batch_size, seq_len, word_dim)
        x = self.dropout(x)
        return x


class Char_Embedding(nn.Module):
    """Character-level Embedding layer used by BiDAF

    Obtain the char-level embedding of each word using CNN. input channel size = output ~

    Args:
        char_vocab_size (int): # of chars, in char2idx.json: 1376
        char_dim (int): args.char_dim, default=64
        drop_prob (float): Probability of zero-ing out activations
        hidden_size(int)
        kernel_size (int): parameter in CNN for char_emb
        padding (int): parameter in CNN for char_emb
    
    return:
        word_emb (torch.Tensor): (batch_size, seq_len, char_dim)

    Reference: https://github.com/galsang/BiDAF-pytorch/blob/master/model/model.py
    """
    def __init__(
        self,
        char_vocab_size,
        char_dim,
        drop_prob,
    ):
        super(Char_Embedding, self).__init__()
        # https://github.com/galsang/BiDAF-pytorch/blob/master/model/model.py
        # https://github.com/hy2632/cs224n/blob/master/a5_public/model_embeddings.py

        self.char_emb = nn.Embedding(char_vocab_size, char_dim,
                                     padding_idx=0)  # 08/10 此处应改成 0.
        # char_vocab_size = len(char2idx) = 1376, char_dim = 64(default) in args
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        self.cnn = CNN(char_dim, char_dim, 5, 2) 
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        """
        @param x: (bs, seq_len, word_len)
        return: (bs, seq_len, char_channel_size=char_dim)

        """
        x = self.char_emb(x)  # (batch_size, seq_len, word_len, char_dim)
        x = self.cnn(x) #(batch_size, seq_len, char_dim) 
        # x = torch.max(x, dim=2, keepdim=True)[0].squeeze(
        #     dim=2)  # (batch_size, seq_len, char_dim)
        x = self.dropout(x)
        return x


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network. 
       Adjusted in order to combine char-level & word-level embeddings.

    Args:
        num_layers (int): Number of layers in the highway encoder.
        word_dim (int): dim of word embedding vectors, fixed as p1=300
        char_dim (int): dim of char embedding vectors, default as p2=200
    """
    def __init__(self, num_layers, word_dim=300, char_dim=200):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([
            nn.Linear(word_dim + char_dim, word_dim + char_dim)
            for _ in range(num_layers)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(word_dim + char_dim, word_dim + char_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x1, x2):
        """
        Args:
            @param x1: (batch_size, seq_len, word_dim)
            @param x2: (batch_size, seq_len, char_dim)
        
        return: 
            x: (batch_size, seq_len, word_dim+char_dim)
        """
        x = torch.cat([x1, x2],
                      dim=-1)  # x (batch_size, seq_len, word_dim+char_dim)

        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, word_dim+char_dim)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
        return x


# 2. Embedding Encoder Layer
# [convolution-layer × # + self-attention-layer + feed-forward-layer] => inside a residual block
# kernel_size = 7, filters=128, conv layers = 4.
# number of heads = 8


class EmbeddingEncoderBlock(nn.Module):
    def __init__(self,
                 word_dim,
                 char_dim,
                 d_model,
                 drop_prob=0.,
                 kernel_size=7,
                 padding=3,
                 num_conv=4,
                 head_num=8,
                 maximum_context_length=400):

        super().__init__()

        self.num_conv = num_conv

        # init只在最开始将 p1+p2=500 转换到 d_model=128 时使用
        # 在posenc前，不用residual_block
        self.cnn_init = CNN(input_dim=word_dim + char_dim,
                            kernel_size=kernel_size,
                            padding=padding,
                            filters=d_model,
                            residual_block=False)

        self.posenc = PositionalEncoding(input_dim=d_model,
                                         dropout=drop_prob,
                                         max_len=maximum_context_length)
        
        self.cnn = CNN(input_dim=d_model,
                       kernel_size=kernel_size,
                       padding=padding,
                       filters=d_model,
                       residual_block=True)

        self.layernorm = LayerNorm(d_model)

        self.att = MultiHeadedAttention(h=head_num,
                                        d_model=d_model,
                                        dropout=drop_prob,
                                        residual_block=True)
        self.ffn = FFN(input_dim=d_model,
                       output_dim=d_model,
                       dropout=drop_prob,
                       residual_block=True)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, word_dim+char_dim)
        # For an input x and a given operation f, the output is f(layernorm(x)) +x,
        
        x = self.cnn_init(x)
        x = self.posenc(x)
        for i in range(self.num_conv):
            x = self.cnn(self.layernorm(x))
        x = self.att(self.layernorm(x), mask)
        x = self.ffn(self.layernorm(x))
        return x


class PositionalEncoding(nn.Module):
    """Implement the PE function.
    Reference:
        (http://nlp.seas.harvard.edu/2018/04/03/attention.html)
    """
    def __init__(self, input_dim, dropout, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, input_dim)
        # https://stackoverflow.com/questions/52922445/runtimeerror-exp-not-implemented-for-torch-longtensor?rq=1 
        # "exp" not implemented for 'torch.LongTensor'
        position = torch.arange(0., max_len).unsqueeze(1) # 0 to 0. (max_len, 1)
        div_term = torch.exp(
            torch.arange(0., input_dim, 2) * # 0 to 0.
            -(math.log(10000.0) / input_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)],
                                        requires_grad=False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details).
    
    Reference:
        (http://nlp.seas.harvard.edu/2018/04/03/attention.html)
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Convolution-layer:
class CNN(nn.Module):
    """ CNN, mapping x to x_conv_out \n

    Args:
        @param input_dim (int): first cnn default as word_dim+char_dim=p1+p2=500, cnn 2-4 as 128\n
        @param kernel_size (int): default as 7 \n
        @param padding(int): default as 3
        @param filters (int): number of filters, default as 128

        @param residual_block (bool): True within a block
    """
    def __init__(
        self,
        input_dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        filters: int = 128,
        residual_block: bool = True
    ):
        super(CNN, self).__init__()
        self.residual_block = residual_block
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x):
        """
        Args:
            @param x (Tensor): with shape of (batch_size, seq_len, input_dim=500/128) \n
        return:
            x_conv_out (torch.Tensor): with shape of (batch_size, seq_len, filters=128)
        """
        x_conv_out = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.residual_block:
            return x_conv_out + x
        else:
            return x_conv_out



class MultiHeadedAttention(nn.Module):
    """
    Reference:
        (http://nlp.seas.harvard.edu/2018/04/03/attention.html)
    """
    def __init__(self, h, d_model, dropout=0., residual_block=True):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.residual_block = residual_block

        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, d_model) for i in range(4)]) # x / q k v 的维度 = head
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """ Compute 'Scaled Dot Product Attention'

        Reference:
            (http://nlp.seas.harvard.edu/2018/04/03/attention.html)
            (https://github.com/BangLiu/QANet-PyTorch/blob/master/model/QANet.py)
        """
        # q, k, v: (batch_size, h=8, seq_len, d_k=16)
        # mask: (batch_size, seq_len)

        d_k = query.size(-1)
        scores = torch.matmul(query, key.permute(0,1,3,2)) \
                / math.sqrt(d_k)
        # scores: (batch_size, h=8, seq_len, seq_len)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
        # (batch_size, h=8, seq_len, d_k=16), (batch_size, h=8, seq_len, seq_len)

    def forward(self, x, mask=None): # 08/11 将query, key, value 三个参数换为 q一个参数
        """
        query = key = value, local attention
        """
        # x: (batch_size, seq_len, d_mod)
        # mask: (batch_size, seq_len)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # (batch_size, 1, seq_len)
        nbatches = x.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(i).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, i in zip(self.linears, (x, x, x))] # 8/11 注意这里三者相同
        # q, k, v: (batch_size, h=8, seq_len, d_k=16)

        # 2) Apply attention on all the projected vectors in batch.
        x_out, self.attn = self.attention(query,
                                 key,
                                 value,
                                 mask=mask,
                                 dropout=self.dropout)
        # (batch_size, h=8, seq_len, d_k=16), (batch_size, h=8, seq_len, seq_len)

        # 3) "Concat" using a view and apply a final linear.
        x_out = x_out.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x_out = self.linears[-1](x_out)
        
        if self.residual_block:
            return x_out + x
        else:
            return x_out

class FFN(nn.Module):
    """ Position-wise Feed-Forward layer in the Encoder Block
    
    Based on the paper:
    "Attention is all you need"
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al.
    (https://arxiv.org/abs/1706.03762)

    """
    def __init__(self, input_dim, output_dim, dropout=0., residual_block = True):
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.w1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w2 = nn.Linear(input_dim, output_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.residual_block = residual_block

    def forward(self, x):
        assert x.size(2) == self.input_dim
        x_out = F.relu(self.w1(x))
        x_out = self.dropout(x_out)
        x_out = self.w2(x_out)
        
        if self.residual_block:
            return x_out + x
        else:
            return x_out


# 3. C2Q Attention Layer
class CQAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.):
        super(CQAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob,
                      self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob,
                      self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)

        # 8/6 维度分析
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        # c:(bs, c_len, h); self.c_weight: (h, 1); mm: (bs, c_len, 1); after expand: (bs, c_len, q_len)
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        # (bs, q_len, 1) => (bs, 1, q_len) => (bs, c_len, q_len)
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        # (bs, c_len, h) * (1,1,h) => (bs, c_len, h), * (bs, h, q_len) => (bs, c_len, q_len)
        s = s0 + s1 + s2 + self.bias

        return s


# 4. Model Encoder Layer
class ModelEncoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 drop_prob=0.,
                 kernel_size=5,
                 padding=2,
                 num_conv=2,
                 head_num=8,
                 maximum_context_length=1000):
        super().__init__()
        self.num_conv = num_conv

        self.posenc = PositionalEncoding(input_dim=4 * d_model,
                                         dropout=drop_prob,
                                         max_len=maximum_context_length)

        self.cnn = CNN(input_dim=4 * d_model,
                       kernel_size=kernel_size,
                       padding=padding,
                       filters=4 * d_model,
                       residual_block=True)

        self.layernorm = LayerNorm(4 * d_model)

        self.att = MultiHeadedAttention(h=head_num,
                                        d_model=4 * d_model,
                                        dropout=drop_prob,
                                        residual_block=True)
        self.ffn = FFN(input_dim=4 * d_model,
                       output_dim=4 * d_model,
                       dropout=drop_prob,
                       residual_block=True)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, word_dim+char_dim)
        # For an input x and a given operation f, the output is f(layernorm(x)) +x,
        x = self.posenc(x)

        for i in range(self.num_conv):
            x = self.cnn(self.layernorm(x))

        x = self.att(self.layernorm(x), mask)
        x = self.ffn(self.layernorm(x))

        return x


# 5. Output Layer
class Output(nn.Module):
    def __init__(self, hidden_size, drop_prob):
        super().__init__()
        self.w1 = nn.Linear(8 * hidden_size, 1)
        self.w2 = nn.Linear(8 * hidden_size, 1)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, M0, M1, M2, mask):
        # 08/10: mod: (batch_size, seq_len, 4h)
        # mask: (batch_size, seq_len, 1)

        logits_1 = F.softmax(self.w1(torch.cat([M0, M1], dim=-1)), dim=-1)
        logits_2 = F.softmax(self.w1(torch.cat([M0, M2], dim=-1)), dim=-1)

        logits_1 = self.dropout(logits_1)
        logits_2 = self.dropout(logits_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2