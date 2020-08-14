import torch
import torch.nn as nn
import torch.nn.functional as F


# class Char_Embedding(nn.Module):
#     """Character-level Embedding layer used by BiDAF

#     Obtain the char-level embedding of each word using CNN. input channel size = output ~

#     Args:
#         char_vocab_size (int): # of chars, in char2idx.json: 1376
#         char_dim (int): args.char_dim, default=64
#         drop_prob (float): Probability of zero-ing out activations
#         hidden_size(int)
#         kernel_size (int): parameter in CNN for char_emb
#         padding (int): parameter in CNN for char_emb
    
#     return:
#         word_emb (torch.Tensor): (batch_size, seq_len, char_dim)

#     Reference: https://github.com/galsang/BiDAF-pytorch/blob/master/model/model.py
#     """
#     def __init__(
#         self,
#         char_vocab_size,
#         char_dim,
#         drop_prob=0.05,
#     ):
#         super(Char_Embedding, self).__init__()
#         # https://github.com/galsang/BiDAF-pytorch/blob/master/model/model.py
#         # https://github.com/hy2632/cs224n/blob/master/a5_public/model_embeddings.py

#         self.char_emb = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
#         nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
#         # self.cnn = CNN_for_Char_Embedding(char_dim, char_dim, 5, 2)
#         self.dropout = nn.Dropout(drop_prob)

#     def forward(self, x):
#         """
#         @param x: (bs, seq_len, word_len)
#         return: (bs, seq_len, char_channel_size=char_dim)

#         """
#         x = self.char_emb(x)  # (batch_size, seq_len, word_len, char_dim)
#         # x = self.cnn(x) #(batch_size, seq_len, char_dim)
#         x = torch.max(x, dim=2, keepdim=True)[0].squeeze(
#             dim=2)  # (batch_size, seq_len, char_dim)
#         x = self.dropout(x)
#         return x


# class Word_Embedding(nn.Module):
#     """Word Embedding layer used by BiDAF

#     Word-level embeddings are further refined using a 2-layer Highway Encoder
#     (see `HighwayEncoder` class for details).

#     Args:
#         word_vectors (torch.Tensor): Pre-trained word vectors. Shape: (word_vocab_size=88714, word_dim=300)
#         drop_prob (float): Probability of zero-ing out activations
        
#     return:
#         word_emb (torch.Tensor): (batch_size, seq_len, word_dim)
#     """
#     def __init__(self, word_vectors, drop_prob=0.1):
#         super(Word_Embedding, self).__init__()
#         self.drop_prob = drop_prob
#         self.embed = nn.Embedding.from_pretrained(word_vectors)
#         self.dropout = nn.Dropout(drop_prob)

#     def forward(self, x):
#         x = self.embed(x)  # (batch_size, seq_len, word_dim)
#         x = self.dropout(x)
#         return x


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

    def forward(self, x):
        """
        Args:
            @param x([x1,x2]): (batch_size, seq_len, word_dim+char_dim)
        
        return: 
            x: (batch_size, seq_len, word_dim+char_dim)
        """
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, word_dim+char_dim)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
        return x


# Word_emb + Char_emb + Highway
class Embedding(nn.Module):
    def __init__(self, word_vectors, char_vocab_size, word_dim, char_dim,
                 drop_prob_word, drop_prob_char):
        super().__init__()

        self.drop_prob_char = drop_prob_char
        self.drop_prob_word = drop_prob_word

        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        self.char_emb = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        self.hwy = HighwayEncoder(2, word_dim, char_dim)
    
    def forward(self, w_idxs, c_idxs):
        
        e_word = self.word_emb(w_idxs)
        e_char = self.char_emb(c_idxs)
        emb = torch.cat([F.dropout(e_word, self.drop_prob_word, self.training), e_char])
        emb = self.hwy(emb)
        return emb