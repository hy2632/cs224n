# """Assortment of layers for use in models.py.

# Author:
#     Hua Yao (hy2632@columbia.edu)
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from util import masked_softmax




# class Embedding(nn.Module):
#     """ Embedding layer used by QANet, including word level & char level embeddings.

#     Args:
#         word_vectors: Pre-trained word vectors from GloVe(p1 = 300 dimensional), fixed
#         char_vectors: p2 = 200, trainable, word length truncated or padded to 16
#         hidden_size: for HighwayEncoder
#         k(int): kernel size for CNN
#         stride(int): stride for CNN
#     return:
#         x = [xw; xc]: R(p1+p2), word embedding and convolution output of char embedding
#     """

#     def __init__(self, word_vectors, char_vectors, hidden_size=128, f=128, k=5, e_char=200, 
#                 e_word = 300, m_word = 16,
#     ):
#         super(Embedding, self).__init__()
#         self.word_embed = nn.Embedding.from_pretrained(word_vectors) # (bs, seq_len, embed_size=p1=300)
#         self.char_embed = nn.Embedding.from_pretrained(char_vectors) # (bs, )       
#         self.cnn = CNN(f, k, e_char)
#         self.hwy = HighwayEncoder(2, hidden_size)


#     def forward(self, x):
#         pass
#         # xw = self.word_embed(x) # (bs, seq_len, embed_size)
#         # xc = self.char_embed(word)

# # char_embedding需要CNN
# class CNN(nn.Module):
#     """ CNN, mapping x_reshaped to x_conv_out \n
#         @param f (int): number of filters \n
#         @param k (int): default as k=5 \n
#         @param e_char (int): default as echar = 200
#     """
#     def __init__(self, f:int=200, k:int=5, e_char:int=200):
#         super(CNN, self).__init__()
#         self.f = f
#         self.conv1d = nn.Conv1d(e_char, f, k)
    
#     def forward(self, x_reshaped: torch.Tensor)->torch.Tensor:
#         """ Map from x_reshaped to x_conv_out\n
#             @param x_reshaped (Tensor): Tensor with shape of (sentence_length, batch_size, e_char, m_word) \n
#             @return x_conv_out (Tensor) : Tensor with shape of (sentence_length, batch_size,  e_word=f) \n
#         """
#         (sentence_length, batch_size, e_char, m_word) = tuple(x_reshaped.size())
#         x_conv = self.conv1d(x_reshaped.contiguous().view(sentence_length*batch_size, e_char, m_word))
#         x_conv_out = torch.max(F.relu(x_conv), dim=2)[0].contiguous().view(sentence_length, batch_size, self.f)
#         return x_conv_out

# class HighwayEncoder(nn.Module):
#     """Encode an input sequence using a highway network.

#     Based on the paper:
#     "Highway Networks"
#     by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
#     (https://arxiv.org/abs/1505.00387).

#     Args:
#         num_layers (int): Number of layers in the highway encoder.
#         hidden_size (int): Size of hidden activations.
#     """
#     def __init__(self, num_layers, hidden_size):
#         super(HighwayEncoder, self).__init__()
#         self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
#                                          for _ in range(num_layers)])
#         self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
#                                     for _ in range(num_layers)])

#     def forward(self, x):
#         for gate, transform in zip(self.gates, self.transforms):
#             # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
#             g = torch.sigmoid(gate(x))
#             t = F.relu(transform(x))
#             x = g * t + (1 - g) * x

#         return x


# class EmbeddingEncoder(nn.module):

# class Attention(nn.module):

# class ModelEncoder(nn.module):

# class Output(nn.module):
