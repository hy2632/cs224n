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

class CNN(nn.Module):
    """ CNN, mapping x_reshaped to x_conv_out \n
        Mostly from Assignment 5. Changed the shape of input to align with the project.

        @param filters (int): number of filters \n
        @param kernel_size (int): default as k=5 \n
        @param stride (int): default as stride=1 \n
    """

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(
        self,
        f: int,
        e_char: int = 50,
        k: int = 5,
        padding: int = 1,
    ):

        super(CNN, self).__init__()
        self.f = f
        self.conv1d = nn.Conv1d(
            in_channels=e_char,
            out_channels=f,
            kernel_size=k,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x_reshaped: torch.Tensor,
    ) -> torch.Tensor:
        """ Map from x_reshaped to x_conv_out\n
            @param x_reshaped (Tensor): Tensor with shape of (batch_size, sentence_length, m_word, e_char) \n
            @return x_conv_out (Tensor) : Tensor with shape of (batch_size, sentence_length, e_word=f) \n
        """
        (batch_size, sentence_length, m_word, e_char) = tuple(x_reshaped.size())
        x_conv = self.conv1d(x_reshaped.contiguous().view(sentence_length*batch_size, e_char, m_word))
        x_conv_out = torch.max(F.relu(x_conv), dim=2)[0].contiguous().view(batch_size, sentence_length, self.f)
        return x_conv_out

# Word_emb + Char_emb + Highway + CNN
class Embedding(nn.Module):
    # def __init__(self, word_vectors, char_vocab_size, word_dim, char_dim,
    #              drop_prob_word, drop_prob_char):
    #     super().__init__()

    #     self.drop_prob_char = drop_prob_char
    #     self.drop_prob_word = drop_prob_word

    #     self.word_emb = nn.Embedding.from_pretrained(word_vectors)
    #     self.char_emb = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
    #     nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
    #     # ===========================================
    #     self.cnn=CNN(char_dim, char_dim, 5, 2)
    #     # ===========================================        
    #     self.hwy = HighwayEncoder(2, word_dim, char_dim)

    def __init__(self, word_vectors, char_vocab_size, word_dim, char_dim, d_model,
                 drop_prob_word, drop_prob_char, drop_prob_general):
        super().__init__()

        self.drop_prob_char = drop_prob_char
        self.drop_prob_word = drop_prob_word

        self.drop_prob_general = drop_prob_general

        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        self.char_emb = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
        
        # QANet-03 增加: 我们希望char_emb初始都能大于0， 这样max(relu(), dim=2(seq_len wise))在初始能起到效果： 失败
        # nn.init.uniform_(self.char_emb.weight, 0, 1)
        # nn.init.xavier_uniform_(self.char_emb.weight)

        # ===========================================
        self.cnn=CNN(char_dim, char_dim, 5, 2)
        # ===========================================        
        self.hwy = HighwayEncoder(2, word_dim, char_dim)

        # QANet-04 08/15==========================================
        self.cnn_proj = nn.Conv1d(char_dim+word_dim, d_model, 1, 1, 0)

    
    def forward(self, w_idxs, c_idxs):
        
        e_word = self.word_emb(w_idxs)
        e_char = self.char_emb(c_idxs)
        e_char = F.dropout(e_char, self.drop_prob_char, self.training)

        # (batch_size, seq_len, m_word, char_dim) = tuple(e_char.size())
        # e_char = e_char.contiguous().view(seq_len*batch_size, char_dim, m_word)
        # e_char = torch.max(F.relu(e_char), dim=2)[0].contiguous().view(batch_size, seq_len, char_dim)

        # QANet-03 去除F.relu 
        # e_char = torch.max(e_char, dim=2)[0].contiguous().view(batch_size, seq_len, char_dim)

        # QANet-04/05/06 08/15==========================================
        e_char = self.cnn(e_char)

        
        emb = torch.cat([F.dropout(e_word, self.drop_prob_word, self.training), e_char], dim=-1)
        emb = self.hwy(emb)

        # QANet-04 08/15==========================================
        emb = self.cnn_proj(emb.permute(0,2,1)).permute(0,2,1)
        emb = F.dropout(emb, self.drop_prob_general, self.training)

        return emb