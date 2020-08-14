"""Assortment of layers for use in models.py.
Author:
    Axel Moyal (axmoyal@stanford.edu)
    Guillermo Bescos (gbescos@stanford.edu)
    Lucas Soffer (lsoffer@stanford.edu)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax, get_available_devices


class Stochastic_Dropout(nn.Module): 

    def __init__(self, final_survival_prob, depth):
        """Stochastic layer dropout 
            https://arxiv.org/pdf/1603.09382.pdf
            Args: 
                Final_survival_prob: survival probability of the last layer, p_L (in paper)
                depth: Depth of the stack of layer (number of layers)
                dropout_rate: regular drop out rate
        """ 
        super(Stochastic_Dropout, self).__init__()
        ######################## PARAMETERS #############################
        self.final_survival = final_survival_prob  
        self.depth = depth
    
    def forward(self, x, layer, norm, current_depth, multihead = (False, None)):
        """ NOTE: ALL ENCODER BLOCK RELUs ARE PERFORMED HERE
            Args: 
                @param x: skip connection
                @param layer: layer to be implemented
                @param norm: norm layer
                @param current_depth: depth at the current time
                @param multihead: tuple(bool indicating whether "layer" is multihead, mask required for multihead)
        """ 
        if self.training:
            survived = 1 - current_depth/self.depth*(1-self.final_survival) > torch.empty(1,1).uniform_(0,1)
            if survived and multihead[0]:
                temp = norm(x)
                temp = temp.permute(1, 0, 2)
                temp, _ = layer(temp, temp, temp, key_padding_mask= (multihead[1] == False))
                temp = temp.permute(1, 0, 2)
                return x + temp
            elif survived and not multihead[0]: 
                return x + F.relu(layer(norm(x)))
            elif not survived: 
                return x
        else:
            if multihead[0]:
                temp = norm(x)
                temp = temp.permute(1, 0, 2)
                temp, _ = layer(temp, temp, temp, key_padding_mask= (multihead[1] == False))
                temp = temp.permute(1, 0, 2)
                return x + temp
            else:
                return x + F.relu(layer(norm(x)))


class DepthSep_conv(nn.Module):
    #Depthwise seperable convolution layer. is memory efficient and has better generalization
    def __init__(self, in_dim, out_dim, kernel_size):
        """
            Args: 
                num_conv: numb
                er of convolutional layers
                kernel_size: kernel size of CNN 
                num_filters: number of filters in CNN
                embed_size: size of embeding of a word
        """ 
        super(DepthSep_conv, self).__init__()
        self.depth = nn.Conv1d(in_dim, in_dim, kernel_size, padding = int(kernel_size // 2), groups = in_dim, bias = False)
        self.seperate = nn.Conv1d(in_dim, out_dim, kernel_size =1 , padding = 0, bias = True)

        #nn.init.kaiming_normal_(self.depth.weight, nonlinearity = 'relu')
        #nn.init.kaiming_normal_(self.seperate.weight, nonlinearity = 'relu')

    def forward(self, x): 
        return self.seperate(self.depth(x.permute(0,2,1))).permute(0,2,1)


class Encoder_Block(nn.Module): 

    def __init__(self, num_conv, kernel_size, hidden_size, num_heads, final_survival_prob, depth): 
        """QAnet Encoder block 
            https://arxiv.org/pdf/1804.09541.pdf
        Args: 
            @param num_conv: number of convolutional layers
            @param kernel_size: kernel size of CNN 
            @param hidden_size: size of embeding of a word
            @param num_heads: number of heads in multihead attention
            @param final_survival_prob: probability of survival of the FINAL sublayer 
            @param depth: depth of the stack
        """

        super(Encoder_Block, self).__init__()

        ######################## PARAMETERS #############################
        self.num_conv = num_conv


        ##################### Initialize layers #########################

        #Positional encoder
        self.position_encoder = PositionEncoder(hidden_size)

        #stochastic depth layer
        self.stochastic_drop = Stochastic_Dropout(final_survival_prob, depth)

        #Convolutional blocks
        self.conv_layers = nn.ModuleList([DepthSep_conv(hidden_size, hidden_size, kernel_size) for _ in range(num_conv)])
        self.conv_norm_layers = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_conv)])

        #Multihead Attention 
        self.multihead = nn.MultiheadAttention(hidden_size, num_heads)
        self.multihead_norm = nn.LayerNorm(hidden_size)

        #Feedforwad layer
        self.frward = nn.Linear(hidden_size, hidden_size)
        self.forward_norm =nn.LayerNorm(hidden_size)
        #nn.init.kaiming_normal_(self.frward.weight, nonlinearity='relu')


    def forward(self, x, mask, depth_start):
        """QAnet Encoder Block
        Args: 
            @param x: minibatch (batch_size, context/query_length, hidden_size)
            @param mask: context/query mask 
            @param depth_start: depth index at the start of encoder block
        Outputs:
            @returns x: minibatch (batch_size, context/query_length, hidden_size)
        """

        #postional encoder
        x = self.position_encoder(x)

        #convolution section
        for i, conv, norm in zip(range(self.num_conv),self.conv_layers, self.conv_norm_layers):
            x = self.stochastic_drop(x, conv, norm, i + depth_start)

        #Multihead attention section
        x = self.stochastic_drop(x, self.multihead, self.multihead_norm, depth_start + self.num_conv, (True, mask))

        #Linear Section 
        x = self.stochastic_drop(x, self.frward, self.forward_norm, depth_start + self.num_conv + 1) 

        return x


class Embedding_Encoder(nn.Module): 

    def __init__(self, num_blocks, num_conv, kernel_size, hidden_size, num_heads, survival_prob): 
        """QAnet embedding encoder 
           https://arxiv.org/pdf/1804.09541.pdf
        Args: 
            @param num_blocks: number of encoder blocks 
            @param num_conv: number of convolutional layers per encoder block 
            @param kernel_size: kernel size of depthwise seperable convolution
            @param hidden_size: hidden dimension of QAnet model
            @param num_heads: number of heads in multihead attention of encoder block
            @param survival_prob: probability of survival of the FINAL sublayer 
        """
        super(Embedding_Encoder, self).__init__()

        ######################## PARAMETERS #############################
        self.num_blocks = num_blocks
        self.num_conv = num_conv
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.survival_prob = survival_prob
        self.total_depth = num_blocks*(num_conv + 2) - 1


        ##################### Initialize layers #########################
        #stack of Encoder blocks
        self.enc= nn.ModuleList([Encoder_Block( self.num_conv, 
                                                self.kernel_size, 
                                                self.hidden_size, 
                                                self.num_heads, 
                                                self.survival_prob,
                                                self.total_depth) for _ in range(num_blocks)])

    def forward(self, x_context, x_query, c_mask, q_mask):
        """QAnet embedding encoder
        Args: 
            @param x_context: context minibatch (batch_size, context_length,hidden_size)
            @param x_query: query minibatch (batch_size, query_length, hidden_size)
            @param c_mask: context mask 
            @param q_mask: query mask 
        Outputs:
            @returns x_context: (batch_size, context_length, hidden_size)
            @returns x_query: (batch_size, query_length, hidden_size)
        """
        for i, block in zip(range(self.num_blocks), self.enc): 
            x_context = block(x_context, c_mask, i*(self.num_conv + 2))
            x_query = block(x_query, q_mask, i*(self.num_conv + 2))
        return x_context, x_query


class Model_Encoder(nn.Module): 

    def __init__(self, num_blocks, num_conv, kernel_size, hidden_size, num_heads, survival_prob):
        """QAnet model encoder 
           https://arxiv.org/pdf/1804.09541.pdf
        Args: 
            @param num_blocks: number of encoder blocks 
            @param num_conv: number of convolutional layers per encoder block 
            @param kernel_size: kernel size of depthwise seperable convolution
            @param hidden_size: hidden dimension of QAnet model
            @param num_heads: number of heads in multihead attention of encoder block
            @param survival_prob: probability of survival of the FINAL sublayer 
        """ 
        super(Model_Encoder, self).__init__()

        ######################## PARAMETERS #############################
        self.num_conv = num_conv
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.survival_prob = survival_prob
        self.total_depth = num_blocks*(num_conv + 2) - 1

        ##################### Initialize layers #########################
        #Encoder Block Stack
        self.enc = nn.ModuleList([Encoder_Block(self.num_conv, 
                                                self.kernel_size, 
                                                self.hidden_size, 
                                                self.num_heads,
                                                self.survival_prob,
                                                self.total_depth) for _ in range(num_blocks)])
        

    def forward(self, x, mask):
        """QAnet model encoder
        Args: 
            @param x: biderectional attention minibatch (batch_size, context_length, 4*hidden size)
            @param c_mask: context mask 
        Outputs:
            @returns out1: (batch_size, context_length, hidden_size)
            @returns out2: (batch_size, context_length, hidden_size)
            @returns out2: (batch_size, context_length, hidden_size)
        """
        for i, block in zip(range(self.num_blocks), self.enc): 
            x = block(x, mask, i*(self.num_conv + 2))
        out1 = x
        for i, block in zip(range(self.num_blocks), self.enc): 
            x = block(x, mask, i*(self.num_conv + 2))
        out2 = x
        x = out2
        for i, block in zip(range(self.num_blocks), self.enc): 
            x = block(x, mask, i*(self.num_conv + 2))
        out3 = x
        return out1, out2, out3

    
class output_layer(nn.Module): 
    def __init__(self, hidden_size):
        """QAnet output layer
           https://arxiv.org/pdf/1804.09541.pdf
        Args: 
            @param hidden_size: hidden dimension of QAnet model
        """
        super(output_layer, self).__init__() 

        ##################### Initialize layers #########################
        self.linear1 = nn.Linear(2*hidden_size, 1, bias = False)
        self.linear2 = nn.Linear(2*hidden_size, 1, bias = False)
        #nn.init.xavier_uniform_(self.linear1.weight)
        #nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, in1, in2, in3, mask): 
        """QAnet output layer
        Args: 
            @param in1: output of first encoder block of model encoder  (batch_size, context_length, hidden_size)
            @param in2: output of second encoder block of model encoder  (batch_size, context_length, hidden_size)
            @param in3: output of third encoder block of model encoder  (batch_size, context_length, hidden_size)
            @param mask: context mask 
        Outputs:
            @returns start: start prediction 
            @returns end: end prediction 
        """
        start = self.linear1(torch.cat((in1, in2 ), 2))
        end = self.linear2(torch.cat((in1, in3), 2))
        start = masked_softmax(start.squeeze(), mask, log_softmax= True)
        end = masked_softmax(end.squeeze(), mask, log_softmax= True)

        return start, end


class Embedding(nn.Module):
    def __init__(self, word_vectors, char_vocab_size, word_emb_size, char_emb_size, drop_prob_char, drop_prob_word):
        """QAnet embedding Layer
            https://arxiv.org/pdf/1804.09541.pdf
        Args:
            @param word_vectors : Pre-trained word vectors.
            @oaram char_vocab_size : Size of character vocabulary
            @param word_emb_size: Size of character embedding
            @param dropout_prob_char : dropout probability of character embedding 
            @param dropout_prob_word: dropout probability of word embedding
        """
        super(Embedding, self).__init__()

        ######################## PARAMETERS #############################
        self.drop_prob_char = drop_prob_char
        self.drop_prob_word = drop_prob_word

        ##################### Initialize layers #########################
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = ModelCharEmbeddings(char_vocab_size,word_emb_size,char_emb_size, drop_prob_char)
        self.hwy = HighwayEncoder(2, word_vectors.size(1) + word_emb_size)

    def forward(self, x, y):
        """
        @param x: Index of words (batch_size, seq_length)
        @param y: Index of characters (batch_size, seq_len, max_word_len)
        @out (batch_size, seq_len, glove_dim + word_emb_size)
        """
        emb_word = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb_char = self.char_embed(y) # (batch_size, seq_len, embed_size)
        emb = torch.cat([F.dropout(emb_word, self.drop_prob_word, self.training),
                         emb_char],dim=2)

        emb = self.hwy(emb)   # (batch_size, seq_len, glove_dim + word_emb_size)
        return emb


class PositionEncoder(nn.Module):
    def __init__(self, hidden_size, max_length = 600):
        """ Postion Encoder from: Attention is all you need
            https://arxiv.org/pdf/1706.03762.pdf
        Args: 
            @param hidden_size: hidden dimension of QAnet model
            @param max_length: maximum length of context 
        """
        super(PositionEncoder, self).__init__()
        #Parameters
        self.hidden_size = hidden_size
        self.max_length = max_length

        #Creating Signal to add
        pos = torch.arange(max_length).float()
        i = torch.arange(self.hidden_size//2)

        sin = torch.ones(self.max_length,self.hidden_size//2).transpose(0,1) * pos
        cos = torch.ones(self.max_length,self.hidden_size//2).transpose(0,1) * pos

        sin = torch.sin(sin.transpose(0,1) / (10000)**(2*i/self.hidden_size))
        cos = torch.cos(cos.transpose(0,1) / (10000)**(2*i/self.hidden_size))

        self.signal2 = torch.zeros((sin.shape[0], 2*sin.shape[1]))
        self.signal2[:,:-1:2] = sin
        self.signal2[:,1::2] = cos
        device, _ = get_available_devices()
        self.signal2 = self.signal2.to(device)

    def forward(self, x):
        return x + self.signal2[:x.shape[1],:]
        

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.
    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).
    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])
#        for gate,trans in zip(self.gates, self.transforms): 
            #nn.init.kaiming_normal_(gate.weight, nonlinearity= 'sigmoid')
            #nn.init.kaiming_normal_(trans.weight, nonlinearity= 'relu')

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class CNN(nn.Module):

    def __init__(self,embed_size,char_embed_size=50,ker=5,pad=1):
        """
        Constructor for the gate model
        @param embed_size : int for the size of the word  embeded size
        @param char_embded_size  : int for the size of the caracter  embeded size
        @param ker : int kernel_size used in Convolutions
        @param pad : int padding used in Convolutions
        @param stri : int  number of stride.
        """
        super(CNN, self).__init__() 

        ##################### Initialize layers #########################
        self.conv_layer=nn.Conv1d(in_channels=char_embed_size, out_channels=embed_size, kernel_size=ker, padding=pad)
        #nn.init.kaiming_normal_(self.conv_layer.weight, nonlinearity='relu')
        self.maxpool=nn.AdaptiveMaxPool1d(1)
        
    def forward(self,xreshaped):
        """
        forward function for computing the output
        @param xreshaped : torch tensor of size [BATCH_SIZE, EMBED_SIZE, max_word_lenght]. 
        @return xconvout : torch tensor after convolution and maxpooling [BATCH_SIZE, EMBED_SIZE].
        """
        xconv=self.conv_layer(xreshaped)
        xconvout=self.maxpool(F.relu(xconv)).squeeze()
        return xconvout


class ModelCharEmbeddings(nn.Module): 

    def __init__(self, char_vocab_size, word_embed_size, char_emb_size=50, prob=0.2):
        """QAnet embedding Layer
                https://arxiv.org/pdf/1804.09541.pdf
            Args:
                @param char_vocab_size : Size of character vocabulary
                @param word_emb_size: Size of character embedding
                @param char_emb_size: Size of character embeddings
                @param dropout_prob_char: dropout probability of character embedding 
        """
        super(ModelCharEmbeddings, self).__init__()

        ######################## PARAMETERS #############################
        self.char_vocab_size = char_vocab_size
        self.word_embed_size = word_embed_size
        self.char_emb_size=char_emb_size
        self.prob = prob
        pad_token_idx = 0 

        ##################### Initialize layers #########################
        self.model_embeddings=nn.Embedding(self.char_vocab_size,self.char_emb_size,pad_token_idx)
        self.convnet=CNN(self.word_embed_size,self.char_emb_size)
        

    def forward(self, input):

        batch_size, seq_len, word_len = input.shape
        x_emb = self.model_embeddings(input)
        x_emb = F.dropout(x_emb, self.prob, self.training)

        x_flat = x_emb.flatten(start_dim=0, end_dim = 1)
        x_conv_out = self.convnet(x_flat.permute((0,2,1)))
        return x_conv_out.view((-1,seq_len,self.word_embed_size))

        return output


class BiDAFAttention(nn.Module):
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
    def __init__(self, hidden_size):
        super(BiDAFAttention, self).__init__()
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        #for weight in (self.c_weight, self.q_weight, self.cq_weight):
            #nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

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

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s