"""Top-level model classes.
Author:
    Axel Moyal (axmoyal@stanford.edu)
    Guillermo Bescos (gbescos@stanford.edu)
    Lucas Soffer (lsoffer@stanford.edu)    
"""

import QANet_layers_temp
import torch
import torch.nn as nn
import torch.nn.functional as F


class QANet(nn.Module):
    """Baseline BiDAF model for SQuAD.
    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).
    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, hidden_size, char_vocab_size , char_emb_size, word_char_emb_size,  drop_prob,
                num_blocks_embd , num_conv_embd , kernel_size , num_heads , num_blocks_model , num_conv_model,
                dropout_char, dropout_word, survival_prob):
        super(QANet, self).__init__()

        ################################ HYPER PARAMETERS ############################################
        self.num_blocks_embd = num_blocks_embd #(default: 1) Number of blocks in embedding encoder
        self.num_conv_embd = num_conv_embd # (default: 4) Number of convolutional layers in a block of Embedding encoder
        self.kernel_size = kernel_size # (default: 7) Kernel size of convolutional layers in block
        self.hidden_size = hidden_size #(default: 128) Number of filters of conv layers, this dictates the hidden size 
        self.num_heads = num_heads #(default: 8) Number of heads for multihead attention, !!!!! NOTE: hidden_size must be a multiple of num_heads
        self.drop_prob = drop_prob #(default: More complicated on paper) dropout rate
        self.num_blocks_model = num_blocks_model # (default: 7) number of blocks in model encoder
        self.num_conv_model = num_conv_model # (default: 2) number of convolutional layers in blocks of model encoder
        self.embd_size = 300 + word_char_emb_size  # (default: 500) Embedding size char + word
        self.dropout_char = dropout_char #(default: 0.05) Dropout rate for character embedding
        self.dropout_word = dropout_word #(default: 0.1) Dropout rate for word embedding
        self.survival_prob = survival_prob

        ############################### Initialize layers ###########################################
        self.emb = QANet_layers_temp.Embedding(word_vectors=word_vectors,
                                    char_vocab_size = char_vocab_size,
                                    word_emb_size = word_char_emb_size,
                                    char_emb_size = char_emb_size,
                                    drop_prob_char= self.dropout_char,
                                    drop_prob_word= self.dropout_word )

        self.embedding_projection = nn.Linear(self.embd_size, self.hidden_size)
        #nn.init.kaiming_normal_(self.embedding_projection.weight, nonlinearity = 'relu')

        self.emb_enc = QANet_layers_temp.Embedding_Encoder( self.num_blocks_embd,
                                                 self.num_conv_embd,
                                                 self.kernel_size,
                                                 self.hidden_size,
                                                 self.num_heads,
                                                 self.survival_prob)
                                
        self.att = QANet_layers_temp.BiDAFAttention(hidden_size= self.hidden_size)

        self.attention_projection = nn.Linear(4 * self.hidden_size, self.hidden_size)
        #nn.init.kaiming_normal_(self.attention_projection.weight, nonlinearity = 'relu')

        self.mod = QANet_layers_temp.Model_Encoder(self.num_blocks_model, 
                                        self.num_conv_model, 
                                        self.kernel_size, 
                                        self.hidden_size, 
                                        self.num_heads, 
                                        self.survival_prob)

        self.out = QANet_layers_temp.output_layer(self.hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        #Embedding Layer
        c_emb = self.emb(cw_idxs, cc_idxs)  #(batch_size, c_len, embed_size)
        q_emb = self.emb(qw_idxs, qc_idxs)  #(batch_size, q_len, embed_size)

        # Dropout 
        c_emb = F.dropout(c_emb, self.drop_prob, self.training)
        q_emb = F.dropout(q_emb, self.drop_prob, self.training) 

        #Project Down 
        c_emb = F.relu(self.embedding_projection(c_emb))
        q_emb = F.relu(self.embedding_projection(q_emb))

        #Embedding Encoder
        c_enc, q_enc = self.emb_enc(c_emb, q_emb, c_mask, q_mask) #(batch_size, q/c_len, hidden size)

        #Dropout 
        c_enc = F.dropout(c_enc, self.drop_prob, self.training)
        q_enc = F.dropout(q_enc, self.drop_prob, self.training)
        
        #Bidirectional Attetnion
        att = self.att(c_enc, q_enc, c_mask, q_mask) #(batch_size, c_len, 4*hidden size)

        #Dropout 
        att = F.dropout(att, self.drop_prob, self.training)
        
        #Project Down
        att = F.relu(self.attention_projection(att))

        #Model Encoder
        mod1, mod2, mod3 = self.mod(att, c_mask)        # (batch_size, c_len,  hidden_size)
        
        #Dropout 
        mod1 = F.dropout(mod1, self.drop_prob, self.training)
        mod2 = F.dropout(mod2, self.drop_prob, self.training)
        mod3 = F.dropout(mod3, self.drop_prob, self.training)

        #Output Layer
        out = self.out(mod1, mod2, mod3, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out