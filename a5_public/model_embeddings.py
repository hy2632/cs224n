#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"


class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    Map a batch of sentences of x_padded to x_word_emb
    """
    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.vocab = vocab
        self.word_embed_size = word_embed_size
        pad_token_idx = vocab.char2id['<pad>']
        # nn.Embedding : input (*), output(*,H), elementwise and only add one dimension
        self.char_embedding = nn.Embedding(num_embeddings=len(self.vocab.char2id), embedding_dim=50, padding_idx=pad_token_idx)
        self.cnn = CNN(f=self.word_embed_size,e_char=50)
        self.highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(0.3)
        ### END YOUR CODE

    def forward(self, x_padded):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param x_padded: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param x_word_emb: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        # x_emb with shape of (sentence_length, batch_size, max_word_length, e_char)
        x_emb = self.char_embedding(x_padded)

        # CNN:
        # reshape x_emb to obtain x_reshaped(,,e_char,m_word)
        x_reshaped = x_emb.permute(0, 1, 3, 2)
        x_conv_out = self.cnn(x_reshaped)
        x_highway = self.highway(x_conv_out)
        x_word_emb = self.dropout(x_highway)
        return x_word_emb
        ### END YOUR CODE
