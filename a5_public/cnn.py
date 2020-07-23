#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

import torch.nn.functional as F


class CNN(nn.Module):
    """ CNN, mapping x_reshaped to x_conv_out \n
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
        m_word: int = 12,
        k: int = 5,
        padding: int = 1,
    ):
        super(CNN, self).__init__()
        self.e_char = e_char
        self.m_word = m_word
        self.k = k
        self.padding = padding
        self.f = f
        self.conv1d = nn.Conv1d(
            in_channels=self.e_char,
            out_channels=self.f,
            kernel_size=self.k,
            padding=self.padding,
            bias=True,
        )
        # maxpool from e_char*(m_word - k + 1) to e_char
        self.maxpool = nn.MaxPool1d(kernel_size=self.m_word - self.k + 1 +
                                    2 * self.padding)

    def forward(
        self,
        x_reshaped: torch.Tensor,
    ) -> torch.Tensor:
        """ Map from x_reshaped to x_conv_out\n
            @param x_reshaped (Tensor): Tensor with shape of (sentence_length, batch_size, m_word, e_char) \n
            @return x_conv_out (Tensor) : Tensor with shape of (sentence_length, batch_size,  e_char) \n
        """

        sentence_length = x_reshaped.size()[0]
        batch_size = x_reshaped.size()[1]
        x_conv = self.conv1d(x_reshaped.contiguous().view(
            sentence_length * batch_size, self.e_char, self.m_word))

        x_conv_out = self.maxpool(F.relu(x_conv)).squeeze(-1)
        x_conv_out = x_conv_out.contiguous().view(sentence_length, batch_size,
                                                  -1)
        return x_conv_out

    ### END YOUR CODE
