#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

import torch.nn.functional as F


# class CNN(nn.Module):
#     """ CNN, mapping x_reshaped to x_conv_out \n
#         @param filters (int): number of filters \n
#         @param kernel_size (int): default as k=5 \n
#         @param stride (int): default as stride=1 \n
#     """

#     # Remember to delete the above 'pass' after your implementation
#     ### YOUR CODE HERE for part 1g
#     def __init__(
#         self,
#         f: int,
#         e_char: int = 50,
#         m_word: int = 21, # According to sanity_check 1h.
#         k: int = 5,
#         padding: int = 1,
#     ):

#         super(CNN, self).__init__()
#         self.f = f
#         self.conv1d = nn.Conv1d(
#             in_channels=e_char,
#             out_channels=f,
#             kernel_size=k,
#             padding=padding,
#             bias=True,
#         )
#         # maxpool from e_char*(m_word - k + 1) to e_char
#         self.maxpool = nn.MaxPool1d(kernel_size=m_word - k + 1 +
#                                     2 * padding)

#     def forward(
#         self,
#         x_reshaped: torch.Tensor,
#     ) -> torch.Tensor:
#         """ Map from x_reshaped to x_conv_out\n
#             @param x_reshaped (Tensor): Tensor with shape of (sentence_length, batch_size, e_char, m_word) \n
#             @return x_conv_out (Tensor) : Tensor with shape of (sentence_length, batch_size,  e_word=f) \n
#         """
#         (sentence_length, batch_size, e_char, m_word) = tuple(x_reshaped.size())
#         x_conv = self.conv1d(x_reshaped.contiguous().view(sentence_length*batch_size, e_char, m_word))
#         x_conv_out = self.maxpool(F.relu(x_conv))
#         x_conv_out = x_conv_out.squeeze(-1)
#         x_conv_out = x_conv_out.contiguous().view(sentence_length, batch_size, -1)
#         return x_conv_out

#     ### END YOUR CODE



# ================================================================================================================
# https://github.com/tessascott039/a5/blob/master/cnn.py
# 果然CNN的初始化不应该包含m_word这一项。
# 解决方案：不包含初始化的maxpool层
# ================================================================================================================
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
            @param x_reshaped (Tensor): Tensor with shape of (sentence_length, batch_size, e_char, m_word) \n
            @return x_conv_out (Tensor) : Tensor with shape of (sentence_length, batch_size,  e_word=f) \n
        """
        (sentence_length, batch_size, e_char, m_word) = tuple(x_reshaped.size())
        x_conv = self.conv1d(x_reshaped.contiguous().view(sentence_length*batch_size, e_char, m_word))
        x_conv_out = torch.max(F.relu(x_conv), dim=2)[0].contiguous().view(sentence_length, batch_size, self.f)
        return x_conv_out

    ### END YOUR CODE
