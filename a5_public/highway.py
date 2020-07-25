#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

import torch.nn.functional as F


class Highway(nn.Module):
    """ Implement the highway network as a nn.Module class.
        - forward() map from x_conv_out to x_highway
        - operate on batch
        About custom nn modules, look at <https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html#pytorch-custom-nn-modules>
    """

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    # Look at NMT in A4, nmt_model.py
    def __init__(self, embed_size):
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.projection = nn.Linear(in_features=self.embed_size,
                                    out_features=self.embed_size,
                                    bias=True)
        self.gate = nn.Linear(in_features=self.embed_size,
                              out_features=self.embed_size,
                              bias=True)
        

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """ Take a mini-batch of sentence of ConvNN

            @param x_conv_out (Tensor): Tensor with shape of (max_sentence_length, batch_size, embed_size)
            
            @return x_highway (Tensor): Tensor with shape of (max_sentence_length, batch_size, embed_size)
        """
        x_proj = F.relu(self.projection(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_highway = torch.mul(x_gate, x_proj) + torch.mul(
            torch.tensor(1) - x_gate, x_conv_out)
        return x_highway

    ### END YOUR CODE
