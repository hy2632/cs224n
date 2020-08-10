import torch
import torch.nn as nn

import torch.nn.functional as F

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