#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2019-20: Homework 5
"""

# from sanity_check import HIDDEN_SIZE
# Circular dependencies!!!!
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(
            hidden_size, len(self.target_vocab.char2id
                             ))  # vocab.py 中 VocabEntry. 字典长度，得出每个字符的得分
        self.decoderCharEmb = nn.Embedding(
            len(self.target_vocab.char2id),
            char_embedding_size,
            padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.

        # firstly convert ints to charembeddings
        char_embeddings = self.decoderCharEmb(input)
        # >>> output, (hn, cn) = rnn(input, (h0, c0))
        hidden_states, dec_hidden = self.charDecoder(char_embeddings,
                                                     dec_hidden)
        scores = self.char_output_projection(hidden_states)
        return scores, dec_hidden
        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass. (compute loss_char_dec summed across the whole batch)
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        # ===============================================================================
        # ?????这里不计算softmax，loss才能收敛。
        # 原因： 题目要求仔细阅读nn.CrossEntropyLoss，Pytorch中CrossEntropyLoss()函数的主要是将softmax-log-NLLLoss合并到一块得到的结果。
        # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
        # Pytorch常用的交叉熵损失函数CrossEntropyLoss()详解 - ShuYini的文章 - 知乎 https://zhuanlan.zhihu.com/p/98785902
        # ===============================================================================
        scores, dec_hidden = self.forward(char_sequence[:-1], dec_hidden)
        loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char_pad,
                                   reduction='sum')
        # target: (length, batch_size), probs: (length, batch_size, self.vocab_size)
        # nn.CrossEntropyLoss requires the classification dimension to be the 2nd one:
        # input:(N,C,d...), target:(N, d...) -> output:(N, d...)
        output = loss(input=scores.permute(0, 2, 1), target=char_sequence[1:])
        return output
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        _, batch_size, hidden_size = tuple(initialStates[0].size())
        decodedWords = []
        for sentence_index in range(batch_size):
            initialState_sent = (initialStates[0][:,sentence_index, :].unsqueeze(dim=1),
                                 initialStates[1][:,sentence_index, :].unsqueeze(dim=1))
            # 提取出每个句子的两个初始参数有些复杂，因为initialStates其实是一个二元tuple，每个元素各是一个(1, batch_size, hidden_size)的tensor，
            # 需要做的是按照每个tensor的第2维（即batch_size维）提取出每个句子的初始状态元组
            # 有尝试用zip + torch.split的方法，好像不好用，还是老老实实地用索引提取。
            output_word = []
            current_char = '{'
            dec_hidden = initialState_sent

            for t in range(max_length):
                # 把current_char转换为一个tensor，首先先对应到int的id，然后强行unsqueeze两遍满足self.forward的参数input的形式
                input_current_char = torch.tensor(self.target_vocab.char2id[current_char],device=device).unsqueeze(dim=0).unsqueeze(dim=0)
                scores, dec_hidden = self.forward(input_current_char,dec_hidden)
                probs = F.softmax(scores, dim=2)
                current_char = self.target_vocab.id2char[int(torch.argmax(probs, dim=2).squeeze())]
                if current_char == '}': break
                output_word.append(current_char)
                
            decodedWords += [''.join(output_word)] # 拼接字符串的技巧
        return decodedWords
        ## END YOUR CODE
