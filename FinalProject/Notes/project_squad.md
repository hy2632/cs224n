SQuAD leaderboard: <https://rajpurkar.github.io/SQuAD-explorer/>

1. Tensorboard
   在 Azure 上训练并在本地显示：
   1. 在 VM 终端运行`tensorboard --logdir save --port 5678`
   2. 打开本地终端，运行`ssh -N -f -L localhost:1234:localhost:5678 hy2632@13.90.229.131`
   3. 本地打开`localhost:1234`

2) 注意事项

   1. 缩小训练样本

3) Improvements:
   1. Non-PCE model type: Transformer
   2. Types of RNN: LSTM to GRU
   3.

# **08/09 BiDAF_Char，在原有 Baseline 上微调，增加 char_emb**

耗时大约是 baseline 的两倍多。每个 epoch 约 23 分钟。RNNEncoder 的 input_size 变为两倍，可以理解。
效果似乎略佳，在 epoch=7 时似乎已经达到了 baseline 后半段的水准。
不知道是不是 hidden_size 翻倍的原因呢?

> **根据论文增加 char_embedding，EM 和 F1 大约各自提高两个点**
>
> **根据 leaderboard, BiDAF-No-Answer (single model) EM:59.174 F1:62.093**

## **layers.py**

1.  增加`Char_Embedding`，将原有的`Embedding`改为`Word_Embdding`.
2.  `Char_Embedding` 没有用 `pretrained embeddings`，用 `char_vocab_size` 和 `char_dim` 初始化。前者采用 `char_emb.json` 的值 1376，后者则在 train 和 test 时指定(第一次指定为 200)
3.  两个`Embedding`层分别在最后做一个 linear projection ，使得具有 (batch_size, seq_len, hidden_size) 的形式
4.  按照原文 "The concatenation of the character and word embedding vectors is passed to a two-layer Highway Network (Srivastava et al., 2015).", 对 `HighwayEncoder`进行修改， 其输入为 `char_emb` 和 `word_emb` ， 先 concatenate, 再 highway 后最终输出 `(batch_size, seq_len, 2*hidden_size)`
5.  之后的 3 个层不修改，只在 models.py 中调整参数

## **models.py**

1. 增加参数: char_vocab_size, char_dim
2. 增加层: self.char_emb, self.word_emb
3. 修改 self.enc ，input_size 和 hwy 后 shape 一致
4. forward 增加 cc_idxs 和 qc_idxs

## **train_BiDAF_Char.py**

1. args.py 对应的 char_dim 参数是需要更改的
2. 增加和 cc_idxs/qc_idxs 有关的部分
3. 原论文的 char_dim=100, 训练中取了200，所以效果稍好
4. 结果：
   > **EM:60.07, F1:63.48**
   > **Leaderboard: EM:59.174 F1:62.093**

# **08/10 BiDAF_Char**

## **layers.py**

6. layers 中，如果 cw_idx/qw_idx = 0，对应位置就会 mask， 使用 masked_softmax 后对应位置是 0. 这是用来解决 padding 问题的。
   > 通过检查 word2idx 字典，0:'--NULL--', 1:'--OOV--'
   > 通过检查 char2idx， padding_idx 也应该为 0。 所以`layers.py>Char_Embedding>__init__>self.char_emb>padding_idx=0`。由于 char_idx 一般没有 OOV，所以对于 `BiDAF_Char-01` 的训练结果没有产生影响。
7. 维度分析: `BiDAFAttention > cw_masks`: (batch_size, seq_len(c_len), 1). 最后一维是flag (word_idx==0?True:False)
8. `BiDAFOutput > mask.sum(-1)`: lengths对于batch中每个句子不同。 `c_len = c_mask.sum(-1)`


# **8/10 QANet**

1. High level structure:
   1. Embedding
      1. xw: 不变，GloVe 的 wordemb
      2. xc：trainable, 每个字母都是 p2=200, 每个单词限定到 16 个字母，最终从 16 个字母中取最大，得到一个 200 维的向量 xc
      3. **08/10** Char_emb 不使用CNN， 而是每个单词直接取max
   2. Embedding Encoder
   3. Context query attention layer
   4. Model encoder layer
   5. Output layer

2. 相关代码实现
   1. (http://nlp.seas.harvard.edu/2018/04/03/attention.html)



# **8/11 QANet**

怎么说，调了一整天的bug 总算是能开始训练了，对不对还要另说。
lr = 0.001, l2_wd = 3e-7
tmux 滚轮： ctrl+b + [ / q 

几个疑问点：
1. char_emb 里还是得用CNN
2. word_emb + char_emb 后每个词是500维，一开始变为d_model=128需要在position_encoding前先cnn降维。
3. 