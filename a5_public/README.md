# Assignment #5

(7.20)增加 Pylance 插件作为语言服务器. 打开 type checking mode(basic).

## 文字题

- (a) We learned in class that recurrent neural architectures can operate over variable length input (i.e., the shape of the model parameters is independent of the length of the input sentence). Is the same true of convolutional architectures? Write one sentence to explain why or why not.

  window t ∈ {1, . . . , mword − k + 1}
  mword 即最长单词的长度可变
  xconv ∈ R^(eword×(mword−k+1))

- (b)...if we use the kernel size k = 5, what will be the size of the padding (i.e. the additional number of zeros on each side) we need for the 1-dimensional convolution, such that there exists at least one window for all possible values of mword in our dataset?

  极端情况 mword=1， 前后各 1 个 token，还需 padding=1.

- (c) In step 4, we introduce a Highway Network with `xhighway = xgate xproj + (1 − xgate) xconv out`. Since xgate is the result of the sigmoid function, it has the range (0, 1).Consider the two extreme cases. If xgate → 0, then xhighway → xconv out. When xgate → 1, then xhighway → xproj. This means the Highway layer is smoothly varying its behavior between that of normal linear layer (xproj) and that of a layer which simply passes its inputs (xconv out) through. Use one or two sentences to explain why this behavior is useful in character embeddings. Based on the definition of `xgate = σ(Wgatexconv out + bgate)`, do you think it is better to initialize bgate to be negative or positive? Explain your reason briefly.
  原因： 所谓的 highway， x_gate=0 可以直接用 x_convout 的值。

  希望默认 x_gate 较小方便 highway，所以 b 取负。

- (d) In Lecture 10, we briefly introduced Transformers, a non-recurrent sequence
  (or sequence-to-sequence) model with a sequence of attention-based transformer blocks. Describe 2 advantages of a Transformer encoder over the LSTM-with-attention encoder in our NMT model

  可以看一下 <<Attention is all you need>>：
  "Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence."
  每一步都是句子里的所有单词之间建立联系。
  主要用到三个矩阵 Key, Query, value, `Attention(Q,K,V) = softmax(QK.T/\sqrt(d_k))V`
  (包学包会，这些动图和代码让你一次读懂「自注意力」 - 机器之心的文章 - 知乎 https://zhuanlan.zhihu.com/p/96492170)


    attention-based transformers的好处（P6 的 Part 4， Why self-attention）：

    未采用RNN就可以避免梯度消失和梯度爆炸等问题,
    从sequential computation 到实现parallelized computation,
    更易学习到"long-range dependencies in the network",
    更加interpretable.

## Implementation 代码实现

### Vocab.py

1. 这种写法很巧妙·

   <code>for i, c in enumerate(self.char_list):</code><br>
   <code> self.char2id[c] = len(self.char2id)</code>

2. 组合用法，类似 zip+enumerate

   <code>from collections import Counter</code><br>
   <code>from itertools import chain</code><br>
   <code>word_freq = Counter(chain(\*corpus))</code>

3. 用字典辅助排序

   <code>valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]</code><br>
   <code>top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True[:size]</code><br>

用到了`json.dump`，Vocab 也用此形式存储。

## (e) Implement `to_input_tensor_char()` in `vocab.py`

    字母∏ Û python执行有问题，改成<pad>, <unk>。

## (f) 要求写一个 sanity_check， (f)本身实现的 highway 很简单，只是一步处理，所以检查一下前后维度就可以。

## (g) cnn.py, CNN

  <!-- 目前的想法是输入(batch_size, sentence_length, m_word, e_char)，前两维不动，对每个词conv完结果应该是e_char， 所以输出是(batch_size, sentence_length, e_char)。接着做，之后看情况修改。
  torch需要使用.contiguous().view(),因为view只能作用在contiguous的变量上 -->

## (h) Model_Embeddings.

一个问题是 f=e_word, e_word 和 e_char 的关系到底如何？？
题目假设 e_char=50, e_word 是初始化 model_embeddings 的参数 word_embedding_size, 默认值 21。

## (j)

wdnmd， vocab.py 里的 sents_var 总是空的，查了半天发现 utils.pad_sents 忘了粘贴。
nmt_model.py 中 step()函数

if enc_masks is not None:

            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

显示'Tensor' object has no attribute 'bool'
原因应该是 torch 版本较低
事实也确实如此，local*env.yml 显示 pytorch=1.0.0，a4 作业就没有限定版本，估计是助教忘了更新。
解决方案：改成 e_t.data.masked_fill*(enc_masks==1, -float('inf'))

epoch 100, iter 500, cum. loss 0.30, cum. ppl 1.01 cum. examples 200
validation: iter 500, dev. ppl 1.001988
Corpus BLEU: 99.66941696422141

达到题设要求。
