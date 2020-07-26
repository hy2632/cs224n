# Assignment #5

(7.20)增加 Pylance 插件作为语言服务器. 打开 type checking mode(basic).

## 文字题

- (a) We learned in class that recurrent neural architectures can operate over variable length input (i.e., the shape of the model parameters is independent of the length of the input sentence). Is the same true of convolutional architectures? Write one sentence to explain why or why not.

  window t ∈ {1, . . . , mword − k + 1}，
  mword 即最长单词的长度可变，
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

# 1. Character-based convolutional encoder for NMT (36 points)

## Vocab.py

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

字母 ∏ Û python 执行有问题，改成`<pad>`, `<unk>`。

## (f) highway

要求写一个 sanity_check, (f)本身实现很简单，只是一步处理，所以检查一下前后维度就可以。

## (g) cnn.py, CNN

输入(sentence_length, batch_size, e_char, m_word)，前两维不动，对每个词 conv 完，后两维应该是 f 和窗口数，再经过 maxpool 所有窗口， 输出是(sentence_length, batch_size, f)
torch 需要使用.contiguous().view(),因为 view 只能作用在 contiguous 的变量上
比较关键的一步。
**07/25 更新**：
果然后面还是出问题了。m_word 是 forward 函数中参数 x_reshaped 的维度属性，如果使用 max_pool layer，一开始并不知道输入的参数 m_word 是多少。所以不应该用 maxpool 层（因为不能对一个多维 tensor 的某一维更新），
而应该在 forward 函数里直接调用 torch.max(dim=2)

## (h) Model_Embeddings.

一个问题是 f=e_word, e_word 和 e_char 的关系到底如何？？
题目假设 e_char=50, e_word 是初始化 model_embeddings 的参数 word_embedding_size, 默认值 21。

## (j)

wdnmd， vocab.py 里的 sents_var 总是空的，查了半天发现 utils.pad_sents 忘了粘贴。
nmt_model.py 中 step()函数

<code>if enc*masks is not None:</code><br>
<code> e_t.data.masked_fill*(enc_masks.bool(), -float('inf'))</code>

显示'Tensor' object has no attribute 'bool'
原因应该是 torch 版本较低
事实也确实如此，local*env.yml 显示 pytorch=1.0.0，a4 作业就没有限定版本，估计是助教忘了更新。
解决方案：改成<br>
e_t.data.masked_fill*(enc_masks==1, -float('inf'))

epoch 100, iter 500, cum. loss 0.30, cum. ppl 1.01 cum. examples 200<br>
validation: iter 500, dev. ppl 1.001988<br>
Corpus BLEU: 99.66941696422141

达到题设要求。

# 2. Character-based LSTM decoder for NMT (26 points)

## (b)

奇怪的点在于, char_decoder.py 中 train_forward 的 loss 计算，不 softmaxloss 才收敛。
题目要求仔细阅读 nn.CrossEntropyLoss，实际上 Pytorch 中 CrossEntropyLoss()函数将 softmax-log-NLLLoss 合并到一块。

`This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.`
loss 0.38, Corpus BLEU: 99.66941696422141
t

## (c)

这部分思路很清晰，用到了一些技巧，比如(tensor,tensor)的 elementwise 的提取，char 拼接成 word 等，详见代码

## (e)

在 VM 上训练。
注意 run.sh 可以进行修改，使得 train_local 也可使用 cuda，提高效率。
仍然遇到了环境问题。 “RuntimeError: Given input size: (256x1x12). Calculated output size: (256x1x0). Output size is too small”
于是只能在 VM 上配一个和本地相同的（过时的）环境。问题解决。
**CNN.py 中存在问题，很久之前埋下的坑！！！**：初始化时如果建立 maxpool 就需要提前知道 m_word 以确定 kernel_size。这个问题可以这样解决：避免 maxpool 层，在 forward 中使用 torch.max 函数，对某个维度进行 max。
对 cnn 和 sanity_check 都进行修改。由于默认使用了 sanitycheck 的值 m_word=21,实际上在写其他函数调用 CNN 类的时候没有定义 m_word 值，所以正好不需要改。
参考：Tessa Scott<https://github.com/tessascott039/a5/blob/master/cnn.py>

<https://github.com/pytorch/pytorch/issues/4166><https://stackoverflow.com/questions/56137869/is-it-possible-to-make-a-max-pooling-on-dynamic-length-sentences-without-padding>
探讨了 nn.MaxPool1d 能不能有一个动态的 kernel_size。

train 的结果：
epoch 29, iter 196300, avg. loss 81.60, avg. ppl 59.75 cum. examples 9600, speed 6086.86 words/sec, time elapsed 20580.30 sec

epoch 29, iter 196310, avg. loss 81.10, avg. ppl 50.14 cum. examples 9920, speed 6458.96 words/sec, time elapsed 20581.33 sec

epoch 29, iter 196320, avg. loss 78.58, avg. ppl 48.75 cum. examples 10240, speed 6548.57 words/sec, time elapsed 20582.32 sec

epoch 29, iter 196330, avg. loss 86.52, avg. ppl 61.24 cum. examples 10537, speed 6019.06 words/sec, time elapsed 20583.36 sec

test：Corpus BLEU: 36.395796664198

# 3. Analyzing NMT Systems (8 points)

## (a)

用 linux 的 grep 命令查找字符串
(base) hy2632_ubuntu20@DESKTOP-8LEIHPS:~/cs224n/a5_public\$ grep tradu vocab.json
"traduciendo": 17349,
"tradujera": 32719,
"traduccin": 4562,
"traduzco": 40154,
"traduzcan": 23440,
"traductores": 19447,
"traducir": 4565,
"traducciones": 12054,
"traductor": 11809,
"traducirse": 36917,
"traducen": 19640,
"tradujo": 25176,
"traducido": 8515,
"traducimos": 18251,
"traduce": 7821,
"traducidas": 20336,
"traduzca": 44710,
"traducirlo": 19205,
"traductora": 13071,
"traduje": 23103,
"traducirlas": 35543,
"traducida": 19350,

traduces, traduzcas not in.
如果是 word-based NMT，将 spanish 翻译为 english， 如果句子中出现 traduces 就会判断为`<unk>`， 无法翻译；但如果是 character-aware NMT，别的类似 traducir(to translate)的动词可能有类似的性质(加 s，从 I 变成 you)，同时又恰好出现在训练集中，那么模型在遇到 traduces 的 s 时就能翻译出 you translate。

## (b)

1. 回顾 Word2Vec。<https://projector.tensorflow.org/> 可以查询 k-nearest words。
   Markdown 表格生成<https://www.tablesgenerator.com/markdown_tables>

| word        | closest word |
| ----------- | ------------ |
| financial   | economics    |
| neuron      | nerve        |
| Francisco   | san          |
| naturally   | occurring    |
| expectation | norms        |

2. 也可以上传自己的 character-aware NMT model 的 embeddings 查找 nearest neighbors.

![Image](https://raw.githubusercontent.com/hy2632/cs224n/master/a5_public/CS224n-A5-emb2-files/embedding_proj.jpg)

| word        | closest word |
| ----------- | ------------ |
| financial   | vertical     |
| neuron      | Newton       |
| Francisco   | France       |
| naturally   | practically  |
| expectation | exception    |

3. 分析 Word2Vec 和 CharCNN 的区别并解释。

除了 naturally/practically 意思相近，CharCNN 的 embeddings 更多的还是按照字母组成（同时也包含一些 pos 和 ner）。Word2Vec 则更多地把握了词义的相似和关联性。原因就是模型本身。

word2vec: skip-grams & CBOW(contiguous bag of words)，给定上下文此预测缺失的中心词 c，概率分布 P(C|W)。因而相近的词都是有较大概率作为中心词被代替。

CharCNN: 对某个单词的各个字母进行 charembedding，然后经过 CNN/Highway 等操作最后生成 wordembedding，较大程度上依赖于 charembedding，如果两个词有相同字母的 subset 则 wordembedding 可能相近。

## (c)

(45)正确的例子：
| Category       |                                                                            |
|----------------|----------------------------------------------------------------------------|
| ES             | A medida que se derrite un tmpano, estoy respirando su atmsfera ancestral. |
| Ref            | As an eardrum melts, I am breathing in its ancient atmosphere.             |
| A4 translation | As a `<unk>` `<unk>` I'm breathing its atmosphere `<unk>`                  |
| A5 translation | As it melts a iceberg, I'm breathing its ancestral atmosphere.             |

tímpano，témpano 是同义词，但一个翻译为耳膜一个翻译为冰山。这里冰山显然更为合适。

(85)错误的例子：
| Category       |                                                   |
|----------------|---------------------------------------------------|
| ES             | Es el sndrome de insensibilidad a los andrgenos.  |
| Ref            | It is the syndrome of insensitivity to androgens. |
| A4 translation | It's called `<unk>` `<unk>`                       |
| A5 translation | It's the syndrome of insulin insulin.             |

对于连续的`<unk>`，CharCNN的表现并没有很好改善（重复出现的insulin）。