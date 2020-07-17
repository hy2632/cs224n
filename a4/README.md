# NMT Assignment

Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

作业分为两部分， 第一部分代码实现 NMT with RNN， 第二部分文字题分析 NMT

## 1. NMT with RNN

- Bidirectional LSTM Encoder & Unidirectional LSTM Decoder
- 勘误：
  - h 为(embedding size)
  - **(3), (4) 式中的下标 1 应改为 m**

* (a) utils.py

  - 要求每个 batch 里的句子有相同 length。utils.py 中实现 pad_sents(padding)

* (b) model_embeddings.py

  - 先去 vocab.py 里了解类的定义。
  - VocabEntry 类初始化参数 word2id(dict: words -> indices), id2word 返回 idx 对应的 word 值，from_corpus 从 corpus 生成一个 VocabEntry 实例，from **collections** import **Counter**，Counter 可以直接查找出字符串中字母出现次数
  - Vocab 类包含 src 和 tgt 语言，初始化参数式两种语言的 VocabEntry， @staticmethod 静态方法
  - VocabEntry.from_corpus 创建一个 vocab_entry 对象。Vocab.build 分别用 src_sents, tgt_sents 创建 src, tgt 两个 vocab_entry 并返回包含两者的 Vocab(src, tgt)
  - 运用 nn.Embedding 初始化词嵌。

* (c) nmt_model.py

  - 按照 pdf 中的维度对各层初始化

* (d) nmt_model.py 中 encode 方法实现

  - `self.encoder` 是一个双向 lstm
  - `encode` 方法传入两个参数：`source_padded, source_lengths`。前者是已经 pad 后(src_len, b)的 tensor，每一列是一个句子。后者是一个整数列表，表示每个句子实际多少词。
  - 需要返回两个值：enc_hiddens = hencs(所有 1<=i<=m(句长),每一句中所有词，同时对于整个 batch 所有句子), dec_init_state = (hdec0, cdec0)
  - lstm 要求输入满足规范形状，所以需要`pad_packed_sequence` 和`packed_pad_sequence`进行变形
  - 第一步用`self.model_embeddings`把 source_padded 转换为词嵌入

* (e) `decode`方法

  - `self.decode`r 是`nn.LSTMCell`，返回值 h、c，但这部分包装在 step 里面，本 decode 方法里从`self.step`取得返回值`dec_state, combined_output, e_t`
  - 还是先用`model_embeddings`将 target_padded 转换为 Y，一个目标词嵌入，(tgt_len, b, e)
  - 用`torch.split`方法， 将 Y 按第 0 维分成步长为 1 的步数，相当于逐词(t)操作。
  - (5)式表明了一个迭代过程，最后关心的`combined_outputs`是 o_t 集合

* (f) `step`方法

  - step 方法具体处理(5)到(12)式。
  - 第一部分，(5)-(7)，运用 bmm、(un)squeeze。bmm 需要注意第 0 维度是留给 batch_size 的，两个三维 tensor 的第一二维相乘，满足维度要求。常见的是在 dim=1/2 做 unsqueeze，乘完再 squeeze
  - 注意到调换乘法次序+不同的变换维度方式会造成最终结果的精度损失。

* (g) 文字题：`generate_sent_masks()` 生成 `enc_masks(b, src_len)`标识 batch 中每个 sentence 每个词是否是 pad，这样做对 attention 计算的影响以及其必要性。

  - `step`中，(8)式 α_t 进行了 softmax，后续 a_t 计算为确保 attention 不受 padding 影响要求 padding 处 α_t=0，即 e_t 设置为-∞。

* (i)

  - git 配置：git remote add origin https://github.com/hy2632/cs224n.git
  - git push origin master
  - ..

  - Corpus BLEU: 31.892219171042335

* (j)
  | Attention Type | Advantage | Disadvantage |
  | ---- | ---- | ---- |
  | Dot Product | 不需要`self.att_projection`层 | 需要满足维度一致 |
  | Multiplicative | - | - |
  | Additive | tanh 操作 normalize 了数值 | 两个参数矩阵，参数更多，空间复杂度大 |

## 2. Analyzing NMT Systems (30 points)

- 参考 lec8 slides P50-
- https://www.skynettoday.com/editorials/state_of_nmt
- Out-of-vocabulary words, Domain mismatch between src&tgt, maintaining context over longer text, Low-resource language pairs.

* (a)

  - i.

    - 1. error: favorite of my favorites,
    - 2. reason: tgt 库中缺乏 one of my favorites 这样的表达, "Low-resource language pairs"
    - 3. to fix: 添加训练数据

  - ii.

    - 1. error: most read 译为了 more reading。
    - 2. reason: 使用 google translator 发现 ms ledo 被译为 read more， 而 ms ledo en los EEUU 被译为 most read in the US。西班牙语的特点？特定的语言构造。
    - 3. to fix: 需要让 ms ledo 和后面的定语建立更强的联系，从而把握语义理解。增大 hidden_size

  - iii.

    - 1. error: "<unk>"
    - 2. reason: Out-of-vocabulary
    - 3. to fix: 添加到词表

  - iv.

    - 1. error: block -> apple
    - 2. reason: "manzana" 多义性
    - 3. to fix: 训练集添加 manzana 作为 block 含义的 phrase 数据，且大于“vuelta a la manzana”因为 google translator 仍将该句错译。

  - v.

    - 1. error: "la sala de profesores": "teacher's lounge" -> "women's room",
    - 2. reason: "profesores"应该是复数，不包含性别，该句既错译又包含性别 bias。
    - 3. to fix: 增加“profesores”/profesor/profesora 的训练数据，平衡性别 bias 的同时也要将 teacher 翻译出来。

  - vi.
    - 1. error: hectare -> acre
    - 2. reason: 常识错误，涉及到单位转换
    - 3. to fix: 没想到好的方法。文章中写：General knowledge about the world is necessary for NMT systems to translate effectively. **However, this knowledge is difficult to encode in its entirety and is not easily extractable from volumes of data. We need mechanisms to incorporate common sense and world knowledge into our neural networks.**

* (b)

  - 88:

    - When he 's born , the baby looks a little bit .
    - When the child is born, she looks like a girl.
    - Cuando nace, el beb tiene aspecto de nia.(Cuando nace, el bebé tiene aspecto de niña.)
    - 错误：语意，原因：src 本身存在错误无法显示符号，解决方案：样本数据和测试数据的编码格式改一哈

  - 109：
    - So , there are many of a lot of sex .
    - So sex can come in lots of different varieties.
    - Entonces, hay muchas variedades de sexo.
    - 错误：语法（many a lot of)，variedades 没有翻译出 variety 的意思。解决方案：增加 muchas variedades 的数据

* (c)
  BLEU 的定义见 <BLEU: a Method for Automatic Evaluation of Machine Translation>
  Candidate c, Reference r, BLEU 包含两部分：reference 中出现 candidate 中 ngram phrase 的概率（注意有个 ceiling）和 candidate 太长导致的 brevity penalty。

- i.
  - for c1,
    - p1 = (0 + 1 + 1 + 1 + 0)/5 = 3/5
    - p2 = (0 + 1 + 1 + 0) /4 = 1/2
    - len(c) = 5
    - len(r) = 6
    - BP = exp(1-6/5) = 0.819
    - `BLEU = 0.819 * exp(0.5*log(3/5) + 0.5*log(1/2)) = 0.449`
  - for c2,
    - p1 = (1 + 1 + 0 + 1 + 1)/5 = 4/5
    - p2 = (1 + 0 + 0 + 1) /4 = 1/2
    - len(c) = 5
    - len(r) = 4
    - BP = 1
    - `BLEU = 1 * exp(0.5*log(4/5) + 0.5*log(1/2)) = 0.632`
  - c2 更好。

- ii.
  - c1: p1 = 3/5, p2 = 1/2, BLEU 不变 0.449
  - c2: p1 = 2/5, p2 = 1/4, len(c) = 5, len(r) = 6, `BLEU = 0.819 * exp(0.5*log(2/5) + 0.5*log(1/4)) = 0.259`
  - 当前 c1 更好。

- iii.
  - 单一 ref 产生类似 ii 的问题， 比如对于 r2，c2 可以说是非常好的翻译，如果没有 r2 仅用 r1 判断，c2 就比 c1 差很多。

- iv.

| BLEU vs Human | 1                    | 2                        |
| ------------- | -------------------- | ------------------------ |
| Pro           | Fast                 | Language independent     |
| Con           | Lack of Common Sense | Need Multiple References |
