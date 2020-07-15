# NMT Assignment

Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

作业分为两部分， 第一部分代码实现 NMT with RNN， 第二部分文字题分析 NMT

## 1. NMM with RNN

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
  - self.encoder 是一个双向 lstm
  - encode 方法传入两个参数：source_padded和source_lengths。前者是已经pad后(src_len, b)的tensor，每一列是一个句子。后者是一个整数列表，表示每个句子实际多少词。
  - 需要返回两个值：enc_hiddens = hencs(所有1<=i<=m(句长),每一句中所有词，同时对于整个batch所有句子), dec_init_state = (hdec0, cdec0)
  - lstm要求输入满足规范形状，所以需要pad_packed_sequence 和packed_pad_sequence进行变形
  - 第一步用self.model_embeddings把source_padded转换为词嵌入

* (e) decode方法
  - self.
