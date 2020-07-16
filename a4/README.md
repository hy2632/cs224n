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
  - `self.encoder` 是一个双向 lstm
  - `encode` 方法传入两个参数：`source_padded, source_lengths`。前者是已经pad后(src_len, b)的tensor，每一列是一个句子。后者是一个整数列表，表示每个句子实际多少词。
  - 需要返回两个值：enc_hiddens = hencs(所有1<=i<=m(句长),每一句中所有词，同时对于整个batch所有句子), dec_init_state = (hdec0, cdec0)
  - lstm要求输入满足规范形状，所以需要`pad_packed_sequence` 和`packed_pad_sequence`进行变形
  - 第一步用`self.model_embeddings`把source_padded转换为词嵌入

* (e) `decode`方法
  - `self.decode`r是`nn.LSTMCell`，返回值h、c，但这部分包装在step里面，本decode方法里从`self.step`取得返回值`dec_state, combined_output, e_t`
  - 还是先用`model_embeddings`将target_padded转换为Y，一个目标词嵌入，(tgt_len, b, e)
  - 用`torch.split`方法， 将Y按第0维分成步长为1的步数，相当于逐词(t)操作。
  - (5)式表明了一个迭代过程，最后关心的`combined_outputs`是o_t集合

* (f) `step`方法
  - step方法具体处理(5)到(12)式。
  - 第一部分，(5)-(7)，运用bmm、(un)squeeze。bmm需要注意第0维度是留给batch_size的，两个三维tensor的第一二维相乘，满足维度要求。常见的是在dim=1/2做unsqueeze，乘完再squeeze
  - 注意到调换乘法次序+不同的变换维度方式会造成最终结果的精度损失。

* (g) 文字题：`generate_sent_masks()` 生成 `enc_masks(b, src_len)`标识batch中每个sentence每个词是否是pad，这样做对attention计算的影响以及其必要性。
  - `step`中，(8)式α_t进行了softmax，后续a_t计算为确保attention不受padding影响要求padding处α_t=0，即e_t设置为-∞。
  - 
