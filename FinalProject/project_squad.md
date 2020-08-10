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

## 08/09 BiDAF_Char，在原有 Baseline 上微调

耗时大约是 baseline 的两倍多。每个 epoch 约 23 分钟。RNNEncoder 的 input_size 变为两倍，可以理解。
效果似乎略佳，在 epoch=7 时似乎已经达到了 baseline 后半段的水准。
不知道是不是 hidden_size 翻倍的原因呢。。

### **layers.py**

1.  增加`Char_Embedding`，将原有的`Embedding`改为`Word_Embdding`.
2.  `Char_Embedding` 没有用 `pretrained embeddings`，用 `char_vocab_size` 和 `char_dim` 初始化。前者采用 `char_emb.json` 的值 1376，后者则在 train 和 test 时指定(第一次指定为 200)
3.  两个`Embedding`层分别在最后做一个 linear projection ，使得具有 (batch_size, seq_len, hidden_size) 的形式
4.  按照原文 `The concatenation of the character and word embedding vectors is passed to a two-layer Highway Network (Srivastava et al., 2015).` 对 `HighwayEncoder`进行修改， 其输入为 char_emb 和 word_emb ， 先 concatenate, 再 highway 后最终输出 (batch_size, seq_len, 2\*hidden_sAize)
5.  之后的 3 个层不修改，只在 models.py 中调整参数

### models.py

1. 增加参数: char_vocab_size, char_dim
2. 增加层: self.char_emb, self.word_emb
3. 修改 self.enc ，input_size 和 hwy 后 shape 一致
4. forward 增加 cc_idxs 和 qc_idxs

### train_BiDAF_Char.py

1. args.py 对应的 char_dim 参数是需要更改的
2. 增加和 cc_idxs/qc_idxs 有关的部分

## QANet

1. Core contribution:

   1. **Conv & self attention**: local structure & global interaction
   2. Speed up: 5x
   3. data augmentation technique: EN -> FR -> EN, enriched the training data by paraphrasing.

2. High level structure:
   1. Embedding
      1. xw: 不变，GloVe 的 wordemb
      2. xc：trainable, 每个字母都是 p2=200, 每个单词限定到 16 个字母，最终从 16 个字母中取最大，得到一个 200 维的向量 xc
   2. Embedding Encoder
   3. Context query attention layer
   4. Model encoder layer
   5. Output layer
