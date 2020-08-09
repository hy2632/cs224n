SQuAD leaderboard: <https://rajpurkar.github.io/SQuAD-explorer/>




1. Tensorboard
    在Azure上训练并在本地显示：
    1. 在VM终端运行`tensorboard --logdir save --port 5678`
    2. 打开本地终端，运行`ssh -N -f -L localhost:1234:localhost:5678 hy2632@13.90.229.131`
    3. 本地打开`localhost:1234`


2. 注意事项
   1. 缩小训练样本

3. Improvements:
   1. Non-PCE model type: Transformer
   2. Types of RNN: LSTM to GRU
   3. 











## 08/08 为BiDAF 增加 char-level
   5.2.1 Character-level Embeddings
   the util.SQuAD class returns character indices, and these are
   loaded in train.py and test.py











## QANet

1. Core contribution:
   1. **Conv & self attention**: local structure & global interaction
   2. Speed up: 5x
   3. data augmentation technique: EN -> FR -> EN, enriched the training data by paraphrasing.

2. High level structure:
   1. Embedding
      1. xw: 不变，GloVe的wordemb
      2. xc：trainable, 每个字母都是p2=200, 每个单词限定到16个字母，最终从16个字母中取最大，得到一个200维的向量xc
   2. Embedding Encoder
   3. Context query attention layer
   4. Model encoder layer
   5. Output layer



