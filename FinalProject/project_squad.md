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



## util.py 注释

1. **class SQuAD**: 处理data.dataset，将字典转化到类变量
2. **def collate_fn(examples) -> (lists)**
   1. 将examples: list(tuple(5)) 分为tuple(5 lists)的技巧 zip(*examples)
      ```python
      example1 = (1, "str", [3])
      example2 = (2, "str2", [100])
      examples = (example1, example2)
      ints, strs, lists = zip(*examples)
      ints, strs, lists
      -----------------
      ((1, 2), ('str', 'str2'), ([3], [100]))
      ```
   2. 输入是一堆example(tuple)的list，输出是5个list的tuple。
      ```python
      (context_idxs, context_char_idxs,
            question_idxs, question_char_idxs,
            y1s, y2s, ids)
      ```
3. **class AverageMeter**
   平均值记录和更新，包含avg, sum, count

4. **class EMA(model, decay)**
   ```python
   self.decay = decay
   self.shadow = {}  
   self.original = {}
   ```

   1. shadow存放需要梯度的参数
   2. model.named_parameters(): 可转换为dict: {name: value}
   3. assign 负责更新（也备份到self.original），resume 负责用 self.original 进行恢复

5. **class CheckpointSaver**
   1. 
   







## 08/08 为BiDAF 增加 char-level
===
   5.2.1 Character-level Embeddings
   the util.SQuAD class returns character indices, and these are
   loaded in train.py and test.py
===











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



