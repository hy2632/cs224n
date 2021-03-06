搭建的 QANet 模型存在 out-of-memory / loss 不收敛等问题
与 (https://github.com/axmoyal/CS224N-project-QAnet-) 进行对比

## 问题汇总：

1. Embedding_projection : context 和 question 的每个词 300+200=500 -> dim=128, 之后再进入 Embedding_Encoder
2. Embedding_Encoder 要实现 Context 和 Question 共享，

```python
c_enc, q_enc = self.emb_enc(c_emb, q_emb, c_mask, q_mask)
```

3. C2QAttention 后， 同样将 4\*d_model -> d_model
4. Model_Encoder 整体应包含 3\*7 次重复，直接输出 M0, M1, M2
5. Embedding_Encoder_Blocks 和 Model_Encoder_Blocks 共用底层 Encoder_Block
6. PositionalEncoding 他取的 maximum_context_length 是固定值 600，且没有作为 Encoder_Block 模块的输入参数
7. 暂时不去实现 `StochasticDepthDropout`

## 08/13 QANet-01

训练参数：

    maximum_context_length: 400 (c_len 多于该值的跳过)
    num_mod_blocks: 3 (原文是 7)
    batch_size: 28
    num_heads: 4 (原文是 8)

改动：

**char_emb 使用 CNN**
    multihead attention 使用 nn 自带的函数
    Output layer 简化，去掉一次 softmax 和 relu
    使用 Adam 训练

## 08/14 QANet-02

改动：

    char embedding去除CNN，只取torch.max
    
```python
    e_char = self.char_emb(c_idxs)
    e_char = F.dropout(e_char, self.drop_prob_char, self.training)

    (batch_size, seq_len, m_word, char_dim) = tuple(e_char.size())
    e_char = e_char.contiguous().view(seq_len*batch_size, char_dim, m_word)
    e_char = torch.max(F.relu(e_char), dim=2)[0].contiguous().view(batch_size, seq_len, char_dim)
```
    num_mod_blocks: 5
    num_heads: 8

问题：
    
    后期loss增加，可能是学习率过大
    提前终止。

## pytorch中分析时间开销的方法
    (https://zhuanlan.zhihu.com/p/41662629)



## 08/15 QANet-03

改动:

**char_embedding 由于softmax(relu()),将权重初始化到0-1的uniform** : 失败，结果不再变化。
学习率：1e-3 -> 1e-4
num_mod_blocks: 4
num_heads: 8
args.batch_size: 28 -> 32


结果:

    train_loss不断下降，val_loss却在epoch10后上升

原因：(https://www.zhihu.com/question/396221084/answer/1236081752)

    数据有坏点
    起始学习率
    模型逻辑可能不对
    全连接更易过拟合

结论：
    overfitting
    可以选用较短的epoch和较小的batch_size 28
    模型的最高值仍然低于baseline



## 08/15 QANet-04
    <!-- 加dropout -->
    batch_size 28
    Char_emb 采用 CNN，同时mclen从400改到500
    embedding 到 emb_enc 的 linear projection 改成 CNN 并直接加到Emb最后

    效果差于QANet - 03

## 8/15 QANet-05
    3个mod_enc_block, 4个multihead, lr=5e-4
    效果和04、02类似

    只在Embedding最后加CNN没有做dropout

    解决方案：去除一些bias，进一步简化，提前终止。

## 8/15 QANet-06
maximum_context_length:400
去除bias: Output
**cnn_proj 改成 kernel_size=1**
改用adadelta, lr=0.5

<https://web.stanford.edu/class/cs224n/reports/default/>
看了一些论文，QANet可能确实还比不上BiDAF_Char，Non contextual embedding model的天花板就这样了，所以还是得加contextual embeddings。下一步做Bert。

