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

## 08/14

训练参数：

    maximum_context_length: 400 (c_len 多于该值的跳过)
    num_mod_blocks: 3 (原文是 7)
    batch_size: 28
    num_heads: 4 (原文是 8)

改动：

    char_emb 使用 CNN
    multihead attention 使用 nn 自带的函数
    Output layer 简化，去掉一次 softmax 和 relu
    使用 Adam 训练
