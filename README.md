# <fontsize = 28>**cs224n 20Winter Natural Language Processing with Deep Learning**</font>
## Intro to Word Vectors
Word2vec, GloVe， 作业一和Gensim model。




# 作业部分
## Reference
1. cs224n 2019 作业笔记 https://looperxx.github.io/CS224n-2019-Assignment/
2. https://github.com/saunter999/NLP_CS224Stanford_2019/tree/36e3799634e5fbb404772b8f3e28d535d22a29c0

## AS2
### word2vec.py （6.30）
- 简单说就是把作业前半部分的结果用代码实现，实际操作和公式略有区别，所有的uw/uo向量不转置，目测是因为第一次作业的词向量默认是行向量（上课好像提过）。
- negSamplingLossAndGradient()函数
    gradCenterVec = np.dot(sigmoid(y_hat_k), uk) + -sigmoid(-y_hat_o) * uo
 
 好像第二部分的表达式和推导结果不一样，之后再检查一下

- gradOutsideVecs一开始直接矩阵乘法再sum行不通，目测是因为negSampleWordIndices有重复值，同一个idx需要加几遍，所以还是用for循环遍历negSampleWordIndices， elementwise地修改。

### sgd.py
- 梯度下降，更新一下loss和x。glob和pickle的用法还需要了解一下。

### run.py
- Iter 40000次...这种交给colab就好了 🙃
