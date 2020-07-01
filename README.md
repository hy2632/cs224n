# <font size = 32>**cs224n 20Winter Natural Language Processing with Deep Learning**</font>
 http://web.stanford.edu/class/cs224n/index.html#schedule

# Lecture部分
## 1. Intro to Word Vectors
Word2vec, GloVe， 作业一和Gensim model。
## 3. Python教程
1. broadcast: dimension has compatible values if either: samevalue / one is 1
2. (3,4,5,6) , (4,6,7) -> (3,4,5,7)
3. pdb， 创建breakpoint调试

## 4. NN (backprop)
复习了一遍cs231n的内容，
1. Loss函数 partial of L/L 是1，在这基础上backprop用链式法则乘回去。最终相当于Loss变化单位1，初始变量需要变动多少。
2. 对每一个地方计算localgradient然后相乘，
3. 开叉的地方backprop梯度叠加
4. 画个graph会更容易看
5. numerical gradient check的方法是公式f'(x) ≈ (f(x+h)-f(x-h)) / 2h。 作业题里的检查方法。
6. Xavier Initialization: Var(Wi) = 2/(n_in + n_out)；Kaiming Init...


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
- 最终用时12000+s.. 还是趁早torch8
