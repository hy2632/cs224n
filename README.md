# <font size = 36>**cs224n 20Winter 
 
 Natural Language Processing with Deep Learning**</font>
 http://web.stanford.edu/class/cs224n/index.html#schedule

Markdown教程： https://guides.github.com/features/mastering-markdown/

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
6. Xavier Initialization: Var(Wi) = 2/(n_in + n_out)；Kaiming Init... 避免symmetry影响learning/specialization.

## 5. Dependence Parsing 依存解析
1. 两种思路，Constituency Parsing(phrase structure grammar/context-free grammars简称CFG)无上下文语法 和 DP。前一种语法分析，看词性等等，后一种直接看单词之间自身（而非词性）的依赖关系。
2. 复杂句子结构的可能结构Catalan numbers：

 >>>![公式](https://wikimedia.org/api/rest_v1/media/math/render/svg/57de4926a69e67cdcdf999030c5ec3c25d97b0c9)
 
 >>>![公式](https://wikimedia.org/api/rest_v1/media/math/render/svg/a9434815d6487cd3786fd39f533175c6ad99c7c6)
3. 各种歧义，比如AdjectivalModifierAmbiguity, VP依存歧义...
4. Treebank: 
5. transition-based dependency parser
6. Dynamic Programming: O(n)



# 作业部分
## Reference
1. cs224n 2019 作业笔记 https://looperxx.github.io/CS224n-2019-Assignment/
2. https://github.com/saunter999/NLP_CS224Stanford_2019/tree/36e3799634e5fbb404772b8f3e28d535d22a29c0

## A2
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


## A3
### Pytorch
- 还是conda env create -f local_env.yaml
- 装的torch又是“无法定位序数242于动态链接库C:\Anaconda3\envs\a3\lib\site-packages\torch\lib\torch_cpu.dll上”， 爪巴(ノ｀Д)ノ
### WSL
- 转到ubuntu，然后conda - env - 配好torch
- yapf代码格式化， shift+alt+F

### 1. (a) Adam

i.	Adam和SGD的区别问题。m指momentum，用动量而非梯度更新，使梯度变化更圆滑，防止梯度爆炸和梯度消失，更新稳定。

ii.	v:梯度平方的移动平均值。先前梯度较小的参数更新将会放大，梯度较大的参数更新将会缩小。 帮助走出局部最优（鞍点）和使更新稳定

### 1. (b) Dropout

i.  
> $ E(h_drop_i) = E(\gamma d \dot h_i) = h_i $

> $ E(\gamma d) = 1 $

> $ \gamma * (1-P_drop) *1 + \gamma * P_drop * 0 = 1 $

> $ \gamma = 1/(1-P_drop) $

ii.  Dropout

防止过拟合（feature之间的相关性）

训练时增加随机性；评估时不希望有随机性

### 2.(b)
2n次，每个单词shift和dependency

### result

Epoch 10 out of 10
Average Train Loss: 0.028997931656633297
- dev UAS: 87.93

TESTING
Restoring the best model weights found on the dev set
Final evaluation on test set
- test UAS: 88.69

### (f)
4种 parsing error: 
- PP attachment error 介词短语连接错误
- VP attachment error 动词短语连接错误
- Modifier attachment error 修饰语连接错误（副词形容词之类）
- Coordination attachment error （协调连接错误）

1. 
- Error type: VP attachment error
- Incorrect dependency: wedding → fearing
- Correct dependency: heading → fearing

2. 
- Error type: Coordination Attachment Error
- Incorrect dependency: makes → rescue
- Correct dependency: rush → rescue

3. 
- Error type:  PP attachment Error
- Incorrect dependency:  named → midland 
- Correct dependency: guy → midland 

4. 
- Error type: Modifier Attachment Error
- Incorrect dependency: elements → most 
- Correct dependency: crucial → most 






