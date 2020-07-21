# Assignment #5

(7.20)增加 Pylance 插件作为语言服务器. 打开 type checking mode(basic).

## 文字题

- (a) We learned in class that recurrent neural architectures can operate over variable length input (i.e., the shape of the model parameters is independent of the length of the input sentence). Is the same true of convolutional architectures? Write one sentence to explain why or why not.

  window t ∈ {1, . . . , mword − k + 1}
  mword 即最长单词的长度可变
  xconv ∈ R^(eword×(mword−k+1))

- (b)...if we use the kernel size k = 5, what will be the size of the padding (i.e. the additional number of zeros on each side) we need for the 1-dimensional convolution, such that there exists at least one window for all possible values of mword in our dataset?

  极端情况 mword=1， 前后各 1 个 token，还需 padding=1.

- (c) In step 4, we introduce a Highway Network with `xhighway = xgate xproj + (1 − xgate) xconv out`. Since xgate is the result of the sigmoid function, it has the range (0, 1).Consider the two extreme cases. If xgate → 0, then xhighway → xconv out. When xgate → 1, then xhighway → xproj. This means the Highway layer is smoothly varying its behavior between that of normal linear layer (xproj) and that of a layer which simply passes its inputs (xconv out) through. Use one or two sentences to explain why this behavior is useful in character embeddings. Based on the definition of `xgate = σ(Wgatexconv out + bgate)`, do you think it is better to initialize bgate to be negative or positive? Explain your reason briefly.

## Vocab.py 阅读

1. 这种写法很巧妙

   <code>
   for i, c in enumerate(self.char_list):
   self.char2id[c] = len(self.char2id)
   </code>

2. 组合用法很有意思，类似 zip+enumerate

   <code>
   from collections import Counter
   from itertools import chain
   word_freq = Counter(chain(\*corpus))
   </code>

3. 用字典辅助排序

   <code>
   valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
   top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
   </code>

用到了`json.dump`，Vocab 也用此形式存储。

## char_decoder.py
