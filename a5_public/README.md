# Assignment #5

(7.20)增加 Pylance 插件作为语言服务器. 打开 type checking mode(basic).


这种写法很巧妙<code> 
for i, c in enumerate(self.char_list): 
    self.char2id[c] = len(self.char2id)
</code>



组合用法很有意思，类似 zip+enumerate<code> 
from collections import Counter <br>
from itertools import chain <br>
word_freq = Counter(chain(*corpus))
</code>



<code> 
valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff] <br>
top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
</code>