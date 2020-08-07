# Lec 14: More on Contextual Word Representations and **Pretraining**

解决"<unk>": test 时, 随机分配 vector 并加入 vocab; 考虑 word classes(如数字,符号)。

## 1. Representation for a word

    传统的word vectors(Word2Vec, GloVe, fastText) 存在两个问题:
    - 不考虑上下文
    - 单一,不能表示词的多个含义/词性POS

    解决方案?
    - NLM中用LSTM层捕捉上下文信息，生成context-specific word representation
    - 提取出LSTM生成的表征?

## 2. TagLM - Pre-ELMo

    **Tag LM (Peters et al. 2017) LSTM BiLM in BiLSTM tagger**
    - pretrained bi-LM。这块是一个黑盒(frozen LM)，结果h_k^LM被直接应用到embedding concatenation。大概也是TagLM名字的由来, LM的产出被贴到左边LSTM embedding上。
    - Semi-supervised, 对于unlabeled data 同时训练WordEmb 和 RLM 两个模型。
    - 每个词有两种表征: word embedding + LM embedding.
    - 最上层sequence tagging model同时应用两个embedding

    - LM embedding是forward LM和backward LM 的 embedding的左右拼接, BiLM
    - h_(k,1) = [h_(k,1)-> ; h_(k,1)<- ; h_k^LM] CharCNN/RNN embedding, Token embedding(直接的embedding), LM Embedding

    Named Entity Recognition(NER)

    **Peters et al. (2018): ELMo: Embeddings from Language Models**
    - LM中不只用最上面一层LSTM的输出
    - 两个参数 gamma^task 和 s_j^task，一个对于句子整体对于task有用程度的权重，一个是softmax的每个词的权重
    
    - 2 biLSTM NLM layers, 并不算深, 两层各自有不同的意义:
      - 底层捕捉lower-level syntax, 如词性、句法依存关系、NER等
      - 上层捕捉higher-level semantics, 如情感、问题回答等

## 3. CoVe, ULMfit(Universal LM Fine-tuning for Text Classification)

    **ULMfit**
    - 类似之前，pretrain LM，一开始在general domain corpus上用biLM。
    - 然后对于target task data 单独 Fine-tune

    ULMfit transfer learning
    - semi-supervised 在val上的表现最好

## 4. Transformer models
    Transformer allows scaling. Faster so we can build bigger models.
    **Parallelization** instead of **Recurrent**
    **Attention is all you need**: no recurrent.
    Jupyter Notebook: <http://nlp.seas.harvard.edu/2018/04/03/attention.html>


    6 tranformer layers.
    each layer: multihead attention, add+norm, feedforward, add+norm

## 5. BERT
    Mask out (15%)
    Input -> Token Embeddings + Segment Embeddings + Position Embeddings 
    Fine tune.
    