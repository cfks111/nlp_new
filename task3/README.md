##task2:

#####1. TF-IDF原理。
***

* TF-IDF(Term Frequency-Inverse DocumentFrequency, 词频-逆文件频率)

*  TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

* 词频TF（item frequency）：某一给定词语在该文本中出现次数。该数字通常会被归一化（分子一般小于分母），以防止它偏向长的文件，因为不管该词语重要与否，它在长文件中出现的次数很可能比在段文件中出现的次数更大。

* 逆向文件频率IDF（inverse document frequency）：一个词语普遍重要性的度量。主要思想是：如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到。
 
* 公式：TF=词的次数/总词数；IDF=log(语料库文档总数/包含该词的文档数+1)；TF-IDF=TF*IDF

#####2. 文本矩阵化，使用词袋模型，以TF-IDF特征值为权重。（可以使用Python中TfidfTransformer库）：见tfidf.py
***
#####3. 互信息的原理。
***

* 点互信息PMI（Pointwise Mutual Information）:点互信息用来衡量两个事物之间的相关性。
![avatar](/Users/cfks111/Desktop/nlpnew/task3/pmi.png)
* 在概率论中，如果x跟y不相关，则 P(x,y) = P(x)P(y)。二者相关性越大，则 P(x,y) 就相比于 P(x)P(y) 越大。在y出现的情况下x出现的条件概率 p(x|y) 除以x本身出现的概率 p(x) ，自然就表示x跟y的相关程度。log是单调递增函数，而且 log 1 = 0 ，则表明P(x,y) = P(x)P(y)，此时相关性为0
* 互信息MI（Mutual Information）:
互信息是信息论里一种有用的信息度量，它可以看成是一个随机变量中包含的关于另一个随机变量的信息量，或者说是一个随机变量由于已知另一个随机变量而减少的不肯定性 
![avatar](/Users/cfks111/Desktop/nlpnew/task3/mutual.png)
* 互信息其实就是对X和Y的所有可能的取值情况的点互信息PMI的加权和
* NLP中的应用:互信息值越高, 表明 X 和 Y 相关性越高, 则 X 和 Y 组成短语的可能性越大; 反之, 互信息值越低,X 和Y之间相关性越低, 则X 和Y之间存在短语边界的可能性越大。公式中的X和Y指的是两个相邻的单词，P值是它的出现概率
*  其实我们所说的互信息，其实就是等同于决策树中定义的信息增益:具体参考https://blog.csdn.net/sir_TI/article/details/93895042
#####4. 使用第二步生成的特征矩阵，利用互信息进行特征筛选：见tfidf.py


