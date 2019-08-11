# -*- coding: utf-8 -*-
#Scikit-Learn中TF-IDF权重计算方法主要用到两个类：CountVectorizer和TfidfTransformer。 
#1.CountVectorizer CountVectorizer类会将文本中的词语转换为词频矩阵
#例如矩阵中包含一个元素a[i][j]，它表示j词在i类文本下的词频。它通过fit_transform函数计算各个词语出现的次数，通过get_feature_names()可获取词袋中所有文本的关键字，通过toarray()可看到词频矩阵的结果。
#2.TfidfTransformer 
#TfidfTransformer用于统计vectorizer中每个词语的TF-IDF值。
#

from sklearn.feature_extraction.text import CountVectorizer

#语料
corpus = [
    'This is the first document.',
    'This is the this second second document.',
    'And the third one.',
    'Is this the first document?'
]

#将文本中的词转换成词频矩阵
vectorizer = CountVectorizer()
print(vectorizer)
#计算某个词出现的次数
X = vectorizer.fit_transform(corpus)
print(type(X),X)
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print(word)
#查看词频结果
print(X.toarray())


from sklearn.feature_extraction.text import TfidfTransformer
#类调用
transformer = TfidfTransformer()
print(transformer)
#将词频矩阵统计成TF-IDF值
tfidf = transformer.fit_transform(X)
#查看数据结构tfidf[i][j]表示i类文本中tf-idf权重
print(tfidf.toarray())

from sklearn import metrics as mr
mr.mutual_info_score(label,x)
#label、x为list或array

from sklearn.feature_selection import SelectKBest
from minepy import MINE
# 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
# 选择K个最好的特征，返回特征选择后的数据
SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)