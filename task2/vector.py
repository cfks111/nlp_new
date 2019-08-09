# -*- coding: utf-8 -*-
import jieba
import re

#3.1 分词（可采用结巴分词来进行分词操作，其他库也可以）；
#3.2 去停用词；构造词表。
stopwords = {}
fstop = open('/Users/cfks111/Desktop/nlpnew/task2/stopwords.txt', 'r',encoding='utf-8',errors='ingnore')
for eachWord in fstop:
    stopwords[eachWord.strip()] = eachWord.strip()  #停用词典
fstop.close()
f1=open('/Users/cfks111/Desktop/nlpnew/task2/cnews.val.txt','r',encoding='utf-8',errors='ignore')
f2=open('/Users/cfks111/Desktop/nlpnew/task2/fenci.txt','w',encoding='utf-8')

line=f1.readline()
while line:
    line = line.strip()  #去前后的空格
    line = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ", line) #去标点符号
    seg_list=jieba.cut(line,cut_all=False)  #结巴分词
    outStr=""
    for word in seg_list:
        if word not in stopwords:
            outStr+=word
            outStr+=" "
    f2.write(outStr)
    line=f1.readline()
f1.close()
f2.close()


#3.3 每篇文档的向量化。
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer=CountVectorizer()
corpus=["I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"] 
print(vectorizer.fit_transform(corpus))