# -*- coding: utf-8 -*-

import pandas as pd
import jieba
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC

def read_data(url):
    data = pd.read_csv(url,encoding='utf-8')
    data.fillna("null",inplace=True)      #使用inplace参数会改掉本身 
    return data

def clean_text(text):
    text = str(text)
    text = text.replace('\n', '')
    text = text.replace('<br />', ' ')
    text = text.replace(';', ',')
    return text
def tokenize_df_content(data):
    df_empty = pd.DataFrame()
    return df_empty.assign(content_tokens=data["content"].map(clean_text)).assign(title_tokens=data['title'].map(clean_text)).assign(label_tokens=data['ann_labels'].map(clean_text))
#将content&title&ann_labels的内容都进行过清洗，形成一个新的dataframe

def stop_words():
    stop_words_file = open('stopword.txt', 'r')
    stopwords_list = []
    for line in stop_words_file.readlines():
        stopwords_list.append(line)
    return stopwords_list

def jieba_data(data,i):
    stopwords_list = stop_words()
    data_title=jieba.cut(data['title_tokens'][i])
    data_content=jieba.cut(data["content_tokens"][i])
    data_result = " "
    for title_word in data_title:
        if(title_word not in stopwords_list):
            data_result += title_word+" "
    for content_word in data_content:
        if(content_word not in stopwords_list):
         data_result += content_word+" "
    
    return data_result

def jieba_label(data_result,data,i):
    stopwords_list = stop_words()
    data_label = jieba.cut(data['label_tokens'][i])
    for label_word in data_label:
        if(label_word not in stopwords_list):
            data_result += label_word+" "
    
    return data_result

def train_jieba(data,i):
    data_result = jieba_data(data,i);
    result = jieba_label(data_result,data,i);
    return result;

def test_jieba(data,i):
    data = jieba_data(data,i);
    return data;

if __name__ == '__main__':
    train_data_x = read_data('df_news_train_ecnu.csv')
    test_data_x = read_data('df_news_test_ecnu.csv')
    y_train = np.loadtxt("y_train_ecnu.csv", delimiter=",",encoding='utf-8')  #读入文件并以矩阵或向量的形式输出
    y_test = np.loadtxt("y_test_ecnu.csv", delimiter=",",encoding='utf-8')
    print(train_data_x.shape)
    print(test_data_x.shape)
    
    unimportant_idx = 44
    train_label_y = y_train[:, unimportant_idx]
    test_label_y = y_test[:, unimportant_idx]
    print("# unimportant / total news in training data: ", int(sum(y_train[:, unimportant_idx])), "/", len(y_train[:, unimportant_idx]))

    train_data_clean_x = tokenize_df_content(train_data_x)
    test_data_clean_x = tokenize_df_content(test_data_x)
    print(train_data_x.shape)
    print(test_data_x.shape)
    print(train_data_clean_x.shape)
    print(test_data_clean_x.shape)   #此时的标签是content_tokens title_tokens label_tokens
    
    train_x = []       #训练集的语料库
    test_x = []        #测试集的语料库
    for i in range(len(train_data_clean_x)):
            train_x.append(train_jieba(train_data_clean_x,i))
    for i in range(len(test_data_clean_x)):
            test_x.append(test_jieba(test_data_clean_x,i))
 
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    np.savetxt("tmp/X_train.csv", train_x, delimiter=",",fmt = "%s",encoding="utf-8")
    np.savetxt("tmp/X_test.csv", test_x, delimiter=",",fmt = "%s",encoding="utf-8")

    vectorizer = CountVectorizer()  #只考虑词汇在文本中出现的频率
    transformer = TfidfTransformer()  #tfidf值
    train_tfidf=transformer.fit_transform(vectorizer.fit_transform(train_x))#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  
    print(train_tfidf.shape)
    test_tfidf = transformer.transform(vectorizer.transform(test_x))  #先拟合fit，找到该part的整体指标，对于test，transform保证train、test处理方式相同。
#     test_weight = test_tfidf.toarray()
#     print(test_weight.shape)

    weight = train_tfidf.toarray() 
    word = vectorizer.get_feature_names()  #获取词袋模型中的所有词语 
    for i in range(5):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
        print("************************第",i,"篇文章的词语tf-idf权重**********************"  )
        for j in range(len(word)):
            if(weight[i][j] >0.1):   
                print(word[j],weight[i][j])


    svmmodel = SVC(C = 1 , kernel= "linear") #kernel：rbf, poly在这里都没有线性的好

    nn = svmmodel.fit(train_tfidf,train_label_y)
    print(nn)
    pre_test_label = svmmodel.predict(test_tfidf)
    test_data_x["category"] = pre_test_label
    np.savetxt("tmp/test_data_label.csv", test_data_x, delimiter=",",fmt ="%s",encoding="utf-8")
               
               
    from sklearn.metrics import classification_report
    print(classification_report(test_label_y, pre_test_label))
