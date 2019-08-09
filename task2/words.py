# -*- coding: utf-8 -*-

import jieba.analyse
from collections import Counter

#字频统计
txt1 = open("/Users/cfks111/Desktop/nlpnew/task2/cnews.val.txt", 'r', encoding='utf8').read()  # 打开文件，并读取要处理的大段文字
txt1 = txt1.replace('\n', '')  # 删掉换行符
txt1 = txt1.replace('，', '')  # 删掉逗号
txt1 = txt1.replace('。', '')  # 删掉句号
txt1 = txt1.replace(' ', '')  # 删掉空格
txt1 = txt1.replace('、', '')  # 删掉 、
mylist = list(txt1)
mycount = Counter(mylist)
for key, val in mycount.most_common(10):  # 有序（返回前10个）
    print(key, val)

#词频统计
text=''
f = open('/Users/cfks111/Desktop/nlpnew/task2/cnews.val.txt', 'r', encoding='utf8')  # 要进行分词处理的文本文件 (统统按照utf8文件去处理，省得麻烦)
#lines = f.readlines()
lines = f.readline()
lines = lines.replace('\n', '')  # 删掉换行符
lines = lines.replace('，', '')  # 删掉逗号
lines = lines.replace('。', '')  # 删掉句号
lines = lines.replace(' ', '')  # 删掉空格
lines = lines.replace('、', '')  # 删掉 、

for line in lines:
    text += line
print(text)

# seg_list = jieba.cut(text, cut_all=False)  #精确模式（默认是精确模式）
seg_list = jieba.cut(text)  # 精确模式（默认是精确模式）
print("[精确模式]: ", "/ ".join(seg_list))

seg_list2 = jieba.cut(text, cut_all=True)    #全模式
print("[全模式]: ", "/ ".join(seg_list2))

seg_list3 = jieba.cut_for_search(text)    #搜索引擎模式
print("[搜索引擎模式]: ","/ ".join(seg_list3))

tags = jieba.analyse.extract_tags(text, topK=5)
print("关键词:    ", " / ".join(tags))

#--------------------------------------------------
#https://www.cnblogs.com/chjxbt/p/10642894.html
#明明已经导包了，为什么使用包里面的方法却报错没有这个属性，原因是有可能导错包了，你要导的包有重名，导致导的这个包其实并不是正确的包
#有可能是因为当前目录中有文件名与导入的包名重名了，导致文件冲突。在python中，在导入模块时，模块的搜索顺序是：
#1、当前程序根目录
#2、PYTHONPATH
#3、标准库目录
#4、第三方库目录site-packages目录
#三、解决方法
#经过分析后，发现我自己的目录下，也有一个同名的jieba.py文件，导致在其他文件中导入jieba这个包时，首先导入当前目录下的文件。
#通过把当前目录下重名的文件修改文件名后，完美解决问题
