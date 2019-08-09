# -*- coding:utf-8 -*-
import re
import string
import operator


def cleanText(input):
    input = re.sub('\n+', " ", input).lower()  # 匹配换行,用空格替换换行符
    input = re.sub('\[[0-9]*\]', "", input)  # 剔除类似[1]这样的引用标记
    input = re.sub(' +', " ", input)  # 把连续多个空格替换成一个空格
    input = bytes(input)  # .encode('utf-8') # 把内容转换成utf-8格式以消除转义字符
    # input = input.decode("ascii", "ignore")
    return input


def cleanInput(input):
    input = cleanText(input)
    cleanInput = []
    input = input.split(' ')  # 以空格为分隔符，返回列表

    for item in input:
        item = item.strip(string.punctuation)  # string.punctuation获取所有标点符号

        if len(item) > 1 or (item.lower() == 'a' or item.lower() == 'i'):
            cleanInput.append(item)
    return cleanInput


def getNgrams(input, n):
    #把一段英文处理成一个个词语，保留了分词后每个词在原短文中的顺序
    #input = cleanInput(input)

    output = {}  # 构造字典
    for i in range(len(input) - n + 1):
        ngramTemp = " ".join(input[i:i + n])
        if ngramTemp not in output:  # 词频统计
            output[ngramTemp] = 0
        output[ngramTemp] += 1
    return output


# 获取数据，content为一段英文
txt1 = open("/Users/cfks111/Desktop/nlpnew/task2/cnews.val.txt", 'r', encoding='utf8') # 打开文件，并读取要处理的大段文字
txt1=txt1.readline()
print(txt1)
#n-grams
ngrams = getNgrams(txt1, 1)
ngrams = getNgrams(txt1, 2)
ngrams = getNgrams(txt1, 3)

sortedNGrams = sorted(ngrams.items(), key=operator.itemgetter(1), reverse=True)
print(sortedNGrams)
