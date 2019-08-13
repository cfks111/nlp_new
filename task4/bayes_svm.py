# -*- coding: utf-8 -*-
import numpy as np
import sklearn

from sklearn.datasets import fetch_20newsgroups
twenty_train=fetch_20newsgroups(subset='train',shuffle=True)
twenty_train.target_names

from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
X_train_counts=count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

A=X_train_counts[0]
print(A.data)
print(A.indptr)
print(A.indices)
print(count_vect.get_feature_names()[A.indices[0]])

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(X_train_counts,twenty_train.target)
twenty_test=fetch_20newsgroups(subset='test',shuffle=True)
X_test_counts=count_vect.transform(twenty_test.data)

predicted=clf.predict(X_test_counts)
np.mean(predicted==twenty_test.target)

#改进 tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)
tfidf_clf=MultinomialNB().fit(X_train_tfidf,twenty_train.target)

X_test_tfidf=tfidf_transformer.fit_transform(X_test_counts)
predicted=tfidf_clf.predict(X_test_tfidf)
np.mean(predicted==twenty_test.target)

#去除停用词
sw_count_vect=CountVectorizer(stop_words='english')
X_train_counts=sw_count_vect.fit_transform(twenty_train.data)
X_test_counts=sw_count_vect.transform(twenty_test.data)
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf=tfidf_transformer.transform(X_test_counts)
tfidf_clf=MultinomialNB().fit(X_train_tfidf,twenty_train.target)
predicted=tfidf_clf.predict(X_test_tfidf)
np.mean(predicted==twenty_test.target)

#svm
from sklearn.linear_model import SGDClassifier
svm_clf=SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,max_iter=5,random_state=42)
svm_clf.fit(X_train_tfidf,twenty_train)
predicted=svm_clf.predict(X_test_tfidf)
np.mean(predicted==twenty_test.target)

from sklearn.linear_model import SGDClassifier
svm_clf=SGDClassifier(loss='hinge',penalty='l2',alpha=8e-5,max_iter=5,random_state=50)
svm_clf.fit(X_train_tfidf,twenty_train)
predicted=svm_clf.predict(X_test_tfidf)
np.mean(predicted==twenty_test.target)










#https://www.cnblogs.com/trickofjoker/p/9306851.html