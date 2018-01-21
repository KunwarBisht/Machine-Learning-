# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 18:01:24 2017

@author: Kunwar
"""
import numpy as np
from sklearn import preprocessing, cross_validation , svm
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/wdbc/wdbc.data', header=None)
df.replace('?', -99999 ,inplace=True)
#df.drop(['id'],1, inplace=True)

x=np.array(df.drop(['class'],1))
y=np.array(df['class'])
x_train ,x_test ,y_train, y_test =cross_validation.train_test_split(x,y,test_size=0.2)
clf=svm.SVC()
clf.fit(x_train,y_train)
accuracy=clf.score(x_train , y_train)
print(accuracy)

example_measures =np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
example_measures=example_measures.reshape(len(example_measures),-1)
prediction =clf.predict(example_measures)
print(prediction)