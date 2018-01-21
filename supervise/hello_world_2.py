# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 23:13:34 2017

@author: Kunwar
"""
from sklearn import tree
import numpy as np

features=np.array([[300,2],[450,2],[200,8],[500,6],[1000,2]])
labels=np.array([["sports-car"],["sports-car"],["mineven"],["Xuv"],["Super-sports car"]])

clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)

print(clf.predict([500,5]))