# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 22:20:24 2017

@author: Kunwar
"""

from sklearn import tree
import numpy as np
##collect data

features=np.array([[140 ,1] ,[130,1],[150,0],[170,0]])

labels=np.array([0,0,1,1])

##train classifier  

clf =tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)

##make predication  0=mango(smooth) ,1=orange (pumpy)  ,,
print clf.predict([[150,0]])