# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 23:35:57 2017

@author: Kunwar
"""
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
iris=load_iris()
test_idx=[0,50,100]  ## index 0,50,100  data

#training data
train_target=np.delete(iris.target ,test_idx)
train_data=np.delete(iris.data ,test_idx, axis=0)

print(train_target,train_data)

#testing data
test_target =iris.target[test_idx]
test_data = iris.data[test_idx]

#classfier
clf=tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print(test_target)
print(clf.predict(test_data))

##visual code
from sklearn.externals.six import StringIO
import pydot
dot_data=StringIO() 
tree.export_graphviz(clf, out_file=dot_data, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         impurity=False)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
#graph.write_pdf("iris.pdf") 

print (test_data[0] ,test_target[0])
print(iris.feature_names ,iris.target_names)


'''
print(iris.data[0])
print(iris.target[0])
print(iris.feature_names)
print(iris.target_names)


for i in range(len(iris.target)):
    print("Example %d : label %s : features %s " %(i,iris.target[i],iris.data[i]))
'''