# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 21:53:44 2017

@author: Kunwar
"""

from math import sqrt
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random
# r and k are classes having two dimensional features 
dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]} 
new_features=[5,7]

#for i in dataset:
    #for ii in dataset[i]:
#[[plt.scatter(ii[0],ii[1], s=100, color=i)for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0],new_features[1])
#plt.show() 
      
#euclidean_distance=sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )
#print(euclidean_distance)

# dat is the testing data (groups)   
def k_nearest_neighbors(data,predict,k=3): 
    if len(data) >=k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances=[]
    for  group in data:
        for features in data[group]:
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict)) #using linear algebra inbuilt function it si faster
            # euclidean_distance =np.sqrt(np.sum(( np.array(features) -np.array(predict))**2))  #using normal formaula dynamic but little slower for large dat
            distances.append([euclidean_distance ,group])
    
    votes=[i[1] for i in sorted(distances)[:k]]  #  i[1] is group in distance
    
    #print(Counter(votes).most_common(1))
    vote_result =Counter(votes).most_common(1)[0][0] #[0][0] first 0 gives  common list second 0 gives you the common first list of array secod (most common group and how many they were)
   
    
    
    return vote_result 
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/wdbc/wdbc.data', header=None)
df.replace('?',-99999 , inplace=True)
#df.drop(['id'],1,inplace=True)
full_data =df.values.tolist()
#print(full_data)

random.shuffle(full_data) #for suffeling data
test_size =0.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))]#will hold all but not last value which is 0.2 *len(fulldata) =initial 80% data
test_data=full_data[-int(test_size*len(full_data)): ]#will hold  last value which is 0.2 *len(fulldata) =last 20 %data
print(train_data)
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])
print (train_set)
    
correct =0
total=0 
'''
for group in test_set:
    for data in test_set[group]:
        vote =k_nearest_neighbors(train_set,  data, k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy :', correct/total)
'''
  







