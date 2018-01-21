# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 09:26:01 2017

@author: Kunwar
"""

import  pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing  ,cross_validation, svm
from sklearn.linear_model import LinearRegression 

quandl.ApiConfig.api_key = 'WZSfr-D7J3uDJPV7iXsK'
df=quandl.get_table('WIKI/PRICES')
#print(df.head())
#print(df.head())


df= df[['adj_open' , 'adj_high','adj_low','adj_close','adj_volume']]
#print(df.head())
df['hl_pct']=(df['adj_high'] -df['adj_close']) / df['adj_close'] * 100.0  #heigh minus low 
df['pct_change']=(df['adj_close']-df['adj_open']) /df['adj_open'] * 100.0

df=df[['adj_close' , 'hl_pct' , 'pct_change' ,'adj_volume']]
#print(df.head())

forcast_col ='adj_close'
df.fillna(-99999,inplace=True)
forcast_out= int(math.ceil(.1*len(df)))

df['label']=df[forcast_col].shift(-forcast_out)
df.dropna(inplace=True)
#print(df.head())
#print(df.tail())

###########video 4
x=np.array(df.drop(['label'],1))
y=np.array(df['label'])
x=preprocessing.scale(x)
#x=x[:-forcast_out+1] not required
#df.dropna(inplace=True)
y=np.array(df['label'])
#print(len(x),len(y))

x_train, x_test, y_train, y_test =cross_validation.train_test_split(x,y,test_size=0.2)
#clf =LinearRegression() #linearRegression() alogrithm by default run 1 job
clf =LinearRegression(n_jobs=-1) # sets job more then 1 (n_jobs=10)
#clf=svm.SVR()   #can cahgne algorithm here this is new algo which we importe on top
 #clf=svm.SVR(kernel='poly') you can feed this function with kernal and check the accuracy 
clf.fit(x_train, y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)














