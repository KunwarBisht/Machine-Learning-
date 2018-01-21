# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 21:22:18 2017

@author: Kunwar
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 09:26:01 2017

@author: Kunwar
"""

import  pandas as pd
import quandl
import math 

from datetime import datetime
import numpy as np
from sklearn import preprocessing  ,cross_validation, svm
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style
import pickle  ###to save clasifeir so it will not train data again and again 
style.use('ggplot')

quandl.ApiConfig.api_key = 'WZSfr-D7J3uDJPV7iXsK'
df=quandl.get_table('WIKI/PRICES')
#print(df.head())



df= df[['adj_open' , 'adj_high','adj_low','adj_close','adj_volume']]
#print(df.head())
df['hl_pct']=(df['adj_high'] -df['adj_close']) / df['adj_close'] * 100.0  #heigh minus low 
df['pct_change']=(df['adj_close']- df['adj_open']) /df['adj_open'] * 100.0

  
  ##     price         x            x             x
df=df[['adj_close' , 'hl_pct' , 'pct_change' ,'adj_volume']]
#print(df.head())

forcast_col ='adj_close'
df.fillna(-99999,inplace=True)
forcast_out= int(math.ceil(0.003*len(df)))
print(forcast_out)
df['label']=df[forcast_col].shift(-forcast_out)
#df.dropna(inplace=True)
print(df.head())
print(df.tail())
#print(df.tail())

###########video 4
x=np.array(df.drop(['label'],1))
x=preprocessing.scale(x)
x=x[:-forcast_out:]
x_lately=x[-forcast_out:] # this is the data which we will pridict again

#x=x[:-forcast_out+1] not required
#df.dropna(inplace=True)
df.dropna(inplace=True)
y=np.array(df['label'])

#print(len(x),len(y))

x_train, x_test, y_train, y_test =cross_validation.train_test_split(x,y,test_size=0.2)
#clf =LinearRegression() #linearRegression() alogrithm by default run 1 job
clf =LinearRegression(n_jobs=-1) # sets job more then 1 (n_jobs=10)
#clf=svm.SVR()   #can cahgne algorithm here this is new algo which we importe on top
 #clf=svm.SVR(kernel='poly') you can feed this function with kernal and check the accuracy 
clf.fit(x_train, y_train)
#saving clasifier so it will not train data again and again 
with open('linearregression.pickle','wb') as f:
    pickle.dumps(clf,f)
pickle_in=open('linearregression.pickle','rb')
clf=pickle.load(pickle_in)
accuracy=clf.score(x_test,y_test)
print(accuracy)
print(df.head())
#predication 
'''
forecast_set=clf.predict(x_lately)
#print(forecast_set ,accuracy ,forcast_out) # it will out next thirty days or more  xtock prices which we don't have

df = pd.Series(np.nan, index = [1, 2, 3, 4, 5, 6, 7])
df[7] = '2016-12-05'
df['Forecast']=np.nan
#ast_date=df.iloc[0].name
last_date = df.iloc[-1]
last_unix = time.mktime(datetime.strptime(str(last_date), "%Y-%m-%d").timetuple())
print ("Laste date is",last_date ,"last unix" ,last_unix)
#last_unix =last_date.to_datetime()
#print("lastdate" ,last_date ,type(last_date))           
#ast_unix =last_date.timestamp()
#last_unix = time.mktime(last_unix.time_tuple())
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set :
    next_date = datetime.fromtimestamp(next_unix)
    next_unix +=one_day
    df.loc[next_date]= [np.nan for _ in range(len(df.columns)-1)] +[i]
#print(df.head())
df['adj_close'].plot()
df['Forecast'].plot()
plt.legendnd(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

'''














