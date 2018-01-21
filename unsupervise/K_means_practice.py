# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 22:11:56 2017

@author: Kunwar
"""
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel('titanic_half.xls') 
print(type(df))
print(len(df))

for i in df:
    print (df[:,1])
#print(df.head())