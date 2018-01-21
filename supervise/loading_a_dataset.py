# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 07:57:06 2017

@author: Kunwar
"""
from sklearn import datasets

iris=datasets.load_iris()
digits=datasets.load_digits()
print(digits.data)