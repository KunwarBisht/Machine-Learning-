# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 09:01:46 2017

@author: Kunwar
"""

import panda as pd
import Quandl as qd 
df =qd.get('WIKI/GOOGL')
print(df.head())
