#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:13:01 2021

@author: fredrikfusdahl
"""

import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from functions import *
from sklearn.linear_model import LinearRegression
os.chdir("/Users/fredrikfusdahl/Documents/MasterThesis")
#Import CRSP data
data = pd.read_csv('testData.csv')
data = data.dropna()
#data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
#Get unique set with company permno
permno = data.PERMNO.unique()
most_active_options = pd.read_csv('active_options.csv')
most_active_options_permno = most_active_options['PERMNO'].unique()
most_active_options_ticker = most_active_options['TICKER'].unique()


data['CLOSEPRC'] = (data['ASK'] + data['BID'])/2
data['SIZE'] = data['SHROUT']*data['CLOSEPRC']

#Create measure for daily, weekly and monthly vol. 
df = RV(data, [5, 20])
df = addLag(df, 5)
#%%
y_hat, y = inSampleOLS(df, most_active_options_permno[0], 'daily')
r_2 = R2(y, y_hat) 

n1 = 0
n2 = 400
x = range(1, y_hat.shape[0]+1)
plt.plot(x[n1:n2], y[n1:n2], '-')
plt.plot(x[n1:n2], y_hat[n1:n2], '-')
plt.title(most_active_options_ticker[0]+ " " + str(r_2))








#%%    


    
    

 
                
        
    
    





