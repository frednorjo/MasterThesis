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

data['CLOSEPRC'] = (data['ASK'] + data['BID'])/2
data['SIZE'] = data['SHROUT']*data['CLOSEPRC']

#Create measure for daily, weekly and monthly vol. 
df = RV(data, [5, 20])

#Estimate model HAR model in panel data
df['future'] = df.groupby('PERMNO')['dVOL'].shift(-1)
df = df.dropna()

#%%# Estimate model HAR model in panel data
X = df[['dVOL', '5VOL', '20VOL']]
y = df['future']
lm = LinearRegression().fit(X,y)

#%%
#Estimate model and measure insample fit for 500 most liquid stocks in 2020
df_2020 = df.loc[df['date'] > 20200000]
mean_volume = df_2020.groupby('PERMNO')['VOL'].mean()

#choose the 500 PERMNO codes with highest average volume
ind = sorted_enumerate(mean_volume)
permno_2020 = permno_2020[ind[-500:-1]]

for i in range(start = 0, stop = permno_2020.shape[0]):
    
#%%
mean_volume = df_2020.groupby('PERMNO')['VOL'].mean()
    
    

    

    


    




    

#%%    
    
    

 
                
        
    
    





