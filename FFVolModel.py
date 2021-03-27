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

#Estimate model 


#%%
start_of_outofsample = 20190000
unique_date = df['date'].loc[df['date'] > 20190000].unique()


#Estimate model and measure insample fit for 500 most liquid stocks in 2020
df_2020 = df.loc[df['date'] < 20200000]
#Unique stocks 
permno_2020 = df_2020.PERMNO.unique()

mean_vol = np.zeros((permno_2020.shape[0], 1))
for i in range(0, permno_2020.shape[0]):
    mean_vol[i] = df_2020['VOL'].loc[df_2020['PERMNO'] == permno_2020[i]].mean()

#choose the 500 PERMNO codes with highest average volume
permno_2020 = 
    



    
    




    

#%%    
    
    

 
                
        
    
    





