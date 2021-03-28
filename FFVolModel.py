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
df['dVOL_2'] = df['dVOL']**2
df['5VOL_2'] = df['5VOL']**2
df['20VOL_2'] = df['20VOL']**2
df = df.dropna()

#%%# Estimate model HAR model in panel data
X = df[['dVOL', '5VOL', '20VOL', 'dVOL_2', '5VOL_2', '20VOL_2']]
y = df['future']
lm = LinearRegression().fit(X,y)

#%%
#Estimate model and measure insample fit for 500 most liquid stocks in 2020
df_2020 = df.loc[df['date'] > 20200000]
permno_2020 = df_2020.PERMNO.unique()
mean_volume = df_2020.groupby('PERMNO')['VOL'].median()

#choose the 500 PERMNO codes with highest average volume
ind = np.argsort(mean_volume)[::-1][0:50]
r2_vec3 = np.zeros((len(ind), 1))



#%%
for i in range(0, len(ind)):
    #lets test with one stock only
    df_stock = df[df['PERMNO'] == permno_2020[ind.iloc[i]]]
    #Predict
    y_hat = lm.predict(df_stock[['dVOL', '5VOL', '20VOL', 'dVOL_2', '5VOL_2', '20VOL_2']])
    
    #Plot predicted vol with with realized vol
    r_squared = R2(df_stock['dVOL'], y_hat)
    r2_vec3[i] = r_squared
    x = range(1, y_hat.shape[0]+1)
    plt.plot(x, df_stock['dVOL'])
    plt.plot(x, y_hat, 'o')
    plt.title(str(r_squared))
    plt.show()



#%%
plt.plot(range(1, len(r2_vec)+1), r2_vec, 'g-')
plt.plot(range(1, len(r2_vec)+1), r2_vec1, 'r-')
plt.plot(range(1, len(r2_vec)+1), r2_vec3, 'b-')
        

    

    


    




#%%    
    
    

 
                
        
    
    





