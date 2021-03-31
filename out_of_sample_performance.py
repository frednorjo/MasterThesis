#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:05:14 2021

@author: fredrikjohansen
"""

#%% Code to evaluate out-of-sample performance. 
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from functions import *
from sklearn.linear_model import LinearRegression

# Import data 
os.chdir("/users/fredrikjohansen/Documents/MasterThesis")
filename = "testData.csv"
data = pd.read_csv(filename)

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

#%%
#Estimate model and measure insample fit for 500 most liquid stocks in 2020
df_2020 = df.loc[df['date'] > 20200000]
permno_2020 = df_2020.PERMNO.unique()
mean_volume = df_2020.groupby('PERMNO')['VOL'].median()

#choose the 500 PERMNO codes with highest average volume
ind = np.argsort(mean_volume)[::-1].iloc[1]

#%% Include a smaller dataset 
start_forcast = 20190000
date_vector = data['date'].unique()[data['date'].unique() > start_forcast] 

#%% Out-of-sample
y_hat = np.zeros((date_vector.shape[0],1))
#%%
# set up loop for expanding window
for i in range(130,date_vector.shape[0]):
    
    # Index data
    forcast_date = date_vector[i] 
    df_train = df.loc[df['date'] < forcast_date]
    
    # make a dataframe with test data
    df_test = df.loc[df['date'] == forcast_date]
    
    # Fit model 
    X_train = df_train[['dVOL', '5VOL', '20VOL', 'dVOL_2', '5VOL_2', '20VOL_2']]
    y_train = df_train['future']
    lm = LinearRegression().fit(X_train,y_train)
    
  
    #lets test with one stock only
    df_stock = df_test[df_test['PERMNO'] == permno_2020[ind]]
    if df_stock.empty == True:
        continue
    else:
        y_hat[i] = lm.predict(df_stock[['dVOL', '5VOL', '20VOL', 'dVOL_2', '5VOL_2', '20VOL_2']])

#%%
x = np.arange(1,505,1)
plt.figure(1)

df_stock = df[df['PERMNO'] == permno_2020[ind]]

plt.plot(x,df_stock['dVOL'][df_stock['date'] > start_forcast])
plt.plot(x,y_hat[0:504,0])      
  #%%

  for j in range(0, len(i)):
        #lets test with one stock only
        df_stock = df_test[df_test['PERMNO'] == permno_2020[ind.iloc[j]]]
        if df_stock.empty == True:
            continue
        else:
            y_hat[i,j] = lm.predict(df_stock[['dVOL', '5VOL', '20VOL', 'dVOL_2', '5VOL_2', '20VOL_2']])
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#%% Make function that calculates increments with different increments of expanding window
def expanding_window(data, start_year, end_year, options, i):
    # Option can take values 1 to 3
        # 1 = 1 day increment
        # 2 = 1 month increment
        # 3 = 1 year increment
    # start_date = starting year of dataset
    # end_date   = end year of dataset
    # data = dataframe with the data
    # i = index of the element in the date vector  
    from datetime import datetime
    
    if options == 1:
        # Make vector with unique dates
        date = np.unique(data)
            
        # Exxtract the data
        reg_data = data.loc[data['date'] < date[i+1]] 
        return(reg_data)                       
    
    if options == 2:
        date = np.arange(start_year, end_year+1, 1)
        
        









