#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:26:24 2021

@author: fredrikfusdahl
"""
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
#from masterthesisfunctions import *
import statsmodels.api as sm

#Import CRSP data
data = pd.read_csv('testData.csv')
data = data.dropna()
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
#Get unique set with company permno
permno = data.PERMNO.unique()

#Import VIX
vix = pd.read_csv('VIX.csv')
#Clean data. Split string into date vector and price vector. 
vix = vix.squeeze()
vix = vix.str.split(pat=';', expand = True)
vix = vix.drop(labels = vix[vix.iloc[:,1].str.strip() == ''].index)
vix.iloc[:,1] = vix.iloc[:,1].str.strip().astype('float')
vix.iloc[:,0] = pd.to_datetime(vix.iloc[:,0], format='%Y-%m-%d')
vix.columns = ['date', 'VIX']
vix.set_index('date', inplace=True)

#Merge VIX and CRSP data
finRatios = pd.read_csv('finRatios.csv')
finRatios['public_date'] = pd.to_datetime(finRatios['public_date'], format='%Y%m%d')
finRatios.rename(columns={'public_date': 'date'}, inplace=True)

#%%
x = data.iloc[(data.PERMNO == permno[0]).tolist(), :]
x = pd.merge_asof(x, finRatios.iloc[(finRatios.permno == permno[0]).tolist(), :], on='date')

#create measure for daily vol
x['dVOL'] = np.log(x['ASKHI']) - np.log(x['BIDLO'])

#Weekly vol
x['wVOL'] = (x['dVOL'].rolling(5).sum())/5

#Monthly vol
x['mVOL'] = (x['dVOL'].rolling(20).sum())/20


#Run a simple OLS regression to test fit
model = sm.OLS(x['dVOL'] , missing='drop')
res = model.fit()
print(res.summary())


























#%%











    
    
    
    
    
    






















