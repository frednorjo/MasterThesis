#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:05:14 2021

@author: fredrikjohansen
"""

#%% Code to evaluate out-of-sample performance. 
import statsmodels.api as sm
import pandas as pd
import numpy as np
import os

# Import data 
os.chdir("/users/fredrikjohansen/Documents/MasterThesis")
filename = "testData.csv"
data = pd.read_csv(filename)

#%% Include a smaller dataset 
index = 1000
data1 = data.iloc[0:index,]
date  = np.unique(data1['date'])

#%%
# set up loop for expanding window
for i in range(date.shape[0]-1):
    
    # Exxtract the data
    reg_data = data1.loc[data1['date'] < date[i+1]]
    
    # Calibrate parameters on expanding window
    
    
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
        
        









