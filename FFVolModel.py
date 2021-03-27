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



os.chdir("/Users/fredrikfusdahl/Documents/Master Thesis")
#Import CRSP data
data = pd.read_csv('testData.csv')
data = data.dropna()
#data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
#Get unique set with company permno
permno = data.PERMNO.unique()

#%% 
#We will first run the 
data['CLOSEPRC'] = (data['ASK'] + data['BID'])/2
data['SIZE'] = data['SHROUT']*data['CLOSEPRC']
