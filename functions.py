import pandas as pd 
import numpy as np
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from functions import *
from sklearn.linear_model import LinearRegression

def RV(df, typeVOL = None): 
    #First calculate daily Volatility
    
    
    df['dVOL'] = np.log(df['ASKHI']) - np.log(df['BIDLO'])
    df['dVOL'] = df['dVOL'].div(df.groupby('PERMNO')['dVOL'].transform('mean'))
    if(typeVOL != None):
        if(len(typeVOL)>0): 
            for typ in typeVOL:
                df[str(typ)+'VOL'] = df.groupby('PERMNO')['dVOL'].rolling(typ).mean().reset_index(0,drop=True)
        else:
            df[str(typeVOL)+'VOL'] = df.groupby('PERMNO')['dVOL'].rolling(typVOL).mean().reset_index(0,drop=True)
    
    return(df)



def sorted_enumerate(seq):
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]

def R2(y, yhat):
    SS_Residual = sum((y-yhat)**2)       
    SS_Total = sum((y-np.mean(y))**2)     
    r_squared = 1 - (float(SS_Residual))/SS_Total
    #adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
    return r_squared


def addLag(df, n_lags):
    for i in range(1,n_lags+1):
        df['dVOL_lag' + str(i)] = df.groupby('PERMNO')['dVOL'].shift(i).reset_index(0,drop=True)
    return(df)


def inSampleOLS(df, permno, type_vol, plot = False):
    if type_vol == 'daily':
        df['future'] = df.groupby('PERMNO')['dVOL'].shift(-1)
        df = df.dropna()
    
    elif type_vol == 'weekly':
        df['future'] = df.groupby('PERMNO')['5VOL'].shift(-1)
        df = df.dropna()
    
    else:
        df['future'] = df.groupby('PERMNO')['20VOL'].shift(-1)
        df = df.dropna()
    
    
   
    
    df_stock = df[df['PERMNO'] == permno]
    
    if type_vol == 'daily':
        X = df.iloc[:,df.columns.get_loc('dVOL'):-1]
        #Fit model 
        lm = LinearRegression().fit(X, df['future'])
        #Predict
        y_hat = lm.predict(df_stock.iloc[:,df.columns.get_loc('dVOL'):-1])
    
    
    elif type_vol == 'weekly':
        X = df.iloc[:,df.columns.get_loc('5VOL'):-1]
        #Fit model 
        lm = LinearRegression().fit(X, df['future'])
        #Predict
        y_hat = lm.predict(df_stock.iloc[:,df.columns.get_loc('5VOL'):-1])
    
        
    else:
        X = df.iloc[:,df.columns.get_loc('20VOL'):-1]
        #Fit model 
        lm = LinearRegression().fit(X, df['future'])
        #Predict
        y_hat = lm.predict(df_stock.iloc[:,df.columns.get_loc('20VOL'):-1])
        
        
    return y_hat, df_stock['future']


        
    
    
        
    



