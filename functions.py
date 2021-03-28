import pandas as pd 
import numpy as np

def RV(df, typeVOL = None): 
    #First calculate daily Volatility
    
    
    df['dVOL'] = np.log(df['ASKHI']) - np.log(df['BIDLO'])
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






