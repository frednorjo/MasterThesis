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






