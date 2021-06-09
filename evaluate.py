import math


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


from pydataset import data

def plot_residuals(y, df):
    # assuming X and y are already defined
    model = LinearRegression().fit(df[['total_bill']], df[y] )
    df['yhat'] = model.predict(df[['total_bill']])
    
    df['yhat_baseline'] = df[y].mean()
    #Residuals
    df['residuals'] = df[y] - df['yhat']
     #baseline residual
    df['baseline_residuals'] = df[y] - df['yhat_baseline']
    #Visualize it
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.hist(tips_df['baseline_residuals'], label='baseline residuals', alpha=.6)
    ax.hist(tips_df['residuals'], label='model residuals', alpha=.6)
    ax.legend()
    return




def regression_errors(y, yhat, df):
    #sum of squared errors
    sse = (df.residuals ** 2).sum()
    #setting n value
    n= tips_df.shape[0]
    #means squared error
    mse= sse/n
    #root mean squared error
    rmse= math.sqrt(mse)
    #explained sum of squares and total sum of squares
    ess= ((df.yhat_baseline- df.tip.mean())**2).sum()
    tss= ((df.tip- df.tip.mean())**2).sum()
    return (f''' Regression Error: sse: {sse}, mse: {mse}, rmse: {rmse}, ess: {ess}, tss: {tss} ''')



def baseline_mean_errors(df):
    #setting baselines
    sse_baseline= (df.baseline_residuals ** 2).sum()
    mse_baseline = sse_baseline/ len(df)
    rmse_baseline= math.sqrt(mse_baseline)
    return(f'''Baseline: sse: {sse_baseline}, mse: {mse_baseline}, rmse: {rmse_baseline}''')




def better_than_baseline(df):
    sse_baseline = (df.baseline_residuals ** 2).sum()
    sse_model = (df.residuals ** 2).sum()
    return sse_model < sse_baseline




########################################## MPG DATA ##############################################################################

def residuals(actual, predicted):
    return actual - predicted

def sse(actual, predicted):
    return (residuals(actual, predicted) ** 2).sum()

def mse(actual, predicted):
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    return math.sqrt(mse(actual, predicted))

def ess(actual, predicted):
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    return ((actual - actual.mean()) ** 2).sum()

def r2_score(actual, predicted):
    return ess(actual, predicted) / tss(actual)





def regression_errors2(actual, predicted):
    return pd.Series({
        'sse': sse(actual, predicted),
        'ess': ess(actual, predicted),
        'tss': tss(actual),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
        'r2': r2_score(actual, predicted),
    })



def better_than_baseline2(df):
    sse_baseline = (df.baseline_residuals ** 2).sum()
    sse_model = (df.residuals ** 2).sum()
    return sse_model < sse_baseline




def better_than_baseline3(actual, predicted):
    sse_baseline = sse(actual, actual.mean())
    sse_model = sse(actual, predicted)
    return sse_model < sse_baseline