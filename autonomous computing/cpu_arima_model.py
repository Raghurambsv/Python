
''' 
Apple Inc. 2019 , All rights reserved

Version - 0.1

DESCRIPTION:
-------------------------------------------------------
This file build the ARIMA model for prediction. The model
is updated every day at midnight. For each update, the last 3
year data is considered (We know we do not have last 3 year data
for this PoC, but we hope to reuse this same script for long time)
'''
import datetime
import time
import math
import itertools
import mysql.connector
import socket
import pandas as pd
import numpy as np
import statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import warnings
import threading
import pickle
import glob
import os

import ac_utils  

arimaModels = {}

'''
    Build the model for the given hostname.
    For the CPU, we only focus on the CPU utilization %
'''
def buildCPUModelForHost(hostName):
    ac_utils.debug(hostName,"Building model for host=" + hostName)
    #We have two variables; day-of-week and time-of-day for the CPU
    #we need to have historical records ...from now till N minutes in past
    minDate = ac_utils.getHistoryStartDate(mins=20)
    ac_utils.debug(hostName, "Minimum date for ARIMA model development " + datetime.datetime.fromtimestamp(minDate).isoformat())
    rows = ac_utils.getCollectedStat( hostName , ac_utils.STATISTICS_ID['CpuUtilization'] , minDate )
    if len(rows) > 0:
        #convert to pandas dataframe for ML
        df = buildDataFrame(rows)
        ac_utils.debug(hostName,"Found " + str(df.shape)+ " historical records...")
        #check if data is stationary...else make it
        ac_utils.debug(hostName,"attempting to make data stationary...")
        diff = 0
        #Split in train and test
        train = df.iloc[:,0].values
        train_copy = train.copy() #use this copy to find PDQ
        
        while isDataStationary( train_copy ) == False:
            #attempt to make stationary
            train_copy = np.diff(train_copy,n=1)
            train_copy = train_copy[1:]
            time.sleep(0.1)
            diff += 1
            if ( diff > 2 ): #library does not support lags greater than 2
                diff = 2
                break
        
        #now determine the best P/D/Q values
        ac_utils.debug(hostName,"Attempting to find best p/d/q combination...with d=" + str(diff))
        
        p=d=q=range(0,9) #beyond 0-8 range, the system hangs frequently
        pdq=list(itertools.product(p,d,q))
        pdqSelected = getBestPDQ(hostName, train,pdq, diff)
        ac_utils.debug(hostName,"PDQ selected = " + str(pdqSelected))
        saveARIMAModel(hostName,train,pdqSelected)
    else:
        ac_utils.debug(hostName,"No data found...skipping...")

'''
    For the give set of rows, retrieved from database,
    create a pandas dataframe
'''
def buildDataFrame(rows):
    cols = ['DateTime','Value']
    dfRows = []
    for r in rows:
        dfRow=[datetime.datetime.fromtimestamp(r[0]),r[1]]
        dfRows.append(dfRow)
    
    df = pd.DataFrame(dfRows,columns=cols)
    df = df.set_index('DateTime')
    #debug(str(df.dtypes) + " " + str(df.index))
    #We do not have a missing value problem here....so no data cleaning
    return df

'''
    For ARIMA to work, we need stationary dataset with no trends
    We ensure, data is stationary by using the Augmented Dickey-Fuller test
'''
def isDataStationary(arr):
    adfTestResult = adfuller(arr, autolag='AIC')
    
    dfoutput = pd.Series(adfTestResult[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print(dfoutput)
    #
    # If p-Value is less than 0.05 (or 5%) we assume data is stationary 
    #
    isStationary= ( adfTestResult[1] <  0.05)
    return isStationary
'''
    Find out the best P,D,Q combination based on the AIC value
'''
def getBestPDQ(hostName, arr, pdq, diff):
    min_aic=9999999999
    param_sel = []
    model_arima_fit = None
    for param in pdq:
        try:
            if(diff >= 0 and param[1] != diff): continue #This option not valid 
            model_arima = ARIMA(arr,order=param)
            model_arima_fit = model_arima.fit(disp=0)
            if(min_aic > model_arima_fit.aic):
                min_aic = model_arima_fit.aic
                param_sel=param
            ac_utils.debug(hostName, "Checked p,d,q " + str(param)  + ", got aic=" + str(model_arima_fit.aic))
        except:
            pass
    
    return param_sel

'''
    Save the ARIMA model.
    Will be used during prediction...
'''
def saveARIMAModel(hostName,arr,pdq):
    arimaModel = ARIMA(arr,order=pdq)
    arimaModelFit = arimaModel.fit()
    ac_utils.debug(hostName, "Updated ARIMA model...")
    #arimaModels[ hostName ] = arimaModelFit
    ac_utils.saveModelToDisk(hostName,arimaModelFit,"arima")
'''
    Get the ARIMA model stored locally.
'''
def getARIMAModel(hostName):
    model = None
    try:
        model = ac_utils.loadLatestModelFromDisk(hostName,"arima")
    except:
        pass
    return model