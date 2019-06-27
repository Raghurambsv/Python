
''' 
Apple Inc. 2019 , All rights reserved

Version - 0.1

DESCRIPTION:
-------------------------------------------------------
This file build the Gradient Booster model for prediction. The model
is updated every day at midnight. For each update, the last 3
year data is considered (We know we do not have last 3 year data
for this PoC, but we hope to reuse this same script for long time)
'''
import datetime
import time
import itertools
import mysql.connector
import socket
import pandas as pd
import numpy as np
import statsmodels
from sklearn import ensemble
from sklearn.metrics import r2_score
import warnings
import threading
import numpy as np

import ac_utils  

gbModels = {}

'''
    Build the model for the given hostname.
    For the CPU, we only focus on the CPU utilization %
'''
def buildCPUModelForHost(hostName):
    ac_utils.debug(hostName, "Building model for host=" + hostName)
    
    #GB will take ARIMA and RF predictions
    #as input and come up with best CPU utilization prediction.
    #we need to have historical records ...from now till all values in past
    since = ac_utils.getHistoryStartDate(mins=20)
    ac_utils.debug(hostName, "Minimum date for GB model development " +  datetime.datetime.fromtimestamp(since).isoformat())
    now = datetime.datetime.now(datetime.timezone.utc)
    upto = int(datetime.datetime.timestamp(now))

    arimaPredictions = ac_utils.getTemporaryPredictions( hostName , 1 , ac_utils.STATISTICS_ID['CpuUtilization'] , since , upto )
    rfPredictions = ac_utils.getTemporaryPredictions( hostName , 2 , ac_utils.STATISTICS_ID['CpuUtilization'] , since , upto)
    #also get the actual values (targets)
    actuals = ac_utils.getCollectedStat(hostName,ac_utils.STATISTICS_ID['CpuUtilization'], since )
    pCount = min( len(arimaPredictions) , len(rfPredictions) , len(actuals))
    
    if pCount > 0:
        #convert to pandas dataframe for ML
        df = buildDataFrame(arimaPredictions , rfPredictions , actuals , pCount)
        ac_utils.debug(hostName,"Found " + str(len(df))+ " records...")
        train_length = (int)(len(df) * 0.8)

        estimators = [100,150,200,250,300,500,700,1000]
        max_depths = [1,5,10]
        learning_rates = [0.1,0.01,0.001,0.005,0.008,1]
        #criterions = ['friedman_mse','mse','mae']
        #We will now attempt to optimize the model
        r2score = -9999999999
        model_params = [0,0,0]

        for estimator in estimators:
            for max_depth in max_depths:
                for learning_rate in learning_rates:
                    params = {
                                'n_estimators': estimator,
                                'max_depth': max_depth,
                                'learning_rate': learning_rate,
                                'criterion': 'friedman_mse'
                            }
                    gbX = ensemble.GradientBoostingRegressor(**params)
                    train = df[:train_length]
                    test = df[train_length:]
                    #gbX.fit( train[['AP' , 'RP' ]], np.ravel(train[['AC']]) )
                    gbX.fit( train[['AP' , 'RP' ]], np.ravel( train[['AC']]) )
                    predictions = gbX.predict( test[['AP' , 'RP' ]] )
                    currR2score = r2_score(test.iloc[:,2].values,predictions)
                    if currR2score > r2score:
                        r2score = currR2score
                        model_params = [ estimator , max_depth , learning_rate ]
                    ac_utils.debug(hostName,
                    "r2score={}, estimators={}, max_depth={}, learning_rate={}"
                    .format(currR2score,estimator,max_depth,learning_rate))
        ac_utils.debug(hostName,"Selected params: r2score={}, estimators={}, max_depth={}, learning_rate={}"
                    .format(r2score,model_params[0],model_params[1],model_params[2]))

        #Gradient booster is trained with CPU utilization
        #values 
        gbX.fit( df[['AP' , 'RP' ]], np.ravel(df[['AC']]))
        saveGBModel(hostName,gbX)
    else:
        ac_utils.debug(hostName, "No data found...skipping...")

'''
    For the give set of arima predictions and random forest predictions,
    create a pandas dataframe
'''
def buildDataFrame(ap,rp,av,count):
    cols = ['AP','RP', 'AC']
    dfRows = []
    
    for r in range(count):
        dfRow=[ ap[r][0] , rp[r][0], av[r][1] ]
        dfRows.append(dfRow)
    
    df = pd.DataFrame(dfRows,columns=cols)
    #df = df.set_index('Seq')
    
    return df

'''
    Save the Gradient Boosting model.
    Will be used during prediction...
'''
def saveGBModel(hostName,gbModelFit):
    ac_utils.debug(hostName, "Updated GB model...")
    #gbModels[ hostName ] = gbModelFit
    ac_utils.saveModelToDisk(hostName,gbModelFit,"gb")

'''
    Get the Gradient Bossting model stored locally
'''
def getGBModel(hostName):
    model = None
    try:
        model = ac_utils.loadLatestModelFromDisk(hostName,"gb")
    except:
        pass
    return model