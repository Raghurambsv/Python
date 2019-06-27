''' 
Apple Inc. 2019 , All rights reserved

Version - 0.1

DESCRIPTION:
-------------------------------------------------------
This file builds the RandomForest model for prediction. The model
is updated every day at midnight. For each update, the last 3
year data is considered (We know we do not have last 3 year data
for this PoC, but we hope to reuse this same script for long time)
'''
import time
import datetime
import ac_utils
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn import preprocessing

rfModels = {}
'''
    Build the model for the given hostname.
    For the CPU, we only focus on the CPU utilization %
'''
def buildCPUModelForHost(hostName):
    ac_utils.debug(hostName,"Building model for host=" + hostName )
    #For RF model, we need cpu utilization, n/w open connection,
    #disk writes, memory utilization, database locks
    #we need to have historical records ...from now till N minutes in past
    minDate = ac_utils.getHistoryStartDate(mins=20) #past 20 mins
    ac_utils.debug(hostName, "Minimum date for RF model development " + datetime.datetime.fromtimestamp(minDate).isoformat())
    #populate cpu utilization
    cpuUtilz = ac_utils.getCollectedStat( hostName , ac_utils.STATISTICS_ID['CpuUtilization'] , minDate )
    #populate open network connections
    nwCons = ac_utils.getCollectedStat( hostName , ac_utils.STATISTICS_ID['NetworkOpenConnections'] , minDate )
    #populate disk writes
    dwCnts = ac_utils.getCollectedStat( hostName , ac_utils.STATISTICS_ID['DiskWriteCount'] , minDate )
    #populate memory utilization
    muCnts = ac_utils.getCollectedStat( hostName , ac_utils.STATISTICS_ID['MemoryUsed'] , minDate )
    #populate database locks (Future)
    dlCnts = ac_utils.getCollectedStat( hostName , ac_utils.STATISTICS_ID['DBLockCount'] , minDate )
    
    if len(cpuUtilz) > 0: #We will always have same number of actuals for a given timeframe
        #convert to pandas dataframe for ML
        df = buildDataFrame(hostName,cpuUtilz,nwCons,dwCnts,muCnts,dlCnts)
        #now, separate the feature with labels
        #FUTURE feature_names=['WeekDay', 'TimeOfDay', 'NWOC', 'DW', 'MU', 'DBL']
        feature_names=['NWOC', 'DW', 'MU']
        label_name = ['CPU']
        features = df[feature_names]
        labels = df[label_name]
        #
        # After much experiments, we find that these features have different scales
        # Due to this, the RF regressor is 'confused'. Thus, we scale all the features
        # to the range 0-1. We leave the label as is because that is output
        # We tested with MinMax and Z-score scaler. Both perform the same
        #
        scaler = MinMaxScaler()
        df[feature_names]= scaler.fit_transform(df[feature_names])

        #split data into train/test
        features_train , features_test , labels_train, labels_test = train_test_split(features,labels,test_size=0.2)
        ac_utils.debug(hostName,"Shape of training features=" + str(features_train.shape))
        ac_utils.debug(hostName,"Shape of training labels=" + str(labels_train.shape))
        #Create the regressor 
        r2score = 0
        attempts = 3
        estimators= 0
        currEstimators = 100
        while r2score < 0.50 and attempts > 0:
            rfRegressor = RandomForestRegressor(n_estimators=currEstimators)
            #now train the model
            rfRegressor.fit(features_train,labels_train)
            #test how good are predictions
            predictions = rfRegressor.predict(features_test)
            currR2score = r2_score(labels_test,predictions)
            if currR2score > r2score:
                r2score = currR2score
                estimators = currEstimators
            ac_utils.debug(hostName,"RF model optimization: Estimators=" + str(currEstimators) + ",r2_score=" + str(currR2score))
            attempts = attempts - 1
            currEstimators = currEstimators * 3
        if(estimators == 0) :
            estimators=100 #found no possible combination that was good
        #build model with best fit
        ac_utils.debug(hostName,"RF model selected: Estimators=" + str(estimators) +",r2_score=" + str(r2score))
        rfRegressor = RandomForestRegressor(n_estimators=estimators)
        #now train the model
        rfRegressor.fit(features_train,labels_train)

        saveRFModel(hostName , rfRegressor)
    else:
        ac_utils.debug(hostName,"No data found...skipping...")

'''
    For the given set of rows, retrieved from database,
    create a pandas dataframe. This dataframe 
    uses features as n/w open connections, disk write counts,
    memory usage and db locks
'''
def buildDataFrame(hostName,cpuRows,nwRows,dwRows,muRows,dlRows):
    #make row count equal
    rowCount = min(len(cpuRows) , len(nwRows), len(dwRows), len(muRows), len(dlRows) )
    
    cols = ['DateTime', 'WeekDay', 'TimeOfDay','CPU', 'NWOC', 'DW', 'MU', 'DBL']
    dfRows = []
    
    for r in range(rowCount):
        dt = datetime.datetime.fromtimestamp(cpuRows[r][0])
        timeOfStat= dt.time()
        
        dfRow=[ dt, 
                dt.weekday() + ac_utils.WEEKDAY_OFFSET, 
                timeOfStat.hour * 3600 + timeOfStat.minute * 60 + timeOfStat.second, 
                cpuRows[r][1],
                nwRows[r][1],
                dwRows[r][1],
                muRows[r][1],
                dlRows[r][1]
                ]
        dfRows.append(dfRow)
    ac_utils.debug(hostName,"Found " + str(len(dfRows))+ " historical records...")
    df = pd.DataFrame(dfRows,columns=cols)
    df = df.set_index('DateTime')
    #TODO::Check for data cleaning
    return df


'''
    Save the RF model.
    Will be used during prediction...
'''
def saveRFModel(hostName,classifier):
    ac_utils.debug(hostName, "Updated RF model...")
    #rfModels[ hostName ] = classifier
    ac_utils.saveModelToDisk(hostName,classifier,"rf")
'''
    Get the RF model stored locally
'''
def getRFModel(hostName):
    model = None
    try:
        model = ac_utils.loadLatestModelFromDisk(hostName,"rf")
    except:
        pass
    return model