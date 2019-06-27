''' 
Apple Inc. 2019 , All rights reserved

Version - 0.1

DESCRIPTION:
-------------------------------------------------------
Predict CPU using ARIMA, RandomForest and GradienBoosting
After that, compute the stat score and issue throttle
commands
'''
import pandas as pd
import numpy as np
import cpu_arima_model as cam
import cpu_rf_model as crm
import cpu_gb_model as cgm
import system_state_score_computer as ssc
import ac_utils
import time
import datetime
import math
import multiprocessing as mp

'''
    The main method
'''
def main():   
    ac_utils.setDebugMode(True)
    ac_utils.debug(ac_utils.HOST_NAME_NONE,"Starting prediction engine....")
    #Open another process to build models
    modelBuilderProcess = mp.Process(target=buildModelProc, args=(0,))
    modelBuilderProcess.start()

    #Now keep predicting
    while(True):
        try:
            #Generate future timestamps for prediction
            future_timestamps = ac_utils.generateFutureTimeseriesForPrediction()
            predictCPUUtilUsingARIMA(future_timestamps)
            predictCPUUtilUsingRF(future_timestamps)
            #Next using gradient booster, take the ARIMA and 
            #RF predictions as input and make predictions
            predictCPUUtilizationUsingGB(future_timestamps)
            ## Now compute state score
            computeCPUSystemStateScore()
            ## And issue throttle commands
            issueThrottleCommandsOnCPUState()    
        except Exception as e:
            ac_utils.debug(ac_utils.HOST_NAME_NONE,"Failed to execute\r\n" + str(e))
            pass
        time.sleep( ac_utils.PREDICTION_INTERVALS_SECONDS ) 

    modelBuilderProcess.join()

'''
    Function to build and update models in parallel
'''
def buildModelProc(args):
    while(True):
        try:
            #update ARIMA model
            updateARIMAModels()
            #update RF model
            updateRFModels()
            #update GB model
            updateGBModels()
            
            ac_utils.deleteTemporaryPredictions()
        except Exception as e:
            ac_utils.debug(ac_utils.HOST_NAME_NONE , str(e))
            pass
        time.sleep(ac_utils.MODEL_REFRESH_INTERVAL_SECONDS)
'''
    This updates the ARIMA model
'''
def updateARIMAModels():
    #Get all machines
    hosts = ac_utils.getListOfMachines()
    for m in hosts:
        ac_utils.debug(m[0],"Updating ARIMA model for host ")
        cam.buildCPUModelForHost( m[0] )

'''
    This updateds the RF model
'''
def updateRFModels():
    #Get all machines
    hosts = ac_utils.getListOfMachines()
    for m in hosts:
        ac_utils.debug(m[0],"Updating RF model for host ")
        crm.buildCPUModelForHost( m[0] )

'''
    This updates the Gradient Booster model
'''
def updateGBModels():
    #Get all machines
    hosts = ac_utils.getListOfMachines()
    for m in hosts:
        ac_utils.debug(m[0],"Updating GB model for host ")
        cgm.buildCPUModelForHost( m[0] )

'''
    Predict the CPU utilization using ARIMA.
''' 
def predictCPUUtilUsingARIMA(future_datetimes):
    hosts = ac_utils.getListOfMachines()
    for m in hosts:
        model = cam.getARIMAModel( m[0] )
        if( model != None ):
            ac_utils.debug(m[0],"Predicting using ARIMA for host " )
            arimaPredictions = model.forecast( steps= len(future_datetimes))[0]
            ac_utils.debug(m[0],"Predicted " + str(len(arimaPredictions)) + " values using ARIMA...")
            #clip values for CPU between 0-100
            arimaPredictions =arimaPredictions.clip(0,100)
            ac_utils.debug(m[0],"ARIMA Points=" + str(arimaPredictions))
            ac_utils.insertTemporaryPredictions( m[0] , 1 , ac_utils.STATISTICS_ID['CpuUtilization'], future_datetimes , arimaPredictions )

'''
    Predict CPU utilization using random forest
'''
def predictCPUUtilUsingRF(future_datetimes):
    hosts = ac_utils.getListOfMachines()
    #Build feature list for prediction
    cols = ['DateTime', 'WeekDay', 'TimeOfDay','CPU', 'NWOC', 'DW', 'MU', 'DBL']
    
    feature_names=['NWOC', 'DW', 'MU']
    #label_name = ['CPU']
    
    for m in hosts:
        model = crm.getRFModel( m[0] )
        if(model != None):
            ac_utils.debug(m[0],"Predicting using RF for host ")
            dataWindow = ac_utils.getHistoryStartDate(mins=20)

            #populate open network connections
            nwStats = ac_utils.getCollectedStat( m[0] , ac_utils.STATISTICS_ID['NetworkOpenConnections'] ,dataWindow )
            #populate disk writes
            dwStats = ac_utils.getCollectedStat( m[0] , ac_utils.STATISTICS_ID['DiskWriteCount'] , dataWindow )
            #populate memory utilization
            muStats = ac_utils.getCollectedStat( m[0] , ac_utils.STATISTICS_ID['MemoryUsed'] , dataWindow )
            
            length = min( len(nwStats) , len(dwStats), len(muStats) , len(future_datetimes) )

            if( length > 0 ):
                #create a dataframe of this length
                #
                # We have the model built using historical values....
                # Now, take the future timestamps and take immdiate
                # values and try to run through the model
                #
                dfRows = []
                for count in range(length):
                    timeOfDay = future_datetimes[count].time()
                    dfRow=[ future_datetimes[count], 
                            future_datetimes[count].weekday() + ac_utils.WEEKDAY_OFFSET, 
                            timeOfDay.hour * 3600 + timeOfDay.minute * 60 + timeOfDay.second,
                            0,
                            nwStats[count][1],
                            dwStats[count][1],
                            muStats[count][1],
                            0
                            ]
                    dfRows.append(dfRow)
                
                df = pd.DataFrame(dfRows,columns=cols)
                df = df.set_index('DateTime')

                features = df[feature_names]

                rfPredictions = model.predict(features)
                ac_utils.debug(m[0], "Predicted " + str(len(rfPredictions)) + " values using RF...")
                #Clip values for CPU between 0-100
                rfPredictions = rfPredictions.clip(0,100)
                ac_utils.debug(m[0],"RF Points=" + str(rfPredictions))
                ac_utils.insertTemporaryPredictions( m[0] , 2 , ac_utils.STATISTICS_ID['CpuUtilization'], future_datetimes , rfPredictions )
'''
    Predict CPU Utilization using the gradient booster
    algorithm
'''
def predictCPUUtilizationUsingGB(future_datetimes):
    hosts = ac_utils.getListOfMachines()
    #
    # We need to get the latest ARIMA and RF predictions for the next N minutes
    #
    #TODO::now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=ac_utils.MACHINE_TIMEZONE_OFFSET_MINS)
    now = datetime.datetime.now(datetime.timezone.utc)
    since = int(datetime.datetime.timestamp(now))
    upto = int(datetime.datetime.timestamp(max(future_datetimes)))
    for m in hosts:
        model = cgm.getGBModel( m[0] )
        if(model != None):
            ac_utils.debug(m[0],"Predicting using GB for host ")
            #first get the temporary predictions using ARIMA
            arimaPredictions = ac_utils.getTemporaryPredictions( m[0] , 1 , ac_utils.STATISTICS_ID['CpuUtilization'] , since , upto)
            #next, get the temporary predictions using RF
            rfPredictions = ac_utils.getTemporaryPredictions( m[0] , 2 , ac_utils.STATISTICS_ID['CpuUtilization'] , since ,upto)
            #build dataframe
            length = min(len(arimaPredictions), len(rfPredictions))
            if(length > 0):
                cols = ['AP','RP', 'AC']
                dfRows = []
                for count in range(length):
                    dfRow=[ arimaPredictions[count][0] , rfPredictions[count][0], 0.00  ]
                    dfRows.append(dfRow)
                df = pd.DataFrame(dfRows,columns=cols)
                
                gbxPredictions = model.predict( df[['AP' , 'RP' ]] )
                ac_utils.debug(m[0],"Predicted " + str(len(gbxPredictions)) + " values using GB...")
                #Clip values for CPU between 0-100
                gbxPredictions = gbxPredictions.clip(0,100)
                ac_utils.debug(m[0],"GB Points=" + str(gbxPredictions))
                #if change from one value to another is over 20%, then smoothen those 
                #values
                smoothenedValues = []
                lastValue= -1
                for cpu in gbxPredictions:
                    if( lastValue > 0 ): #next time
                        if( abs(lastValue - cpu) > 20.0 ):
                            cpu = cpu/2
                    lastValue = cpu
                    smoothenedValues.append( cpu )
                ac_utils.insertFinalPredictions(m[0], ac_utils.STATISTICS_ID['CpuUtilization'] , future_datetimes, smoothenedValues)
            else:
                ac_utils.debug(m[0],"Failed to do predictions using GBX...no current predictions found")

'''
    Compute the system state score
    with respect to the CPU
'''
def computeCPUSystemStateScore():
    #Get all predictions from now till future
    #now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=ac_utils.MACHINE_TIMEZONE_OFFSET_MINS)
    now = datetime.datetime.now(datetime.timezone.utc)
    #
    # In java 1=MONDAY ... 7=SUNDAY
    # In MySQL 1=SUNDAY ... 7=SATURDAY
    # In Python 0=MONDAY .... 6=SUNDAY
    # We use MySQL definition...
    #
    dayOfWeek = now.weekday() + ac_utils.WEEKDAY_OFFSET 
    if(dayOfWeek > 7): dayOfWeek=1 #rollover
    benchmarkStats = ac_utils.getBenchmarkStats( ac_utils.STATISTICS_ID['CpuUtilization'] , dayOfWeek , now )
    hosts = ac_utils.getListOfMachines()
    for m in hosts:
        predictedStats = ac_utils.getFinalPredictions( m[0] , ac_utils.STATISTICS_ID['CpuUtilization'] , int(datetime.datetime.timestamp(now) ) )
        statsToConsider= min( len(benchmarkStats) , len(predictedStats) )
        if statsToConsider == 0: continue
        #compute state score
        predictedStats = predictedStats[:statsToConsider]
        tmpBenchmarkStats = benchmarkStats[:statsToConsider]
        state_score = ssc.computeNormalizedStateScore( np.ravel(predictedStats) , np.ravel(tmpBenchmarkStats) )
        ac_utils.debug(m[0],"State score for host " + m[0] + " is " + str(state_score))
        ac_utils.updateHostCPUStateScore(m[0] , state_score)

'''
    Issue the throttle command based on system state score
    of the CPU
'''
def issueThrottleCommandsOnCPUState():
    hosts = ac_utils.getListOfMachines()
    for m in hosts:
        state_score = float(ac_utils.getHostCPUStateScore(m[0]))
        ##
        ## We need to experiment and see what is a good range
        ##
        throttleCmd = ac_utils.THROTTLE_COMMANDS['FullSpeed']
        if( state_score <= 0.25 ):
            #system cool, issue throttle command to process more
            throttleCmd = ac_utils.THROTTLE_COMMANDS['FullSpeed']
        elif(state_score > 0.25 and state_score <= 0.75):
            #heating up, attempt to cool down
            throttleCmd = ac_utils.THROTTLE_COMMANDS['HalfSpeed']
        else:
            #system hot, issue command to stop processing requests
            throttleCmd = ac_utils.THROTTLE_COMMANDS['FullStop']
        
        ac_utils.debug(m[0],"Throttle command for host " + m[0] + " is " + str(throttleCmd))
        ac_utils.sendThrottleCommandToHost(m[0] , throttleCmd)

if __name__ == "__main__" : main()
