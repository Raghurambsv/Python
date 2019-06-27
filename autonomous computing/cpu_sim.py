''' 
Apple Inc. 2019 , All rights reserved

DESCRIPTION:
-------------------------------------------------------
This file simulates server CPU usage for a typical day. Day starts at midnight and ends at 11:59:59 pm.
A typical day will see usage increase after 9 am and peak at about 12 noon. Then it dips a bit till
2 pm and then again picks up until 4 pm and then goes down, settling at the minimum level at about 6:00pm

This file simulates 3 conditions 
1) Low usage for entire day (e.g. weekends)
2) Medium usage for entire day (e.g. normal business days)
3) High usage for entire day (e.g. special days)

Low/Medium/High means the usage pattern remains the same, but net value at each instance is high.
For example, for 9 am, low usage could be 30%, medium usage could be 50% and high usage could be 70%.

This file simulates the conditions to increase/decrease CPU usage and writes the values for in a file
as a comma separated value.
'''

import sys
import datetime
import psutil
import time
import threading
import math
import multiprocessing
import random
from multiprocessing import Pool
from multiprocessing import Process

import os
import subprocess


LOW_SYMBOL = 'L'
MEDIUM_SYMBOL = 'M'
HIGH_SYMBOL = 'H'

_lastCpuUtilization=0.00

##
## This table gives the guidance on how to simulate the CPU % based on time
## of day. Modify this table to get desired simulation. The key represents 
## hour of the day and the value represents the maximum CPU utlization that
## is allowed for the hour of the day. 
## These values represent medium utilization. To simulate low utilization
## these values are reduced by 20% whereas to simulate high utilization,
## these values are increased by 50%.
## On saturday and sunday, these values are reduced further
##
LOAD_PERCENTAGES = {
    0 : 10.00, 1 : 10.00, 2 : 10.00, 3 : 10.00, 4 : 10.00, 5 : 10.00, 6 : 10.00,
    7 : 10.00, 8 : 15.00, 9 : 25.00, 10: 35.00, 11: 50.00, 12: 50.00, 13: 25.00,
    14: 50.00, 15: 50.00, 16: 50.00, 17: 30.00, 18: 20.00, 19: 10.00, 20: 10.00,
    21: 10.00, 22: 10.00, 23: 10.00 
}

lastCommand = None

'''
The main method. Entry point. The only argument it needs
is the type of simulation to perform (L/M/H)
'''
def main():

    if(len(sys.argv)) < 2:
        print("Error : Command line argument missing. Must provide the type of simulaton to perform")
        print("Usage: python cpu_sim.py [L/M/H] <recording interval in seconds, optional, default 30>")
    elif(sys.argv[1] != "L" and sys.argv[1] !="M" and sys.argv[1] != "H"):
        print("Error : Invalid command line argument. Must be L/M/H")
    else:
        if(len(sys.argv)>=3):
            logIntervalSecs = int(sys.argv[2])  
        else:
            logIntervalSecs = 30
        debug("Starting CPU simulation type - " + sys.argv[1] + " , log interval=" + str(logIntervalSecs) +
            ", CPU cores=" + str(multiprocessing.cpu_count()))
        
        file = open(getLogFilename(),"w+")
        #start logger...
        loggerThread = threading.Thread(target=logStatisticsThreadFunc, args=(file, logIntervalSecs), daemon=True)
        loggerThread.start()
        prevLoad = 0
        global lastCommand
        lastCommand = None
        while(True):
            try:
            	curLoad = cpuLoadToSimulate(sys.argv[1])
                
            	if(curLoad != prevLoad) :
            		if(lastCommand is not None) :
            			lastProcess = psutil.Process(lastCommand.pid)
            			for proc in lastProcess.children(recursive=False):
            				proc.kill()
            			lastProcess.kill()

            		prevLoad = curLoad
            		cmd = "stress-ng  -c " + str(psutil.cpu_count()) + " -l " + str(curLoad)
            		lastCommand = subprocess.Popen(cmd, shell=True)
            	time.sleep(logIntervalSecs/3)
            except Exception as e:
                debug( str(e) )
        
'''
Statistics logger thread
'''
def logStatisticsThreadFunc(fileObj,logIntervalSecs):
    beginTime = 0
    global _lastCpuUtilization
    global lastCommand
    try:
        while(True):
            elapsedTime = time.time() - beginTime
            if(elapsedTime > logIntervalSecs):
                _lastCpuUtilization=psutil.cpu_percent(interval=logIntervalSecs/2)
                recordLine=datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "," + str(_lastCpuUtilization) + "\r\n"
                fileObj.write(recordLine)
                fileObj.flush()
                debug("Logged CPU Utilization " + str(_lastCpuUtilization) + "%")
                beginTime = time.time() #reset start time
            time.sleep(logIntervalSecs/2)
    except Exception as e:
        debug( str(e) )        

'''
Get the new log file name
'''
def getLogFilename():
    return "cpu_usage_log_" + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M") + ".csv"
    
'''
This method determines the Cpu load for the given time of the day
It takes into consideration the type of simulation
'''
def cpuLoadToSimulate(stype):
    maxLoad=0.0
    now = datetime.datetime.now()
    hourOfDay = now.hour
    minuteOfHour = now.minute
    maxLoad = LOAD_PERCENTAGES.get(hourOfDay)

    #
    # We need clean ramp-up and ramp-down...
    #
    if( hourOfDay == 23 ):
        nextHour = 0
    else:
        nextHour = hourOfDay + 1
    
    nextLoad = LOAD_PERCENTAGES.get(nextHour)
    #compute slope
    slope = (nextLoad - maxLoad)/1 #divided by 1 to indicate 1 hour period...just readability
    #compute next load value...we simulate linear increase/decrease
    nextLoad = (slope * minuteOfHour/60) + maxLoad
    
    if(nextLoad < 10): 
        nextLoad = 10.00 #Lower limit
    elif(nextLoad > 75.00):
        nextLoad = 75.00 #upper limit

    if(stype == LOW_SYMBOL):
        #Reduce load'
        nextLoad *= 0.8
    elif (stype == MEDIUM_SYMBOL):
        #Keep load as defined above'
        nextLoad *= 1.0
    else:
        #increase load by 50%'
        nextLoad *= 1.5 
    
    #Take into consideration weekdays too'
    dayOfWeek = datetime.datetime.now().weekday()
    if(dayOfWeek == 5):
        #Saturday...reduce load to 50% of what we want'
        nextLoad *= 0.5
    elif(dayOfWeek == 6):
        #Sunday...reduce load to 10% of what we want'
        nextLoad *= 0.1
    
    return nextLoad


'''
    Just to print debug messages. Accepts a string input
'''
def debug(msg):
    print("[" + datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + "] " + msg)
    
if __name__ == "__main__" : main()
