''' 
Apple Inc. 2019 , All rights reserved

Version - 0.1

DESCRIPTION:
-------------------------------------------------------
This file contains various utilities for the project
'''
import datetime
import time
import mysql.connector
import socket
import pickle
import os
import glob


DB_CONNECT_PROPERTIES = {
    'host': 'rn-act-lapp02.rno.apple.com',
    'port': 9797,
    'user':'acuser',
    'password':'acpassword',
    'database':'autocompute'
}
'''
DB_CONNECT_PROPERTIES = {
    'host': '192.168.1.42',
    'port': 3306,
    'user':'acuser',
    'password':'Exilant@123+',
    'database':'autocompute'
}
'''

STATISTICS_ID = {'CpuUtilization':10,'MemoryUsed':20, 'DiskWriteCount':31,'NetworkOpenConnections':40,'DBLockCount':50}
THROTTLE_SOCKET_PORT = 6767
LOG_SERVER_PORT = 9090
TIMEZONE_DELTA_MINS = 420 #We are simulating for PST now...
WEEKDAY_OFFSET = 2 #To sync between weekday constants of Python and MySQL
COUNT_POINTS_TO_PREDICT = 5 #we are collecting data once every 12 seconds. We predict for next 1 minute
MODEL_REFRESH_INTERVAL_SECONDS = 120
SECONDS_BETWEEN_POINTS = 12
PREDICTION_INTERVALS_SECONDS = SECONDS_BETWEEN_POINTS * COUNT_POINTS_TO_PREDICT #must be less than 1/3rd of model refresh interval
HOST_NAME_NONE = "__NONE__"
MACHINE_TIMEZONE_OFFSET_MINS = 420

THROTTLE_COMMANDS = {
    'FullSpeed' : 100,
    'HalfSpeed' : 200,
    'FullStop' : 300
}


progEpoch = time.time()
debugMode = False
logServer = None
'''
    Set the debug mode
'''
def setDebugMode(mode):
    global debugMode
    debugMode = mode
'''
    Just to print debug messages. Accepts a string input
'''
def debug(hostName, msg):
    global logServer
    if(debugMode) : 
        logMsg = "[" + datetime.datetime.now(datetime.timezone.utc).strftime("%d-%m-%Y-%H-%M-%S") + "] " + msg + "\r\n"
        print(logMsg)
        try:
            #Now remove oldest entries from debug table
            mysqlcon= connectToMySQL()
            cursor = mysqlcon.cursor()
            now = datetime.datetime.now(datetime.timezone.utc)
            oneHourAgo = now - datetime.timedelta(hours=1)
            ts = int(datetime.datetime.timestamp(oneHourAgo) * 1000)
            cursor.execute("DELETE FROM debug_logs WHERE dbg_timestamp < %s" , (ts,))
            #Now add new entries
            if( len(msg) > 1024 ):
                msg = msg[0:1024]
            #We cannot use UNIX_TIMESTAMP...as it is in seconds
            #here, the timestamp represents milliseconds
            milliseconds = int(round(time.time() * 1000))
            cursor.execute("INSERT INTO debug_logs(host_name,dbg_timestamp,dbg_msg) VALUES(%s,%s,%s)", (hostName, milliseconds, msg,))
            time.sleep(0.01)
            mysqlcon.commit()
            mysqlcon.close()
        except Exception as e:
            print(e)
            pass

'''
    Check if number of seconds have elapsed since last call or not
    If reset is marked as true, then epoch is reset
'''
def didSecondsElapsed(secs):
    global progEpoch
    diff = time.time() - progEpoch
    result = (diff >= secs)
    return result

'''
    Reset program epoch
'''
def resetProgEpoch():
    global progEpoch
    progEpoch = time.time()

'''
    The minimum date we want to consider for the historical data.
    Based on numerous experiments, we found that if we consider 
    past data for last 20 minutes, we have better accuracy of
    predicting the current trend
'''
def getHistoryStartDate(wks=0, dys=0,hrs=0,mins=0):
    #
    # We work with UTC
    #
    now = datetime.datetime.now(datetime.timezone.utc)
    minDate = now - datetime.timedelta( weeks=wks, days= dys, hours=hrs, minutes = mins )
    minDateAsUnixTimestamp = int(datetime.datetime.timestamp(minDate))
    
    return minDateAsUnixTimestamp
'''
    Generate a set of future timestamps against which we need to predict
'''
def generateFutureTimeseriesForPrediction(): 
    #
    # We work with UTC
    #
    future_dates=[]
    #TODO::now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=MACHINE_TIMEZONE_OFFSET_MINS)
    #From current UTC time, predict for next N minutes
    now = datetime.datetime.now(datetime.timezone.utc)
    pointToPredict = getPointsToPredict()
    for count in range(pointToPredict):
        future_dates.append(now + datetime.timedelta( seconds = SECONDS_BETWEEN_POINTS * (count + 1)))
    return future_dates

'''
    How many points to predict in future
'''
def getPointsToPredict():
    #We take one measure every 12 seconds...
    #Our history is past 20 minutes...
    #We predict for only next 1 minute
    #Yes harcoded for now because there are model dependencies...
    return COUNT_POINTS_TO_PREDICT
'''
    Connect to MySQL and return the connection object
'''
def connectToMySQL():
    return mysql.connector.connect(
            host = DB_CONNECT_PROPERTIES['host'],
            port = DB_CONNECT_PROPERTIES['port'],
            user = DB_CONNECT_PROPERTIES['user'],
            password = DB_CONNECT_PROPERTIES['password'],
            database = DB_CONNECT_PROPERTIES['database'],
            )
'''
    Get the list of machines currently 
    participating in this simulation
'''
def getListOfMachines():
    mysqlcon = connectToMySQL()
    cursor = mysqlcon.cursor()
    cursor.execute("SELECT host_name from machines")
    rows = cursor.fetchall()
    mysqlcon.close()
    return rows

'''
    Insert the final predictions to the database
'''
def insertFinalPredictions(hostName,sid,ts,predictions):
    mysqlcon = connectToMySQL()
    cursor = mysqlcon.cursor()

    #first delete any previous predictions that may be there 
    minTimestamp = int(datetime.datetime.timestamp(min(ts)) ) #remove past predicted
    delSql = """DELETE FROM machine_stats 
                WHERE host_name = %s 
                AND stat_type = %s    
                AND stat_timestamp >= %s  
                AND stat_id = %s"""
    delParams = (hostName , 2 , minTimestamp , sid)
    cursor.execute(delSql , delParams)
    #Now insert the new predictions
    sql = """INSERT INTO machine_stats (host_name,stat_id,
            stat_type,stat_timestamp,stat_value) VALUES (%s,%s,%s,%s,%s)"""
    max= min( len(ts) , len(predictions) )
    for counter in range(max):
        dt = ts[counter]
        #insert these predictions
        cursor = mysqlcon.cursor()
        unixTs = datetime.datetime.timestamp(dt)
        params = ( hostName , sid , 2, round(unixTs) , float(round (predictions[counter],2)) )
        try:
            cursor.execute( sql , params )
            counter += 1
        except:
            continue #TODO
    mysqlcon.commit()
    mysqlcon.close()

'''
    Insert the temporary predictions to the database
'''
def insertTemporaryPredictions(hostName,ptype,sid,ts,predictions):
    mysqlcon = connectToMySQL()
    cursor = mysqlcon.cursor()

    #first delete any previous predictions that may be there 
    minTimestamp = int(datetime.datetime.timestamp(min(ts)) ) #remove past predicted
    delSql = """DELETE FROM predicted_temp_data 
                WHERE host_name = %s 
                AND prediction_type = %s    
                AND stat_timestamp >= %s  
                AND stat_id = %s"""
    delParams = (hostName , ptype, minTimestamp , sid)
    cursor.execute(delSql , delParams)
    
    #now insert the new predictions
    sql = """INSERT INTO predicted_temp_data (host_name,prediction_type,
            stat_timestamp,stat_id,stat_value) VALUES (%s,%s,%s,%s,%s)"""
    counter=0
    for prediction in predictions:
        dt = ts[counter]
        unixTs = datetime.datetime.timestamp(dt)
        try:
            cursor.execute( sql , ( hostName , ptype , round(unixTs) , sid , float(round(prediction,2)) ) )
            counter += 1
        except:
            continue #we might get primary  key violations...TODO
    mysqlcon.commit()
    mysqlcon.close()

'''
    Get the temporary predictions from the database.
'''
def getTemporaryPredictions(hostName,ptype,sid , sinceTs , uptoTs):
    mysqlcon = connectToMySQL()
    sql = """SELECT stat_value 
             FROM predicted_temp_data
             WHERE host_name=%s 
             AND prediction_type=%s
             AND stat_timestamp BETWEEN %s AND %s  
             AND stat_id=%s"""
    cursor = mysqlcon.cursor()
    params = (hostName, ptype,sinceTs,uptoTs, sid )
    cursor.execute(sql,params)
    rows = cursor.fetchall()
    mysqlcon.close()
    if( rows != None ):
        return rows
    else:
        return [[]]

'''
Delete temporary predictions older than one hour
'''        
def deleteTemporaryPredictions():    
    mysqlcon = connectToMySQL()
    cursor = mysqlcon.cursor()

    #first delete any previous predictions that may be there 
    delTS = int(datetime.datetime.timestamp(datetime.datetime.utcnow()-datetime.timedelta(hours=1, minutes=0))) #remove past predicted
    delSql = """DELETE FROM predicted_temp_data 
                WHERE stat_timestamp <= %s"""
    cursor.execute(delSql , (delTS,))
    mysqlcon.commit()
    mysqlcon.close()
    
    
'''
    Get the final predictions from the database.
    These final predictions are output from GB regressor
'''
def getFinalPredictions(hostName,sid,sinceTs):
    mysqlcon = connectToMySQL()
    sql = """SELECT stat_value 
             FROM machine_stats 
             WHERE host_name=%s 
             AND stat_timestamp >= %s  
             AND stat_type=2 
             AND stat_id=%s"""
    cursor = mysqlcon.cursor()
    params = (hostName,sinceTs, sid )
    cursor.execute(sql,params)
    rows = cursor.fetchall()
    mysqlcon.close()
    if( rows != None ):
        return rows
    else:
        return [[]]
'''
    For the given hostname, get the actual stat value.
    An array is returned
'''
def getCollectedStat(hostName, statId, sinceTs):
    mySQLConn =connectToMySQL()
    cursor = mySQLConn.cursor()
    sql = """SELECT stat_timestamp,stat_value from machine_stats 
        where host_name=%s and stat_id=%s and stat_type = 1
        and stat_timestamp >=%s"""
    
    cursor.execute(sql,(hostName, statId, sinceTs ))
    rows = cursor.fetchall()
    mySQLConn.close()
    if( rows != None ):
        return rows
    else:
        return [[]]

'''
    Get the benchmark stats...
    For now, the benchmark is for all hosts..
    Benchmark data is stored as UTC. We need to account for
    the timezone in a special way. For example, if local time is 5pm, then
    we need to construct a date as '2000-01-01 17:00:00' , take a UNIX
    timestamp value of this and then query the database
'''
def getBenchmarkStats(sid, wd, sinceDtUtc):
    mySQLConn =connectToMySQL()
    cursor = mySQLConn.cursor()
    sql = """SELECT value from benchmark_data 
        where day_of_the_week=%s 
        and time_of_the_day >= %s
        and stat_id=%s"""
    #benchmark stats are stored with date as 2000-01-01...
    #We have to convert the benchmark stored as UTC to the current server local time
    adjustedLocalDt = sinceDtUtc - datetime.timedelta(minutes=MACHINE_TIMEZONE_OFFSET_MINS)
    baseDt = datetime.datetime(2000,1,1 , adjustedLocalDt.hour, adjustedLocalDt.minute , 0 ,0,datetime.timezone.utc)
    baseDtStamp = datetime.datetime.timestamp(baseDt)
    cursor.execute(sql,(wd, baseDtStamp, sid ))
    rows = cursor.fetchall()
    mySQLConn.close()
    if( rows != None ):
        return rows
    else:
        return [[]]
'''
    Update the state score in the database
'''
def updateHostCPUStateScore(hostName,ss):
    mySQLConn =connectToMySQL()
    cursor = mySQLConn.cursor()
    sql ="""UPDATE machines
            SET last_update_timestamp=%s 
            , cpu_ss= %s 
            WHERE host_name=%s"""
    updateTs = int(datetime.datetime.timestamp(datetime.datetime.utcnow()))
    state_score = float(round(ss,3))
    cursor.execute(sql,( updateTs , state_score , hostName ))
    mySQLConn.commit()
    mySQLConn.close()

'''
    Read the state score in the database
'''
def getHostCPUStateScore(hostName):
    mySQLConn =connectToMySQL()
    cursor = mySQLConn.cursor()
    sql ="""SELECT cpu_ss 
            FROM machines  
            WHERE host_name = %s"""
    cursor.execute(sql,( hostName, ))
    ss = cursor.fetchone()
    mySQLConn.close()
    return ss[0]
'''
    Send the throttle command to the given host
'''
def sendThrottleCommandToHost(hostName, tc):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ip_addr = hostName
        try:
            if(hostName.lower().index(".local") >= 0 or 
                hostName.lower().startswith("localhost")):
                ip_addr = socket.gethostbyname( "localhost" )
        except:
            pass
            
        server_address = (ip_addr, THROTTLE_SOCKET_PORT)
        sock.settimeout(10) #don't try connecting for more than 10 seconds
        sock.connect(server_address)
        sock.send(str(tc).encode())
        sock.settimeout(None)
        sock.close()
    except:
        debug(hostName,"Failed to send throttle command to host ")
'''
    Save the model file to disk
'''
def saveModelToDisk(hostName, model, prefix):
#save to disk as pickle file
    now = time.time()
    fileObject = open( str(now) + "." + hostName + "." + prefix + ".pickle" , 'wb' )
    pickle.dump(model,fileObject)
    fileObject.close()
    #also delete files older than one hour
    for f in os.listdir("./"):
        if os.stat(f).st_mtime < now - 60*60:
            if os.path.isfile(f) and f.endswith("." + prefix + ".pickle"):
                fullPath = os.path.join("./", f)
                os.remove(fullPath)
'''
    Load the model from disk. The prefix identifies the model
    type
'''
def loadLatestModelFromDisk(hostName, prefix):
    listOfFiles = glob.glob('./*' + "." + hostName + "." + prefix + ".pickle") 
    latestFile = max(listOfFiles, key=os.path.getctime)
    modelContents = None
    with open(latestFile, 'rb') as f:
        modelContents = f.read()
    
    model = pickle.loads(modelContents)
    
    return model