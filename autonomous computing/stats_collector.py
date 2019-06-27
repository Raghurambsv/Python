''' 
Apple Inc. 2019 , All rights reserved

DESCRIPTION:
-------------------------------------------------------
This file simulates collects machine health statistics and
sends it to a collector

You need admin access to collect the statistics
This can collect statistics for database also. 
Change the configuration parameters to define behavior
'''

import psutil
import time
import socket
import subprocess
import datetime

DEBUG = True
REMOTE_HOST = 'localhost'
REMOTE_PORT = 7777
PORT_TO_SCAN= 8888
THIS_HOST = socket.gethostname()
COLLECTION_INTERVAL_SECS=10
COLLECT_DB_STATS = True
LOG_TO_FILE = True

#Populate below if you need to collect database stats
DB_USER='acuser'
DB_PWD='acpassword'

logFileHandle = None
prev_disk_write_count = None

#----------------------------------
# Method to collect machine 
# health statistics
#----------------------------------
def collect_machine_stats():
    global prev_disk_write_count
    stat_str = THIS_HOST
    #Get CPU statistics
    cpu_utilization_percent = psutil.cpu_percent(interval=2)
    #construct CPU stat string
    stat_str += "|" + str(cpu_utilization_percent)

    #Get Memory statistics
    memory_used = psutil.virtual_memory()[2]
    #construct Memory stat string
    stat_str +=  "|" + str(memory_used)

    #Get disk statistics
    disk_write_count=psutil.disk_io_counters()[1] #currently getting disk write count
    if prev_disk_write_count is None :
    	prev_disk_write_count = disk_write_count
    delta_disk_write_count = disk_write_count - prev_disk_write_count
    prev_disk_write_count = disk_write_count

    #Construct disk stat string
    stat_str += "|" + str(delta_disk_write_count) 

    #Get network statistics...
    #We want to know the number of open connections for socket in question
    cmd = subprocess.Popen("netstat -an | grep :*" + str(PORT_TO_SCAN) + " | wc -l", shell=True, stdout=subprocess.PIPE)
    net_connections = str(cmd.stdout.readline(),"utf-8").strip()
    cmd.kill()
    
    if(DEBUG):
        print("cpu %=" + str(cpu_utilization_percent) +",memory used %="+ str(memory_used) 
            + ",disk writes=" + str(disk_write_count) + ",n/w open conn=" + str(net_connections))
        
    #Construct network stat string
    stat_str += "|" + str(net_connections)

    return stat_str
#-------------------------------------------
# Collect database health statistics.
# For this simulation, uses Mysql tools
# to get just the "locks taken parameter"
#-------------------------------------------
def collect_database_stats():
    dbStat="0"
    try:
        cmd = subprocess.Popen('mysqladmin --user='+ DB_USER +' --password=' + DB_PWD +'extended-status', shell=True, stdout=subprocess.PIPE)
        for line in cmd.stdout:
            if b"Com_lock_tables" in line:
                parts =line.split(b'|')
                dbStat = str(parts[2],'utf-8')
                break
        print("DB Lock Count=" + dbStat)
        return dbStat
    except Exception as e:
        print(e)
#------------------------------------
# Method to send health statistics
# to remote server
#------------------------------------
def send_stats(stat):
    
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.settimeout(5.0) #5 second timeout
            s.connect((REMOTE_HOST,REMOTE_PORT))
            s.sendall((stat + '\n').encode(encoding="utf-8", errors="strict") )
            s.close()
    except Exception as e:
        print("***Failed to connect to remote host")
        print(e)

#Collect stats and send to remote
while True:
    try:
        stat_str_collected = collect_machine_stats()
        if( COLLECT_DB_STATS == True ):
            stat_str_collected += "|" + collect_database_stats()
        else:
            stat_str_collected += "|-1" #ignore
        if( LOG_TO_FILE == True ): #we write to a pipe separated file...easy to reuse
            if( logFileHandle == None ):
                logFileHandle = open("sys_usage_log_" + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M") + ".csv", "w+")
                logFileHandle.write("Timestamp|Hostname|CPU Util %|Memory Used %|Disk Write Count|Open N/W Conn|DB Locks\r\n")
            logFileHandle.write(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + "|" + stat_str_collected + "\r\n")

        #TODO::send_stats(stat_str_collected)
        send_stats(stat_str_collected)
        time.sleep(COLLECTION_INTERVAL_SECS)
    except Exception as e:
        time.sleep(1) #do nothing

    
