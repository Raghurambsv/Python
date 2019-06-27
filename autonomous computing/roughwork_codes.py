ARIMA
######
select avg(stat_value) from predicted_temp_data where  host_name='rn-act-lapp02.rno.apple.com' and prediction_type=1 ;  --1.53
select avg(stat_value) from predicted_temp_data where  host_name='rn-act-lapp04.rno.apple.com' and prediction_type=1 ;  --4.89
select avg(stat_value) from predicted_temp_data where  host_name='rn-act-lapp05.rno.apple.com' and prediction_type=1 ; --2.5


RF
##
select avg(stat_value) from predicted_temp_data where  host_name='rn-act-lapp02.rno.apple.com' and prediction_type=2; --6.99
select avg(stat_value) from predicted_temp_data where  host_name='rn-act-lapp04.rno.apple.com' and prediction_type=2; --2.8 
select avg(stat_value) from predicted_temp_data where  host_name='rn-act-lapp05.rno.apple.com' and prediction_type=2; --8.3

########PANDAS TO GENERATE DATES AS PER YOUR LIKE (CORRECT SAMAS DATA)
import pandas as pd
#data = pd.read_csv("stat_train_mindata.csv",index_col=0)
colnames=['hostName', 'statId', 'timestamp', 'value']
#data = pd.read_csv("/Users/raghuram.b/Desktop/test/cpu_usage_log_07_05_2019_11_25.csv",index_col=0,names=colnames)
data = pd.read_csv("/Users/raghuram.b/Desktop/test/samas_44hours.csv",index_col=0,names=colnames,skiprows=[0])
data.head()
print("size of csv :",data.shape)
rowcount=data.shape[0]
 
data['datetime']=pd.date_range(start='5/19/2019 00:00:00',periods=rowcount,freq='5s')
data= data.drop(['timestamp'], axis = 1)

data.to_csv('/Users/raghuram.b/Desktop/test/corrected_samas_44hours.csv') #save the dataframe

# =============================================================================

########PANDAS TO GENERATE DATES AS PER YOUR LIKE
import pandas as pd
#data = pd.read_csv("stat_train_mindata.csv",index_col=0)
colnames=['time', 'cpuutilisation'] 
#data = pd.read_csv("/Users/raghuram.b/Desktop/test/cpu_usage_log_07_05_2019_11_25.csv",index_col=0,names=colnames)
data = pd.read_csv("/Users/raghuram.b/Desktop/test/week.csv",index_col=0,names=colnames,skiprows=[0])
data.head()
print("size of csv :",data.shape)
rowcount=data.shape[0]
#data['datetime']=pd.date_range(start='1/1/2019',periods=rowcount,freq='M')  
#data['datetime']=pd.date_range(start='1/31/2019',periods=rowcount,freq='D') 
#data['datetime']=pd.date_range(start='1/31/2019',periods=rowcount,freq='H')     
data['datetime']=pd.date_range(start='5/06/2019 00:00:00',periods=rowcount,freq='5s')


#######Try by giving start & end date and Frequency
#date_list=pd.date_range('2019-01-01', '2019-01-10',freq='D')
#df=pd.DataFrame(date_list,columns=['raghu_time'])


data.to_csv('/Users/raghuram.b/Desktop/test/correctweek.csv') #save the dataframe

#USING CORRECTED WEEK
colnames=['time', 'cpuutilisation','datetime'] 
data = pd.read_csv("/Users/raghuram.b/Desktop/test/correctweek.csv",index_col=0,names=colnames,skiprows=[0])
data.head()
print("size of csv :",data.shape)
rowcount=data.shape[0]
data=data.set_index('datetime') #make the resampled as index



# =============================================================================
#                          TIMESTAMP PLAYING AROUND
# import datetime
# from datetime import timedelta
# start_date = datetime.datetime.now()
# end_date = start_date + timedelta(days=5)
# print('start_date is',start_date)
# print('end_date is ',end_date)
# 
# 
# #pandas
# import pandas as pd
# startdate = "10/10/2011"
# enddate = pd.to_datetime(startdate) + pd.DateOffset(days=5)
# 
# 
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# 
# print ('Today: ',datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
# date_after_month = datetime.now()+ relativedelta(days=5)
# print ('After 5 Days:', date_after_month.strftime('%d/%m/%Y %H:%M:%S'))
# 
# 

# =============================================================================

#####PANDAS RESAMPLING

#data.index = pd.to_datetime(data.index) #convert index to datetime index (MUST RULE)
##Hourly Summary:
#hourly_summary = pd.DataFrame()
#hourly_summary['hour']=data.cpuutilisation.resample('H').mean()
#
##Min Summary:              
#Min_summary = pd.DataFrame()
#Min_summary['Minutes']=data.cpuutilisation.resample('min').mean()
#
##second_summary:
#second_summary = pd.DataFrame()
#second_summary['secs']=data.cpuutilisation.resample('S').mean()
#


#Alias	Description
#B	Business day
#D	Calendar day
#W	Weekly
#M	Month end
#Q	Quarter end
#A	Year end
#BA	Business year end
#AS	Year start
#H	Hourly frequency
#T, min	Minutely frequency
#S	Secondly frequency
#L, ms	Millisecond frequency
#U, us	Microsecond frequency
#N, ns	Nanosecond frequency
#These are some of the common methods you might use for resampling:
#
#Method	Description
#bfill	Backward fill
#count	Count of values
#ffill	Forward fill
#first	First valid data value
#last	Last valid data value
#max	Maximum data value
#mean	Mean of values in time range
#median	Median of values in time range
#min	Minimum data value
#nunique	Number of unique values
#ohlc	Opening value, highest value, lowest value, closing value
#pad	Same as forward fill
#std	Standard deviation of values
#sum	Sum of values
#var	Variance of values

#Arima forloop function
# =============================================================================
# import itertools
# p=d=q=range(0,5)
# pdq= list(itertools.product(p,d,q))
# print(pdq)
# for param in pdq:
#     try:
#         model_arima= ARIMA(train,order=param)
#         model_arima_fit= model_arima.fit()
#         print(model_arima_fit.aic)
#     except:
#         continue
#     
# =============================================================================



mysql autocompute -e "select * from machine_stats" -B > mytable.tsv;
 mysql mydb -e "select * from machine_stats" -B > mytable.tsv;
 SELECT * INTO OUTFILE '/act/machine_stats.txt' FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n' FROM machine_stats;
 
#prediction  in MACHINE_STATS (stat_type=2) 
------------------------------------------------------------------------------
select count(*) from machine_stats where  host_name='rn-act-lapp02.rno.apple.com' and stat_type=2; --15011 rows
select count(*) from machine_stats where  host_name='rn-act-lapp04.rno.apple.com' and stat_type=2; -- 15011 rows
select count(*) from machine_stats where  host_name='rn-act-lapp05.rno.apple.com' and stat_type=2; -- 15011 rows
 
#prediction row value counts in PREDICTION_TEMP_DATA (prediction_type=1(ARIMA) ......prediction_type=2(RF)) 
------------------------------------------------------------------------------ 
select count(*) from predicted_temp_data where  host_name='rn-act-lapp02.rno.apple.com' and prediction_type=1; --21630 rows (ARIMA)
select count(*) from predicted_temp_data where  host_name='rn-act-lapp04.rno.apple.com' and prediction_type=1; -- 24629 rows (ARIMA)
select count(*) from predicted_temp_data where  host_name='rn-act-lapp05.rno.apple.com' and prediction_type=1; -- 23908 rows (ARIMA)

select count(*) from predicted_temp_data where  host_name='rn-act-lapp02.rno.apple.com' and prediction_type=2; --21630 rows (RF)
select count(*) from predicted_temp_data where  host_name='rn-act-lapp04.rno.apple.com' and prediction_type=2; -- 24620 rows (RF)
select count(*) from predicted_temp_data where  host_name='rn-act-lapp05.rno.apple.com' and prediction_type=2; -- 23908 rows (RF)



 
#To check if DB locks enabled only in M2 (Not in M4 & m5) ...for M4 & M5 sud be "-1"
------------------------------------------------------------------------------ 
 select max(stat_value) from machine_stats where  host_name='rn-act-lapp02.rno.apple.com' and stat_id=50 and stat_type = 1;
 select max(stat_value) from machine_stats where  host_name='rn-act-lapp04.rno.apple.com' and stat_id=50 and stat_type = 1;
 select max(stat_value) from machine_stats where  host_name='rn-act-lapp05.rno.apple.com' and stat_id=50 and stat_type = 1;
 
 
select count(*)  from machine_stats where  host_name='rn-act-lapp04.rno.apple.com' and stat_id=10 and stat_type = 1;
select count(*)  from machine_stats where  host_name='rn-act-lapp04.rno.apple.com' and stat_id=20 and stat_type = 1;
select count(*)  from machine_stats where  host_name='rn-act-lapp04.rno.apple.com' and stat_id=31 and stat_type = 1;
select count(*)  from machine_stats where  host_name='rn-act-lapp04.rno.apple.com' and stat_id=40 and stat_type = 1;
select count(*)  from machine_stats where  host_name='rn-act-lapp04.rno.apple.com' and stat_id=50 and stat_type = 1;
 
select count(*) from machine_stats where  host_name='rn-act-lapp04.rno.apple.com' and stat_type = 1 ;
select count(*)  from machine_stats where  host_name='rn-act-lapp04.rno.apple.com' and stat_type = 2 ;
select  count(*)  from predicted_temp_data where  host_name='rn-act-lapp04.rno.apple.com'

select from_unixtime(stat_timestamp, '%Y-%c-%d %H:%i:%s') IN
(select max(stat_timestamp) from machine_stats where host_name='rn-act-lapp04.rno.apple.com' and stat_type = 1) from machine_stats;

select max(stat_timestamp) from machine_stats where host_name='rn-act-lapp04.rno.apple.com' and stat_type = 1;
select min(stat_timestamp) from machine_stats where host_name='rn-act-lapp04.rno.apple.com' and stat_type = 1;
select from_unixtime(1559543771, '%Y-%c-%d %H:%i:%s') from machine_stats limit 2; --2019-6-03 06:36:11  (MAX)
select from_unixtime(1559157018, '%Y-%c-%d %H:%i:%s') from machine_stats limit 2; --2019-5-29 19:10:18 (MIN)

select max(stat_timestamp) from predicted_temp_data where host_name='rn-act-lapp05.rno.apple.com' and prediction_type = 1;
select min(stat_timestamp) from predicted_temp_data where host_name='rn-act-lapp05.rno.apple.com' and prediction_type = 1;
select from_unixtime(1559324534, '%Y-%c-%d %H:%i:%s') from machine_stats limit 2; --2019-5-28 09:33:10 (MIN)
select from_unixtime(1559324534, '%Y-%c-%d %H:%i:%s') from machine_stats limit 2;-- 2019-5-31 17:42:14 (MAX)

select from_unixtime(stat_timestamp, '%Y-%c-%d %H:%i:%s') from machine_stats limit 10;



Machine2
########
Predicted ((cpuvalue < 20 is too much more)
--------
select distinct stat_value from predicted_temp_data where  host_name='rn-act-lapp02.rno.apple.com'; --TOTAL : 1382 rows 
select distinct stat_value from predicted_temp_data where stat_value > 20 and host_name='rn-act-lapp02.rno.apple.com';--GREATER THAN 20 => 123 outof 1382 rows
select distinct stat_value from predicted_temp_data where stat_value < 20 and host_name='rn-act-lapp02.rno.apple.com'; -- LESS THAN 20 => 1259 outof 1382 rows

Machine_stats ((cpuvalue < 20 is way too less)
-------------
select count(distinct stat_value) from machine_stats where  host_name='rn-act-lapp02.rno.apple.com'; --TOTAL : 31096 rows
select count(distinct stat_value) from machine_stats where stat_value > 20 and host_name='rn-act-lapp02.rno.apple.com'; --GREATER THAN 20 => 30711 outof 31096
select count(distinct stat_value) from machine_stats where stat_value < 20 and host_name='rn-act-lapp02.rno.apple.com'; -- LESS THAN 20 => 387 outof 31096



Machine4
########
Predicted ((cpuvalue < 20 is considerable more)
--------
select  count(distinct stat_value) from predicted_temp_data where  host_name='rn-act-lapp04.rno.apple.com'; --TOTAL : 1784 rows 
select count(distinct stat_value) from predicted_temp_data where stat_value > 20 and host_name='rn-act-lapp04.rno.apple.com';--GREATER THAN 20 => 574 outof 1784 rows
select count(distinct stat_value) from predicted_temp_data where stat_value < 20 and host_name='rn-act-lapp04.rno.apple.com'; -- LESS THAN 20 => 1210 outof 1784 rows

Machine_stats ((cpuvalue < 20 is considerable less)
------------- 
select count(distinct stat_value) from machine_stats where  host_name='rn-act-lapp04.rno.apple.com'; --TOTAL : 3176 rows
select count(distinct stat_value) from machine_stats where stat_value > 20 and host_name='rn-act-lapp04.rno.apple.com'; --GREATER THAN 20 => 2661 out of 3176
select count(distinct stat_value) from machine_stats where stat_value < 20 and host_name='rn-act-lapp04.rno.apple.com'; -- LESS THAN 20 => 514 out of 3176



Machine5
########
Predicted (cpuvalue < 20 is considerable more)
--------
select  count(distinct stat_value) from predicted_temp_data where  host_name='rn-act-lapp05.rno.apple.com'; --TOTAL : 1349 rows 
select count(distinct stat_value) from predicted_temp_data where stat_value > 20 and host_name='rn-act-lapp05.rno.apple.com';--GREATER THAN 20 => 479 outof 1349 rows
select count(distinct stat_value) from predicted_temp_data where stat_value < 20 and host_name='rn-act-lapp05.rno.apple.com'; -- LESS THAN 20 => 870 outof 1349 rows

Machine_stats ((cpuvalue < 20 is considerable less)
------------- 
select count(distinct stat_value) from machine_stats where  host_name='rn-act-lapp05.rno.apple.com'; --TOTAL : 3849 rows
select count(distinct stat_value) from machine_stats where stat_value > 20 and host_name='rn-act-lapp05.rno.apple.com'; --GREATER THAN 20 => 3373 out of 3849
select count(distinct stat_value) from machine_stats where stat_value < 20 and host_name='rn-act-lapp05.rno.apple.com'; -- LESS THAN 20 => 475 out of 3849


All queries at June 3rd 10:20 AM
---------------------------------
mysql> select count(*) from machine_stats where stat_type=1;
+----------+
| count(*) |
+----------+
|   471907 |
+----------+
1 row in set (1.67 sec)

mysql> 
mysql> select count(*) from machine_stats where stat_type=2;
+----------+
| count(*) |
+----------+
|    45033 |
+----------+
1 row in set (1.58 sec)

mysql> select count(*) from machine_stats where stat_type=1 and host_name='rn-act-lapp02.rno.apple.com';
+----------+
| count(*) |
+----------+
|   159652 |
+----------+
1 row in set (2.47 sec) 


mysql> select count(*) from machine_stats where stat_type=1 and host_name='rn-act-lapp04.rno.apple.com';
+----------+
| count(*) |
+----------+
|   156325 |
+----------+
1 row in set (2.30 sec)

mysql> select count(*) from machine_stats where stat_type=1 and host_name='rn-act-lapp05.rno.apple.com';
+----------+
| count(*) |
+----------+
|   156070 |
+----------+
1 row in set (2.34 sec)

mysql> select count(*) from predicted_temp_data;
+----------+
| count(*) |
+----------+
|   140325 |

