import re
import os
import pandas as pd
import csv  



##Open the file to be parsed
with open("//Users//raghuram.b//Desktop//Savita Projects//(Ritu)DDL Parser//WRK_COLL_ENTITY_INCR_DTL_CONT.ddl",'r') as file:
    content=file.readlines()

##Store the data in Txt file
data=open("//Users//raghuram.b//Desktop//result.txt",'w')


#####################Database & Tablename#####################
for line in content:
    result=re.match(r"^CREATE\s\w+\s\w+\s(\w+)\.(\w+).*",line)

    if result == None:
        pass
    else:
##Collecting Database & Tablename for final report
        df1 = pd.DataFrame([result.group(1)],columns=['DATABASE_NAME'])
        df2= pd.DataFrame([result.group(2)],columns=['TABLE_NAME'])
        
#####################for capturing columns of table#####################

data.writelines('\n') 

for line in content:
    Datatype=re.match(r"\s*(\w+)\s*(\S+)\s+[\S*\s*]{1,}TITLE\s\'(\S+.*)\'",line)
    if Datatype == None:
        pass
    else:
        part=Datatype.group(2)
        if ')' in part:
    
             result=re.match(r"\s*(\w+)\s*(\w+)\((\d+)([,\d]{0,})\)\s+[\S*\s*]{1,}TITLE\s\'(\S+.*)\'",line)
             if (result == None):
                 pass
             else:
                 data.writelines(result.group(1)+'\t'+result.group(2)+'\t'+result.group(3)+'\t'+result.group(4).strip(',')+'\t'+result.group(5)+'\n')
       
        else:
            result=re.match(r"\s*(\w+)\s*(\S+)\s+[\S*\s*]{1,}TITLE\s\'(\S+.*)\'",line)
            if (result == None):
                pass
            else:
                data.writelines(result.group(1)+'\t'+result.group(2)+'\t'+' '+'\t'+' '+'\t'+result.group(3)+'\n')
        

#####################NON-PII_None columns#####################    
comment=[]        
for line in content:
    result=re.match(r"^COMMENT\s(\w+).*\'(\w+.*)\'",line)
    if result == None:
        pass
    else:
        var=result.group(2)
      
        if var == 'PII_None':
           pass
        else:
           comment.append(var)
###Collecting comments for final report            
df3=pd.DataFrame(comment,columns=['PII_COLUMN'])

data.close()
            
##########################Converting Txt to Csv file only for 5 columns##############################        

import csv

with open('//Users//raghuram.b//Desktop//result.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("\t") for line in stripped if line)
         
    with open('//Users//raghuram.b//Desktop//temp.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('COLUMN_NAME','DATA_TYPE','LENGTH','SCALE','TITLE'))
        writer.writerows(lines)            
            
#########################Generating Final report with 5 columns##################################   

    
df = pd.read_csv('//Users//raghuram.b//Desktop//temp.csv',header=0,
                 names=['COLUMN_NAME','DATA_TYPE','LENGTH','SCALE','TITLE'])
df = pd.DataFrame(df,columns=['COLUMN_NAME','DATA_TYPE','LENGTH','SCALE','TITLE'])

###Adding below columns to report
df.insert(loc=0, column='DATABASE_NAME', value=df1)
df.insert(loc=1, column='TABLE_NAME', value=df2)
df.insert(loc=7, column='PII_COLUMN', value=df3)

##Storing Final result as CSV file
df.to_csv('//Users//raghuram.b//Desktop//TD_Parser_Report.csv')

##Removing Temp File
os.remove("//Users//raghuram.b//Desktop//temp.csv")
