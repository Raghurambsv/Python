import re
import os
#with open("//Users//raghuram.b//Desktop//WRK_COLL_ENTITY_INCR_DTL_CONT.ddl",'r') as file:
#    content=file.read()
#    
##Pattern catching for COMMENTS    
#pattern=re.compile('COMMENT ON COLUMN Chnlsls_Trng_App.WRK_COLL_ENTITY_INCR_DTL_CONT.(\w+)\s+')
#Comment_List=pattern.findall(content)
#print("The list of Comments in file is as below")
#print("========================================")
#for var in Comment_List:
#    print(var)
#

#Pattern catching for capturing DATAFIELDS
with open("//Users//raghuram.b//Desktop//WRK_COLL_ENTITY_INCR_DTL_CONT.ddl",'r') as file:
    content=file.readlines()


for line in content:
    if line.startswith('VARCHAR'):
        print(line)
#        file=open("//Users//raghuram.b//Desktop//result.txt",'a') 
#        file.writelines(line)
#        file.close()
    
#    result= re.findall('^(.*)DECIMAL.*$',line)
#    result=result+re.findall('^(.*)VARCHAR.*$',line)
#    result=result+re.findall('^(.*)DATE.*$',line)
#    result=result+re.findall('^(.*)TIMESTAMP.*$',line)
#    result=result+re.findall('^(.*)\\bCHAR\\b.*$',line)
#    file=open("//Users//raghuram.b//Desktop//result.txt",'a') 
#    file.writelines(result)
#    file.close()
#    
#for line in content:
#    result= re.match('\s*(\w+)\s*(\w+)',line)
##    result=result+re.findall('^(.*)DECIMAL.*$',line)
##    result=result+re.findall('^(.*)DATE.*$',line)
##    result=result+re.findall('^(.*)TIMESTAMP.*$',line)
##    result=result+re.findall('^(.*)\\bCHAR\\b.*$',line)
#    if result is None:
#        print(" ")
#    else:
#        print(result.group())
#    file=open("//Users//raghuram.b//Desktop//result.txt",'a') 
#    file.write(result.group())
#    file.close()
#    
#with open("//Users//raghuram.b//Desktop//result.txt",'r') as file:
#    content=file.read()
#    
#Fields=content.split()
#print("The list of DataFields in the file are as below")
#print("===============================================")
#for var in Fields:
#    print(var)
#            

#Using fo cleanup activity


#os.remove("//Users//raghuram.b//Desktop//result.txt")
#print("\n\n*******Temp File Removed after use******")