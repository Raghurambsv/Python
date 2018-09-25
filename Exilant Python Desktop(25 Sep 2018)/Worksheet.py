import re
import os



from __future__ import unicode_literals
import spacy
import re
import sys
import docx2txt        
with open("/Users/raghuram.b/Desktop/Chinmaya_Mishra_12.6yrs_Java_Hissar.doc",'r') as file:
    content=file.read()

print(content)    
 


lines = re.split(r'[\n\r]+', document)   
#####################Pattern catching for COMMENTS#####################    
pattern=re.compile('COMMENT ON COLUMN Chnlsls_Trng_App.WRK_COLL_ENTITY_INCR_DTL_CONT.(\w+)\s+')
Comment_List=pattern.findall(content)
print("The list of Comments in file is as below")
print("========================================")
for var in Comment_List:
    print(var)


#####################Pattern catching for capturing DATAFIELDS#####################
with open("//Users//raghuram.b//Desktop//WRK_COLL_ENTITY_INCR_DTL_CONT.ddl",'r') as file:
    content=file.readlines()


    
for line in content:
    result= re.findall('^(.*)DECIMAL.*$',line)
    result=result+re.findall('^(.*)VARCHAR.*$',line)
    result=result+re.findall('^(.*)DATE.*$',line)
    result=result+re.findall('^(.*)TIMESTAMP.*$',line)
    result=result+re.findall('^(.*)\\bCHAR\\b.*$',line)
    file=open("//Users//raghuram.b//Desktop//result.txt",'a') 
    file.writelines(result)
    file.close()
    
with open("//Users//raghuram.b//Desktop//result.txt",'r') as file:
    content=file.read()
    
Fields=content.split()
print("The list of DataFields in the file are as below")
print("===============================================")
for var in Fields:
    print(var)
            
with open("//Users//raghuram.b//Desktop//WRK_COLL_ENTITY_INCR_DTL_CONT.ddl",'r') as file:
    content=file.readlines()    
file=open("//Users//raghuram.b//Desktop//attributes.txt",'a')
for line in content:
    if line.find('DECIMAL') == -1:
        pass
    else:    
        file.writelines(line)
    if line.find('VARCHAR') == -1:
        pass
    else:    
        file.writelines(line) 
    if line.find('CHAR') == -1:
        pass
    else:    
        file.writelines(line) 
    if line.find('DATE') == -1:
        pass
    else:    
        file.writelines(line)
    if line.find('TIMESTAMP') == -1:
        pass
    else:    
        file.writelines(line)  
file.close()    

with open("//Users//raghuram.b//Desktop//attributes.txt",'r') as file:
    content=file.read()
    
Fields=content.splitlines()
print("The list of ATTRIBUTES in the file are as below")
print("===============================================")
for var in Fields:
    print(var)
#####################Pattern for capturing 3 attributes#####################
with open("//Users//raghuram.b//Desktop//attributes.txt",'r') as file:
    content=file.readlines()
print("The list of ATTRIBUTES in the file are as below")
print("===============================================")    
for line in content:
    result= re.match("\s*(\w+)\s*(\S+)\s+TITLE\s\'(\S+.*)\'",line)
    if result is None:
        pass
    else:
        print( re.sub('TITLE','',result.group()))
#        

for line in content:
    result= re.match("\s*(\w+)\s*(\S+)\s+CHARACTER SET LATIN NOT CASESPECIFIC TITLE\s\'(\S+.*)\'",line)
    if result is None:
        pass
    else:
         print( re.sub('CHARACTER SET LATIN NOT CASESPECIFIC TITLE','',result.group()))  
        
for line in content:
    result= re.match("\s*(\w+)\s*(\S+)\s+FORMAT 'YYYY-MM-DD' TITLE\s\'(\S+.*)\'",line)

    if result is None:
        pass
    else:
        print( re.sub('FORMAT \'YYYY-MM-DD\' TITLE','',result.group()))        
        
           
    

                

#Using fo cleanup activity


os.remove("//Users//raghuram.b//Desktop//result.txt")
#os.remove("//Users//raghuram.b//Desktop//attributes.txt")
#os.remove("//Users//raghuram.b//Desktop//attributes.txt")
print("\n\n*******Temp File Removed after use******")



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


import re
import os
with open("//Users//raghuram.b//Desktop//WRK_COLL_ENTITY_INCR_DTL_CONT.ddl",'r') as file:
    content=file.read()
    
#####################Pattern catching for COMMENTS#####################    
pattern=re.compile('COMMENT ON COLUMN Chnlsls_Trng_App.WRK_COLL_ENTITY_INCR_DTL_CONT.(\w+)\s+')
Comment_List=pattern.findall(content)
print("The list of Comments in file is as below")
print("========================================")
for var in Comment_List:
    print(var)


#####################Pattern catching for capturing DATAFIELDS#####################
with open("//Users//raghuram.b//Desktop//WRK_COLL_ENTITY_INCR_DTL_CONT.ddl",'r') as file:
    content=file.readlines()


    
for line in content:
    result= re.findall('^(.*)DECIMAL.*$',line)
    result=result+re.findall('^(.*)VARCHAR.*$',line)
    result=result+re.findall('^(.*)DATE.*$',line)
    result=result+re.findall('^(.*)TIMESTAMP.*$',line)
    result=result+re.findall('^(.*)\\bCHAR\\b.*$',line)
    file=open("//Users//raghuram.b//Desktop//result.txt",'a') 
    file.writelines(result)
    file.close()
    
with open("//Users//raghuram.b//Desktop//result.txt",'r') as file:
    content=file.read()
    
Fields=content.split()
print("The list of DataFields in the file are as below")
print("===============================================")
for var in Fields:
    print(var)
            
with open("//Users//raghuram.b//Desktop//WRK_COLL_ENTITY_INCR_DTL_CONT.ddl",'r') as file:
    content=file.readlines()    
file=open("//Users//raghuram.b//Desktop//attributes.txt",'a')
for line in content:
    if line.find('DECIMAL') == -1:
        pass
    else:    
        file.writelines(line)
    if line.find('VARCHAR') == -1:
        pass
    else:    
        file.writelines(line) 
    if line.find('CHAR') == -1:
        pass
    else:    
        file.writelines(line) 
    if line.find('DATE') == -1:
        pass
    else:    
        file.writelines(line)
    if line.find('TIMESTAMP') == -1:
        pass
    else:    
        file.writelines(line)  
file.close()    

with open("//Users//raghuram.b//Desktop//attributes.txt",'r') as file:
    content=file.read()
    
Fields=content.splitlines()
print("The list of ATTRIBUTES in the file are as below")
print("===============================================")
for var in Fields:
    print(var)
#####################Pattern for capturing 3 attributes#####################
with open("//Users//raghuram.b//Desktop//attributes.txt",'r') as file:
    content=file.readlines()
print("The list of ATTRIBUTES in the file are as below")
print("===============================================")    
for line in content:
    result= re.match("\s*(\w+)\s*(\S+)\s+TITLE\s\'(\S+.*)\'",line)
    if result is None:
        pass
    else:
        print( re.sub('TITLE','',result.group()))
#        

for line in content:
    result= re.match("\s*(\w+)\s*(\S+)\s+CHARACTER SET LATIN NOT CASESPECIFIC TITLE\s\'(\S+.*)\'",line)
    if result is None:
        pass
    else:
         print( re.sub('CHARACTER SET LATIN NOT CASESPECIFIC TITLE','',result.group()))  
        
for line in content:
    result= re.match("\s*(\w+)\s*(\S+)\s+FORMAT 'YYYY-MM-DD' TITLE\s\'(\S+.*)\'",line)

    if result is None:
        pass
    else:
        print( re.sub('FORMAT \'YYYY-MM-DD\' TITLE','',result.group()))        
        
           
    

                

#Using fo cleanup activity


os.remove("//Users//raghuram.b//Desktop//result.txt")
#os.remove("//Users//raghuram.b//Desktop//attributes.txt")
#os.remove("//Users//raghuram.b//Desktop//attributes.txt")
print("\n\n*******Temp File Removed after use******")

