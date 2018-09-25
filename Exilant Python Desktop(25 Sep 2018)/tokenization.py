import pandas
import re
from nltk.corpus import stopwords
from textblob import TextBlob
stop = stopwords.words('english')

df=pandas.read_csv("//users//raghuram.b//Desktop//RFA.csv")


###############################FOR Disposition COLUMN###############################
df=df['Disposition']

print(df.str.split().str.len().sum()) #No of words BEFORE

df=df.fillna(0) #Make NaN to zero


#Eliminating speacial and unwanted characters
df =df.str.replace('^\s*$','')
df =df.str.replace('\.', ' ')
df =df.str.replace('\n',' ')
df =df.str.replace('+',' ')
df =df.str.replace('-',' ')
df =df.str.replace(',',' ')
df =df.str.replace(';',' ')
df =df.str.replace("\'",' ')
df =df.str.replace('\d+',' ')
df =df.str.replace('[#\?\=\^\&\Ø\(\)\@\~\`\%\[\]\_\<\>\':,\"\/\*]', '').drop_duplicates().astype(str)
     
#for eliminating stop words        
df = df.apply(lambda x: " ".join(x for x in x.split() if x not in stop))    
#df=df.apply(lambda x: str(TextBlob(x).correct())) #For checking grammer   

#Convert to List and remove duplicates of sentences
df =df.drop_duplicates().values.tolist() 

#For converting sentences to indivual words
list=[]
for x in df:
    for y in x.split(): 
        list.append(y)    

         
#To convert to lowercase          
list=[x.lower() for x in list]

#For removing duplicates words
newList = []
for i in list:
    if i not in newList:
        newList.append(i)
        


print(len(newList)) #No of words AFTER
 
with open("//Users//raghuram.b//Desktop//DispositionColumnResult.txt",'w') as file:
    for i in newList:
        file.writelines(i+' ')
    

###############################FOR ISSUES COLUMN###############################
df=pandas.read_csv("//users//raghuram.b//Desktop//RFA.csv")

df=df['Issue']

print(df.str.split().str.len().sum()) #No of words BEFORE

df=df.fillna(0) #Make NaN to zero

#Eliminating speacial and unwanted characters
df =df.str.replace('^\s*$','')
df =df.str.replace('\.', ' ')
df =df.str.replace('\n',' ')
df =df.str.replace('+',' ')
df =df.str.replace('-',' ')
df =df.str.replace(',',' ')
df =df.str.replace(';',' ')
df =df.str.replace("\'",' ')
df =df.str.replace('\d+',' ')
df =df.str.replace('[#\?\=\^\&\Ø\(\)\@\~\`\%\[\]\_\<\>\':,\"\/\*]', '').drop_duplicates().astype(str)

#for eliminating stop words        
df = df.apply(lambda x: " ".join(x for x in x.split() if x not in stop))    
#df=df.apply(lambda x: str(TextBlob(x).correct())) #For checking grammer   

#Convert to List and remove duplicates of sentences
df =df.drop_duplicates().values.tolist() 

#For converting sentences to indivual words
list=[]
for x in df:
    for y in x.split(): 
        list.append(y)    

print(list)             
#To convert to lowercase          
list=[x.lower() for x in list]

#For removing duplicates words
newList = []
for i in list:
    if i not in newList:
        newList.append(i)
        


print(len(newList)) #No of words AFTER
 
with open("//Users//raghuram.b//Desktop//IssueColumnResult.txt",'w') as file:
    for i in newList:
        file.writelines(i+' ')
    

