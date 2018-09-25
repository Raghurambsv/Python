


###################DICTIONARY#####################

##Get method is only for dictonary objects ( It helps for counting NO OF LETTERS)
count = {}
with open("//Users//raghuram.b//Desktop//Python//file2.txt",'r') as file:
    read_file=file.read()
    for ch in read_file:
        count[ch]=count.get(ch,0)+1
        
print(count)
print(count.keys())##Accesing ALL keys in dict
print(count.values())##Accesing ALL values in dict
print(count['h']) ##Accesing values in dict
count['h']=65 ##updating the dict values
print(count['h'])

dict=count #copy one dictionary to another
print(dict)

print(len(count))

#count.clear()#delete all the contents
#del count #drop entire dictionary



###############TIME###################
import time
print(time.localtime())
print(time.asctime(time.localtime())) ##Readable format for time

'''
dir(list) or dir(<any listname>) gives the list of all the methods you can apply on them

dir(str) or dir(<any string name>) gives you list of all methods you can apply on string
'''

## Date and timestamp related
from datetime import datetime
delta = datetime.now() - datetime(2017,7,9)
print(delta.days,delta.seconds)

now=datetime.now()
then=datetime(2016,9,23)
time=now - then

print('The time now is ',time)

#formatting of  date
whenever= datetime.strptime('2017-12-24','%Y-%m-%d')
print(whenever)


whenever=datetime.now()
print(whenever.strftime('%Y'))



## ITERATE MULTIPLE LISTS or SEQUENCES

a = ['a','b','c','d']
b = [1,2,3,4]
c = ['aa','bb','cc','dd']
for i,j,k in zip(a,b,c):
    print('%s is %s and %s'% (i,j,k))
    

## concatenating file contents

import glob2
from datetime import datetime 
filename=glob2.glob("//Users//raghuram.b//Desktop//Python//file*")

Newfile=str(datetime.now()) + ".txt"

with open("//Users//raghuram.b//Desktop//Python//"+Newfile,'w') as Newfile:
    for file in filename:
        with open (file,'r') as f:
            Newfile.write(f.read() + "\n")

print("job done")            
                



    
    
Numpy
=====
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 17:11:55 2018

@author: raghuram.b
"""

import numpy
n=numpy.arange(27)
print(n)

n=n.reshape(3,3,3)
print(n)

#lists to arrays
m=numpy.asarray([[123,121,33,44,55],[ ],[5,6,7,8]])
print(m)
print(type(m))


import cv2 # for image processing
im_grey=cv2.imread("//users//raghuram.b//Desktop//Python//Datasets//smallgray.png",0) #0 means greyimage and 1 means RGB image
#o means 2D image and 1 means 3D image
print(im_grey)

im_grey=cv2.imread("//users//raghuram.b//Desktop//Python//Datasets//smallgray.png",1)
print(im_grey)

#create image from input array im_grey which has the numbers to create a image
cv2.imwrite("Newsmallgrey.png",im_grey)

Pandas
=======
import pandas
#df=pandas.DataFrame([[10,20,30],[1,3,5],[2,4,6]],columns=["Price","Age","Value"],index=["First","Second","Third"])
#print(df,"\n")
#print("The mean of the dataframe",df.mean())
#print("The mean of the FULL dataframe",df.mean().mean())
#print("The mean of indivual columns PRICE:",df.Price.mean(),"AGE:",df.Age.mean())
#
#
#print("\nCSV file with index\n")
#df1=pandas.read_csv("//users//raghuram.b//Desktop//Python//Datasets//supermarkets.csv")
#print(df1.set_index("ID"))
#
#
#print("\nExcel file with index\n")
#df2=pandas.read_excel("//users//raghuram.b//Desktop//Python//Datasets//supermarkets.xlsx",sheet_name=0)
#print(df2)
#
#print("\nJSON file with index\n")
#df3=pandas.read_json("//users//raghuram.b//Desktop//Python//Datasets//supermarkets.json")
#print(df3.set_index("ID"))
#
#print("\nSEMICOLON seperated file\n")
#df4=pandas.read_csv("//users//raghuram.b//Desktop//Python//Datasets//supermarkets-semi-colons.txt",sep=";")
#print(df4)
#
#print("\nDIRECT from URL\n")
#df5=pandas.read_csv("http://pythonhow.com/supermarkets.csv")
#print(df5)


#loc for referencing thru LABELS
print("\nJSON file with Indexing")
df7=pandas.read_json("//users//raghuram.b//Desktop//Python//Datasets//supermarkets.json")
print("\n\nThe size of your dataframe is:",df7.shape,"the no of rows:",df7.shape[0],"the no of cols:",df7.shape[1])
df7=df7.set_index("Address") #setting this as unique indexing
print(df7.loc["735 Dolores St":"3995 23rd St","City":"ID"])
print(df7.loc["735 Dolores St":"3995 23rd St","ID":]) #only rows 2 to 3 and columns from City to ID
print(list(df7.loc[:,"City"])) #Only city but all rows and make it list


#iloc for referencing through indices
print(df7.iloc[:,2:3]) #all rows and 2 and 3 column
print(df7.iloc[2:5,3:6]) #2 to 5 rows and 3to6th column

#ix for single elements and more
print(df7.ix[3,4])
print(df7.ix[3:6,2:5])

#any order of columns (but all rows) use list to give the column names
print(df7[['City','Name','Country']])

#you can give any combination of rows and cols by giving them in List format
df8=df7
df8=df8.set_index("ID")
print(df7.index)
print(df8.columns)
print(df8.loc[[1,3,2,5],['City','Name']])

#dropping columns (Give 1 at last ==> 1 means columns)
print(df8.drop("Name",1))
print(df8.drop(["City","State"],1))
print(df8.drop(df8.columns[3:5],1))

#dropping rows (Give 0 at last ==> 0 means Rows)
print(df8.drop([1,2,3],0))
print(df8.drop(df8.index[2:5],0))


#Add extra column 
#print(df8["Continent"]=df8.shape[0]*["North America"]))

#Modifying column
df8["Continent"]=df8["Country"]+" belongs to "+df8["Continent"]


#Adding row (one way by transpose ..add a new column..reverse back transpose)
df8_t=df8.T #T is a method to transpose
df8_t["My_Address"]=["My city","India",9999,"raghu","karnataka","Asia"]
df8=df8_t.T
print(df8)

#or you can modify existing row contents as below
#df8_t=df8.T #T is a method to transpose
#df8_t[6]=["My city","India",9999,"raghu","karnataka","Asia"]
#df8=df8_t.T
e #print(df8)

Webscarping 
===========
import requests
from bs4 import BeautifulSoup
import re

url=requests.get("https://www.pythonhow.com/real-estate/rock-springs-wy/LCWYROCKSPRINGS/")
str=url.content
soup=BeautifulSoup(str,"html.parser")

#print(soup.prettify())


all=soup.find_all("div",{"class":"propertyRow"})
for item in all:
    print(item.find("h4",{"class":"propPrice"}).text.replace("\n",''))
    print(item.find_all("span",{"class":"propAddressCollapse"})[0].text)
    print(item.find_all("span",{"class":"propAddressCollapse"})[1].text)
    try:
        print(item.find("span",{"class":"infoBed"}).find("b").text)
    except:
        print("None")
    
    try:
        print(item.find("span",{"class":"infoSqft"}).find("b").text)
    except:
        print("None")
    

    try:
        print(item.find("span",{"class":"infoValueFullBath"}).find("b").text)
    except:
        print("None")
            
        
    try:
        print(item.find("span",{"class":"infoValueHalfBath"}).find("b").text)
    except:
        print("None")
        
    
    try:
        for columnGroup in item.find_all("div",{"class":"columnGroup"}):
            for featureGroup,featureName in zip(columnGroup.find_all("span",{"class":"featureGroup"}),columnGroup.find_all("span",{"class":"featureName"})):
                if "Lot Size" in featureGroup.text:
                    print(featureName.text)
    except:
         print("None")
                
        
    print(" ")
    
        
 Webscraping with Dataframes
=============================   
    
 import requests
from bs4 import BeautifulSoup
import re
import pandas
import string


base_url="https://www.pythonhow.com/real-estate/rock-springs-wy/LCWYROCKSPRINGS/"
l=[]

for page in range(0,30,10):   
    url=base_url+"t=0&s="+"%s"%page+".html"
    
    url=requests.get(url)
    
    str=url.content
    soup=BeautifulSoup(str,"html.parser")
    
    #print(soup.prettify())  /t=0&s=10.html
    
    
    all=soup.find_all("div",{"class":"propertyRow"})
    
    for item in all:
        d={}
        d["Price"]=item.find("h4",{"class":"propPrice"}).text.replace("\n",'')
        d["Address"]=item.find_all("span",{"class":"propAddressCollapse"})[0].text
        d["Locality"]=item.find_all("span",{"class":"propAddressCollapse"})[1].text
        try:
            d["Beds"]=item.find("span",{"class":"infoBed"}).find("b").text
        except:
            d["Beds"]=None
        
        try:
            d["Area"]=item.find("span",{"class":"infoSqft"}).find("b").text
        except:
            d["Area"]=None
        
    
        try:
            d["Full Baths"]=item.find("span",{"class":"infoValueFullBath"}).find("b").text
        except:
            d["Full Baths"]=None
                
            
        try:
            d["Half Baths"]=item.find("span",{"class":"infoValueHalfBath"}).find("b").text
        except:
            d["Full Baths"]=None
            
        
        try:
            for columnGroup in item.find_all("div",{"class":"columnGroup"}):
                for featureGroup,featureName in zip(columnGroup.find_all("span",{"class":"featureGroup"}),columnGroup.find_all("span",{"class":"featureName"})):
                    if "Lot Size" in featureGroup.text:
                        d["Lot Size"]=featureName.text
        except:
             print("None")
        l.append(d)     
                    
df=pandas.DataFrame(l)
print(df)   


Classes and sub classes
=======================
class Account:
    '''This is Account's class code'''
    
    def __init__ (self,filepath):
        self.filepath=filepath
        with open(filepath,'r') as file:
            self.balance=int(file.read())
        
    def withdraw(self,amount):
        self.balance=self.balance - amount
        
    def deposit(self,amount):
        self.balance=self.balance + amount   
        
    def commit(self):
        with open(self.filepath,'w') as file:
            file.write(str(self.balance))
            
#subclassing            
class Checking(Account):
    '''This is checkings class code'''
 
    def __init__(self,filepath):
        Account.__init__(self,filepath)
        
    def transfer(self,amount):
        self.balance = self.balance - amount
        
              
       
       
Bank=Checking('//users//raghuram.b//Account.py') 

print(Bank.balance) 
Bank.commit()
Bank.transfer(100)
Bank.commit()
print(Bank.balance) 
print(Bank.__doc__)



Sample Dataframes and pandas
=============================
import requests
from bs4 import BeautifulSoup
import re

url=requests.get("https://www.pythonhow.com/real-estate/rock-springs-wy/LCWYROCKSPRINGS/")
str=url.content
soup=BeautifulSoup(str,"html.parser")

#print(soup.prettify())

all=soup.find_all("div",{"class":"propertyRow"})
for item in all:
    print(item.find("h4",{"class":"propPrice"}).text.replace("\n",''))
    print(item.find_all("span",{"class":"propAddressCollapse"})[0].text)
    print(item.find_all("span",{"class":"propAddressCollapse"})[1].text)
    try:
        print(item.find("span",{"class":"infoBed"}).find("b").text)
    except:
        print("None")
    
    try:
        print(item.find("span",{"class":"infoSqft"}).find("b").text)
    except:
        print("None")
    

    try:
        print(item.find("span",{"class":"infoValueFullBath"}).find("b").text)
    except:
        print("None")
            
        
    try:
        print(item.find("span",{"class":"infoValueHalfBath"}).find("b").text)
    except:
        print("None")
        
    
    try:
        for columnGroup in item.find_all("div",{"class":"columnGroup"}):
            for featureGroup,featureName in zip(columnGroup.find_all("span",{"class":"featureGroup"}),columnGroup.find_all("span",{"class":"featureName"})):
                if "Lot Size" in featureGroup.text:
                    print(featureName.text)
    except:
         print("None")
                
        
    print(" ")
    
Justdial
========
import requests
from bs4 import BeautifulSoup
import re

req = requests.get("https://www.goibibo.com/flights/air-BLR-SIN-20180516--2-0-0-E-I/?utm_source=google&utm_medium=cpc&utm_campaign=DF-Brand-EM&utm_adgroup=Only%20Goibibo&campaign=DF-Brand-EM&gclid=Cj0KCQjw5-TXBRCHARIsANLixNzakRPbTWVWAsy-Eh0YvH2tR7krPXr5F0WnTk9XwkdUvquvvJ3iJF8aAu-UEALw_wcB/")
str=req.content
soup=BeautifulSoup(str,"html.parser")


all=soup.find_all("div",{"class":"headerBox col-md-12 col-sm-12 col-xs-12"})
print(all)    
    
Grocery (try n except.....if and while loops)  
==============================================      
    
file = open("/Users/raghuram.b/Desktop/Python/sample.txt")
pin={"raghu":1234,"mike":1111,"draco":2222}

try:
    check=input("Enter PIN to authenticate: ")
    check=int(check)
except:
    print("Only numbers are expected as PIN")

  
if check in pin.values():
    for key in pin.keys():
        if pin[key] == check:
            print("You are logged on successfuly" + key)
            
            
            while True:
                check=int(input("Press 1 for Insert Fruit and 2 for check Fruit exists"))
                
                if check == 1:
                    fruit=input("which fruit u want to enlist:").lower()
                    with open("/Users/raghuram.b/Desktop/Python/sample.txt","a") as bile:
                        bile.write("\n")
                        bile.write(fruit)
                    break
                    
                if check == 2:    
                
                    fruit=input("which fruit u want to check exists:").lower()
                    myfile=file.read().lower()                
                    if fruit in myfile:
                        print("The fruit "+fruit+" is found and will be added to cart")
                        break
                    else:
                        print("The fruit is not found")
                    
                   
                    
                   
                    
                

else:
    print("You are not authorized user")
    



#if fruit in myfile:
#    print("The fruit "+fruit+" is found")
#else:
#    print("The fruit is not found")
#        