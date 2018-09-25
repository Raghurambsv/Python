import pandas as pd
#df=pd.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//bachelors.xlsx")
#print(df.head(3))
#df=pd.read_csv(“E:/Test.txt”,sep=’\t’) # Load Data from text file having tab ‘\t’ delimeter print 

#df=pd.read_excel("//Users//raghuram.b//Desktop//Python//Jupyter workings//fileraghu.xlsx",sheet_name='Sheet 2',header=1,names=['slno','data','values','Col4','Col5'])
#print(df.head(3))
#
#result=df.pivot(index='slno',columns='data',values='values')
#print(result)
#
df=pd.read_excel("//Users//raghuram.b//Desktop//Python//Jupyter workings//fileraghu.xlsx",sheet_name='Sheet 1',header=1,usecols="A,B,C")
print(df.head(4))

#df.sort_values(['Sales'],ascending=[True])
#print(df.head(4))


import matplotlib.pyplot as plt
import pandas 
df=pandas.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//bachelors.xlsx")
#Plots in matplotlib reside within a figure object, use plt.figure to create new figure
fig=plt.figure()
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
ax.hist(df['Psychology'],bins = 5)
bx = fig.add_subplot(1,1,1)
bx.hist(df['Public Administration'],bins = 5) #different hist means diff graph

plt.title('Score distribution')
plt.xlabel('Score')
plt.ylabel('#Students')
plt.show()


fig=plt.figure()
cx = fig.add_subplot(1,1,1)
cx.scatter(df['Year'],df['Psychology'])
plt.title('Sales and Age distribution')
plt.xlabel('Year')
plt.ylabel('Psychology')
plt.show()

#Gender maps analysis
df=pandas.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//Genders.csv")
print(df.head(10))

test=   df.groupby(['Gender','BMI'])
print(test.size())

#Get sample data
import numpy as np
import pandas as pd
from random import sample
df=pandas.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//Genders.csv")
# create random index
rindex = np.array(sample(range(len(df)), 5))
# get 5 random rows from df
dfr = df.ix[rindex]
print(dfr)


#Remove Duplicate Values based on values of variables "Gender" and "BMI"
import numpy as np
import pandas as pd
from random import sample
df=pandas.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//Genders.csv")
#print(df.describe()) => Gives count mean std min max n all
rem_dup=df.drop_duplicates(['Gender', 'BMI'])
print(rem_dup)
test= df.groupby(['Gender'])
print(test.describe())

print(df.isnull()) #Identify missing values from dataframe


#Example to impute missing values in Age by the mean
meanAge = np.mean(df.Age) #calculate mean
print(meanAge)
df.Age = df.Age.fillna(meanAge) # filling the missing values with mean age
print(df.Age)

