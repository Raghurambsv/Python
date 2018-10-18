import matplotlib.pyplot as plt
import pandas as pd                    #http://pandas.pydata.org/pandas-docs/stable/10min.html
#https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf ==> Pandas cheatsheet
#https://jeffdelaney.me/blog/useful-snippets-in-pandas/
import numpy as np
import re
import os

###########Display options of spyder,jupyter,terminal###################
pd.show_versions() #show all installed library versions
pd.describe_option() #shows all options of HOW TO DISPLAY

pd.get_option('display.max_rows')
pd.set_option('display.max_rows',None) #unlimted display of rows
pd.reset_option('display.max_rows')

pd.get_option('display.max_columns')
pd.set_option('display.max_columns',None) #unlimted display of columns
pd.reset_option('display.max_columns')

#### In jupyter note book
If there is a function to know all the attributes u can use "Shift+Tab twice"

####################################
Canopy Data Import  ==> its a tool to generate python code from auto import CSV file
####################################
PANDAS is built on NUMPY
Dataframe is a grid like representation of data
Series is part of dataframe (like one column data is a SERIES OBJECT)
in otherwords you can say group of SERIES makes a DATAFRAME

DataFrame.at : Access a single value for a row/column label pair #df1.at[(0, 'A')]
DataFrame.loc : Access a group of rows and columns by label(s) #df1.loc[:,'col1':'col5']  or  df.loc[city =='Bengaluru','col2' :'col10']
DataFrame.iloc : Access a group of rows and columns by integer position(s)  #df1.iloc[:,:]
Dataframe['rowindex','column_name'] 
 
df.head()       # first five rows
df.tail()       # last five rows
df.sample(5)    # random sample of rows but gives 5 rows
df.describe()   # calculates measures if any numeric columns present in df
df.describe(include=['object'])  # it gives the measures of Non-Numeric columns
df.info()       # memory footprint and datatypes (it just gives the memory usage of the object references)
df.info(memory_usuage='deep') #It gives actual usuage of memory of the whole dataframe
df.memory_usage(deep=True) #gives the size consumed by indivual columns
df.memory_usage(deep=True).sum() #final mem size
df.shape        # number of rows/columns in a tuple
df.dtypes       # gives types of each column
df.columns
df.mean()
df.mean(axis=1) or df.mean(axis='columns') or df.mean(axis='index')
inplace=True  #Will make the change effective in the dataframe immediatedly its like sed -i option

######################################DATAFRAMES IN PANDAS###################################################
# create a DataFrame from a dictionary (keys become column names, values become data)
pd.DataFrame({'id':[100, 101, 102], 'color':['red', 'blue', 'red']})
f = pd.DataFrame({'a':[1,2,3,4,5], 'b':[10,20,30,40,50]})

# optionally specify the order of columns and define the index
df = pd.DataFrame({'id':[100, 101, 102], 'color':['red', 'blue', 'red']}, columns=['id', 'color'], index=['a', 'b', 'c'])
df

# create a DataFrame from a list of lists (each inner list becomes a row)
pd.DataFrame([[100, 'red'], [101, 'blue'], [102, 'red']], columns=['id', 'color'])

# create a NumPy array (with shape 4 by 2) and fill it with random numbers between 0 and 1(by default it takes 0 to 1)
import numpy as np
arr = np.random.rand(4, 2)
arr

# create a DataFrame from the NumPy array
pd.DataFrame(arr, columns=['one', 'two'])

# create a DataFrame of student IDs (100 through 109) and test scores (random integers between 60 and 100 and 10 off them)
pd.DataFrame({'student':np.arange(100, 110, 1), 'test':np.random.randint(60, 101, 10)})


# 'set_index' can be chained with the DataFrame constructor to select an index
pd.DataFrame({'student':np.arange(100, 110, 1), 'test':np.random.randint(60, 101, 10)}).set_index('student')

# create a new Series using the Series constructor
s = pd.Series(['round', 'square'], index=['c', 'b'], name='shape')
s

# concatenate the DataFrame and the Series (use axis=1 to concatenate columns)
pd.concat([df, s], axis=1)


######to access data using the default or customised INDEX####
drinks.iloc[drinks.index.get_loc(4),1] #default index

drinks.set_index('country',inplace=True) #set index to country
drinks.iloc[drinks.index.get_loc('Angola'),1] #Access by giving index name



###############Read into pandas###########

df=pd.read_csv("//users//raghuram.b//Desktop//result.txt",delimiter=',')
df = pd.DataFrame(df,columns=['COLUMN_NAME','DATA_TYPE','TITLE'])

df = pd.read_fwf('//users//raghuram.b//Desktop//result.txt', header=None,na_values=[])
df=df.replace(to_replace=r'\t', value=' ', regex=True)
df=df.split()

df = pd.read_table('//users//raghuram.b//Desktop//result.txt',  header=None,skiprows=(0))
df.head()
         
df=pd.read_csv("//users//raghuram.b//Desktop//result.txt",delimiter='\n',keep_default_na=False)

use_Cols=['A','B','C','D']
df=pd.read_csv("//users//raghuram.b//Desktop//result.txt",delimiter='\n',names=Use_Cols,header=None) ##To rename columns while importing

####################################

Count the total number of NaNs present:
df.isnull().sum()   #NaN Indviual columns
df.isnull().sum().sum() # sum of all Nan in all columns present


var="hello.raghu.vineeth.ritu"
result=var.split('.',2)
print(result) ##to split only one first DOT

s = pd.Series([1,3,5,np.nan,6,8])
print(s)

#Numpy array with DATETIME index
dates = pd.date_range('20130101', periods=6 ,freq='D')
print(dates)

#DataFrame wit columns and index info
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
print(df)


#Creating a DataFrame by passing a dict of objects that can be converted to series-like.

dict_a = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
#                     'C' : (pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })
print(dict_a)

df=pd.DataFrame()
df['hastags']=dict_a.keys()
df['count']=dict_a.values()

df2.keys()
df2.items()
df2.values()


>>> data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}
>>> pd.DataFrame.from_dict(data)
   col_1 col_2
0      3     a
1      2     b
2      1     c
3      0     d

>>> data = {'row_1': [3, 2, 1, 0], 'row_2': ['a', 'b', 'c', 'd']}
>>> pd.DataFrame.from_dict(data, orient='index')
       0  1  2  3
row_1  3  2  1  0
row_2  a  b  c  d

>>> pd.DataFrame.from_dict(data, orient='index',
...                        columns=['A', 'B', 'C', 'D'])
       A  B  C  D
row_1  3  2  1  0
row_2  a  b  c  d




data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'year': [2012, 2012, 2013, 2014, 2014], 
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])


sum([len(x) for x in list_a]) #No of characters in a List

###remove duplicates from list
from  more_itertools import unique_everseen
items = [1, 2, 0, 1, 3, 2]
list(unique_everseen(finallist)) 

###Unifying two datasets
df = pd.DataFrame({'A': [0, 1],
'B': [5, 6],
'C': ['a', 'b']})
df1 = pd.DataFrame({'D': [0, 0], 'E': [4, 4], 'F': [99, 95]})
df=df.join(df1)  

#make new column from existing ones

df = pd.DataFrame({'A': [0, 1],
'B': [5, 6],
'C': ['a', 'b']})

df['D']= df.A + df.B 

print(df)

#To choose only alphabets 
str = '1208uds9f8sdf978qh39h9i#H(&#*H(&H97dgh'
result = ''.join(c for c in str if c.isalpha())
print(result)

df.columns = df.columns.str.replace(' ','_') # to replace all the NO_NAME columns with UNDERSCORE

new_list_without_comma = [x.replace(",", "") for x in list_with_comma]

Apply function
###############

Count = df['fruits'].str.split().apply(len).value_counts

bulb_temp_df['A'].map(lambda x: x.rstrip('aAbBcC'))

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0]) #Just for a function you can give a single column of dataframe filtered by a value
#hashtag_extract is a function for which your passing a dataframe column

#######Applying Function for pandas column
valid = '1234567890.' #valid characters for a float
def sanitize(data):
    return float(''.join(filter(lambda char: char in valid, data)))
df['column'] = df['column'].apply(sanitize)


df.apply(np.cumsum) #cumulative sum across columns
df.apply(lambda x: x.max() - x.min())


######Replace and extract

df = pd.DataFrame({'A': [0, 1, 2, 3, 4],
'B': [5, 6, 7, 8, 9],
'C': ['a', 'b', 'c', 'd', 'e']})

   
df.replace(0, 5)    #Convert whereever Zero to Five  

df.replace([0, 1, 2, 3,4], [4, 3, 2, 1,99]) # Convert full column 'A'

df = pd.DataFrame({'A': ['bat', 'foo', 'bait'],'B': ['abc', 'bar', 'xyz']}) #Converion with regex
df.replace(to_replace=r'^ba.$', value='new', regex=True)
or
df.replace(regex={r'^ba.$':'new', 'foo':'xyz'})

"(\d+\.\d+|\*|NaN)" ##choose digit.digit or Nan

df['A1'].replace(regex=True, inplace=True, to_replace=r'[^0-9.\-]', value=r'']

df["column"] = df["column"].str.replace(",","").astype(float)

a['Name'] = a['Name'].str.replace('\d+', '')

df.Name = df.Name.str.replace('\d+', '')

df['code'].str.extract('(\d*\.?\d*)', expand=False).astype(float)

#####CALCULATING NULL VALUES THROUGH OUT DATAFRAME
df = pd.read_csv('/Users/raghuram.b/Desktop/FreshPython/train.csv')
#Create a new function:
def num_missing(x):
    return sum(x.isnull())

#Applying per column:
print ("Missing values per column:")
print (df.apply(num_missing, axis=0))  #axis=0 defines that function is to be applied on each column

#Applying per row:
print ("\nMissing values per row:")
print (df.apply(num_missing, axis=1).head())  #axis=1 defines that function is to be applied on each column

####IMPUTING (picking the most common value)
df = pd.read_csv('/Users/raghuram.b/Desktop/FreshPython/train.csv')
#First we import a function to determine the mode
from scipy.stats import mode
mode(df['Age']).mode[0]

#Impute the values:
df['Age'].fillna(mode(df['Age']).mode[0], inplace=True)

#Create a new function:
def num_missing(x):
    return sum(x.isnull())

print (df.apply(num_missing, axis=0))



####CROSS TAB
import pandas as pd
data = pd.read_csv('/Users/raghuram.b/Desktop/FreshPython/train.csv')
#Cross tab of function with totals
pd.crosstab(data["Survived"],data["Sex"],margins=True)

#Cross tab of function with totals ka percentage
def percConvert(ser):
    return ser/float(ser[-1])
pd.crosstab(data["Survived"],data["Sex"],margins=True).apply(percConvert, axis=1)

pd.crosstab(data["Survived"],data["Sex"],data["Age"],margins=True,aggfunc=[np.mean])


############TO CONVERT LIST ITEMS INTO PANDA using DATAFRAME command####
df=pd.DataFrame(q_list, columns=['q_data'])


#Convert DF to List    
df = pd.DataFrame({'a':[1,3,5,7,4,5,6,4,7,8,9],
                   'b':[3,5,6,2,4,6,7,8,7,8,9]})
df['a'].values.tolist()
df['a'].tolist()
df['a'].drop_duplicates().values.tolist()


#Duplicates
# read a dataset of movie reviewers (modifying the default parameter values for read_table)
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table('http://bit.ly/movieusers', sep='|', header=None, names=user_cols,index_col='user_id')


users.duplicated().sum() #entire users dataframe duplicate records
users.loc[users.duplicated(keep='first'),:]
users.loc[users.duplicated(keep='last'),:]
users.loc[users.duplicated(keep=False),:] #All duplicate records (including the unique one)


users.zip_code.duplicated().sum() #Total count of duplicates only for zipcode
users.loc[users.zip_code.duplicated(),:].shape #print out all the duplicate rows

users.drop_duplicates().shape #drops duplicates and keeps one version of it
users.drop_duplicates(keep=False).shape #drops duplicates and drops its original version also



#To convert DF to STRING
result = ''.join([i for i in df['Name'][1] if not i.isdigit()])


#Conditional Operations inside DF
df.sort_values('A').loc[lambda df: df.A > 80].loc[lambda df: df.B > df.A]

df[(df['coverage']  > 50) & (df['reports'] < 4)]
var=df[(~df['Review Body'].str.contains('cartridge') & df['Review Body'].str.contains('ink'))] #222

#Conditionals on pandas columns
#Conditionals on pandas columns
polarity=df['polarity']
df['sentiment'][polarity > 0.25]='Positive'
df['sentiment'][polarity < -0.25]='Negative'

df['sentiment'] = np.where(df['polarity']>=0, 'yes', 'no')

df['data'].apply(lambda x: 'true' if x <= 2.5 else 'false')
or
condn=df['polarity'] > 0
df['sentiment'][condn]='Positive'
df['sentiment'][~condn]='Negative'

or
#For each row
group_df['Age'] = group_df.apply(lambda row:defect_age(row), axis=1)

data.loc[(data["Gender"]=="Female") & (data["Education"]=="Not Graduate") & (data["Loan_Status"]=="Y"), ["Gender","Education","Loan_Status"]]

import pandas as pd
import numpy as np
df = pd.DataFrame(data={'portion':np.arange(10000), 'used':np.random.rand(10000)})

%%timeit
df.loc[df['used'] == 1.0, 'alert'] = 'Full'
df.loc[df['used'] == 0.0, 'alert'] = 'Empty'
df.loc[(df['used'] >0.0) & (df['used'] < 1.0), 'alert'] = 'Partial'


# DF from string and functions
from io import StringIO
import re
import numpy as np
import pandas as pd

s = StringIO('''\
       A           B
1   10.1        33.3
2   11.2       44.2s
3   12.3       11.3s
4   14.2s          *
5   15.4s        nan
''')


df = pd.read_table(s, sep='\s\s+',engine='python')
df['A'] = df['A'].astype(str)
df['B'] = df['B'].astype(str)

df = df.applymap(lambda x: re.sub(r'[^0-9^\-\.]+', '', x)).replace('', np.float64(0)).astype('float64') 

print(df)

##FUNCTION FOR FULL DATASET
#Create a new function:
def num_missing(x):
    return sum(x.isnull())

#Applying per column:
print "Missing values per column:"
print data.apply(num_missing, axis=0)  #axis=0 defines that function is to be applied on each column

#Applying per row:
print "\nMissing values per row:"
print data.apply(num_missing, axis=1).head()  #axis=1 defines that function is to be applied on each column


#Imputing missing files

############String to Pandas SERIES object##############
str="hello.raghu.vineeth.ritu"
list=[]
for x in str.split('.'):
    list.append(x)


df=pd.Series(list)
print(df)
############TO CONVERT LIST ITEMS INTO STRING through JOIN METHOD##############
list=['shown', 'drawing', 'drawing', 'diameters', 'Please', 'supply', 
'redline', 'drawing', 'D', 'MU', 'Rev', 'B', 'The', 'component', 
'drawing', 'top', 'level', 'drawing', 'attached', 'review', 'The', 'part',
 'assembly', 'consists', 'following', 'parts', 'PT', 'Breaker', 'Housing',
 'DWG', 'D', 'MU', 'Rev', 'B', 'PT', 'Internal', 'Key', 'DWG', 'D', 'Rev',
 'A', 'PT', 'Capscr', 'SOCK', 'X', 'STL', 'DWG', 'REVISION', 'UNKNOWN', 'Part']

print(list)

result = ' '.join(i for i in list )
print(result)

# unnesting list ( Making a Nested list of values into single strings)
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


#To find out the respective types
print(df2.dtypes)

#In IPythonConsole
#df2.<TAB>
#df2.A                  df2.bool
#df2.abs                df2.boxplot
#df2.add                df2.C
#df2.add_prefix         df2.clip
#df2.add_suffix         df2.clip_lower
#df2.align              df2.clip_upper
#df2.all                df2.columns
#df2.any                df2.combine
#df2.append             df2.combine_first
#df2.apply              df2.compound
#df2.applymap           df2.consolidate
#df2.D

############Using PANDAS and calling function from lambda##############
import pandas as pd
movies= pd.read_csv('http://bit.ly/imdbratings')
movies.shape


Long_Movies=[]
for length in movies.duration:
    if length >= 200:
        Long_Movies.append(True)
    else:
        Long_Movies.append(False)
        
is_long=pd.Series(Long_Movies)   
is_long.head()
movies[is_long]     
    
or

var1=movies.duration >= 200
var1.head() #It converts the full column into Boolean values TRUE or FALSE ...TRUE for if the DURATION IS GREATER THAN 200
movies[var1] #remember not using quotes for VAR coz its a list of movie >=200 mins

or

movies[movies.duration >= 200] # It gives the only those rows which the DURATION IS GREATER THAN 200

movies[movies.duration >= 200].genre
or
movies[movies.duration >= 200]['genre'] ##Gives only the column genre not whole dataset
or
movies.loc[movies.duration >= 200,'genre']  ##Loc also gives same output...its best practise coz its more efficient way 

movies.dtypes
movies.genre.describe() #gives count,unique values,max movie ,freq
movies.genre.value_counts()
movies.genre.value_counts(normalize=True)#gives the percentage of values
movies.genre.unique()
movies.genre.nunique()#get the no of unique values

pd.crosstab(movies.genre,movies.content_rating) #gives crosstab analysis

%matplotlib inline
movies.duration.plot(kind='hist')
movies.duration.plot(kind='bar')

import pandas as pd
import re
from textblob import TextBlob
def sentiment_calc(text):
    try:
        return TextBlob(text).correct()
    except:
        return None

df=df.apply(sentiment_calc) #For checking grammer 

print(df) 



df.head() #by default gives top 5 rows
df.head(50).drop('col1',axis=1) #you can use head to limit the columns u want
df.tail(3)#last 3 rows
df.index
df.columns# column names of the dataframe
df.values #values of the dataframe

#describe() shows a quick statistic summary of your data:
df.describe()

#Transposing your data:
df.T

#Sorting by an axis: (sort it by column labels)
df.sort_index(axis=1, ascending=True)

#Sorting by values/columns
df.sort_values(by='B',ascending=False)
data_sorted = data.sort_values(['ApplicantIncome','CoapplicantIncome'], ascending=False)
data_sorted[['ApplicantIncome','CoapplicantIncome']].head(10)

#Indivual column
df['A']

#selecting range of rows
df[0:3] #first 3 rows   also SAME df['20130102':'20130104'] (coz dates are defined as index)

#For getting a cross section using a label:
df.loc[dates[0]] #first row of df

#Selecting on a multi-axis by label:
df.loc[:,['A','B']]  #all rows of column 'A' and 'B'
df.loc[df.city=='blore','Chennai',:] #only rows with CITY == BLORE and Chennai
drinks[drinks.continent == 'Australia'] #just another example

df.set_index('col1',inplace=True) #To set country has the row index

df.index.name='country'
df.reset_index(inplace=True)

#Showing label slicing, both endpoints are included:
df.loc['20130102':'20130104',['A','B']] #choosing which rows needed

#scalar value or 1row,1col
df.loc[dates[0],'A']

#For getting fast access to a scalar ( same as above but not sure why its fast)
df.at[dates[0],'A']

#Select via the position of the passed integers:
df.iloc[3] # 4th row ( coz referencing single value it starts from '0' so '3 means 4th row')

#By integer slices,
df.iloc[3:5,0:2] #(when you refer in slices ....upperbound value is nothing but upperbound_value-1)

#By lists of integer position locations
df.iloc[[1,2,4],[0,2]]

#For slicing rows explicitly but listing all columns
df.iloc[1:3,:]

#For slicing cols explicitly but listing all rows
df.iloc[:,1:3]

#For getting a value explicitly: 
df.iloc[1,1] #or df.loc[dates[1],'B']

#Boolean Indexing
df[df.A > 0] #means all the values in COL 'A' which are greater than zero
df[df > 0] #means all the values in the df_MATRIX which are greater than zero

#Using the isin() method for filtering:
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three'] #Add extra 'E' column
df2[df2['E'].isin(['two','four'])] #select matrix rows based on 'E' column value


drinks=pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()
drinks.set_index('country',inplace=True)
drinks.head()

drinks.continent.value_counts()
drinks.continent.value_counts().values
drinks.continent.value_counts()['Africa']
drinks.continent.value_counts().sort_values() #sort by values
drinks.continent.value_counts().sort_index() #sort by index (country is the index now)

#New dataframe or series creation
people = pd.Series([300000,85000],index=['Albania','Andorra'],name='population')
print(people)

#take 2 diff dataframes and multiply
drinks.beer_servings * people #It automatically does by using index (which is country in both the dataframes)

#to put the result in one common dataframe
pd.concat([drinks,people],axis=1)




#Setting a new column automatically aligns the data by the indexes.
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
print(s1)
print(df)
df['F'] = s1 #add column F
print(df)

#Setting values by label:
df.at[dates[0],'A'] = 0

#Setting values by position:
df.iat[0,2] = 0

#Setting by assigning with a NumPy array:
df.loc[:,'D'] = np.array([5] * len(df))

#where operation with setting.
df2 = df.copy()
df2[df2 > 0] = -df2 #convert all the positive values into negative
df2

#Reindexing allows you to change/add/delete the index on a specified axis.
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])

#assign 10 value to first 2 rows of 'E'
df1[dates[0]:dates[1]]=10

#drop any row which has 'NaN' in any of its cols
df1.dropna(how='any')
df1.dropna(how='any').shape #need to give inplace to actual reflect

#drop any row which has 'NaN' in ALL of its cols
df1.dropna(how='all')

#subset dropna
df1.dropna(subset=['City','Shape Reported'],how='all')

#Fill all NaN values to 5
df1.fillna(value=5)

#boolean mask for all the NaN values of df1
pd.isna(df1)

s = pd.Series(np.random.randint(0, 7, size=10))
print(s)

s.value_counts()


###Deleting a column
del df['column_name'] 
or 
del df.column_name
 or
df.drop('column_name',axis=1,inplace=True) #here Axis=1 means vertical ......Axis =0 means horizontal

###Deleting a Row
df.drop([0,1],axis=0,inplace=True) #Axis =0 means horizontal (Deleting the first 2 rows by using by default index= 0 and 1)

##Rename a column
df.rename(columns={'old_col_name':'new_col_name'},inplace=True})


 #################UFO and SAMPLE method################################   
import pandas as pd
ufo=pd.read_csv('http://bit.ly/uforeports')

ufo.sample(n=5)# random sample of rows but gives 5 rows (each time u run ...diff 5 rows)
ufo.sample(n=5,random_state=32) #gives random 5 rows but wen you give random_state=<number>...how much ever times u run you get same 5 rows

Training_Set=ufo.sample(frac=0.75,random_state=99)#gives 75% of ufo dataset

Testing_set=ufo.loc[~ufo.index.isin(Training_Set.index),:] #to generate the rest 25% as Testing_Set


######STRING METHODS###################
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()

'HELLO'.lower()
orders = pd.read_table('http://bit.ly/chiporders')
orders.head()
orders.item_name.str.upper()

orders.item_name.str.contains('Chicken')  #Gives in boolean True/False
orders[orders.item_name.str.contains('Chicken')] #The above boolean you can put inside the dataframe orders[<above result>] to get only those
                                                 #rows to be displayed
#To convert boolean to normal ZERO and ONE
orders.item_name.str.contains('Chicken').astype(int)     
index=df[df['Review Body'].str.contains('print')].index.values
index=df[df['Review Body'].str.contains(r'\w\sprint')].index.values
                                            

orders.choice_description.str.replace('[','') #To remove all '[' braces]                                                
orders.choice_description.str.replace(']','') #To remove all ']' braces] 

orders.choice_description.str.replace('[','').str.replace(']','') #combine above both  in one line   
orders.choice_description.str.replace('[\[\]]','') #same as above but using regexpression                                          



###Change datatypes in pandas
drinks=pd.read_csv('http://bit.ly/drinksbycountry')
drinks.dtypes
drinks['beer_servings']=drinks.beer_servings.astype(float) #convert beer_servings from INT to FLOAT
drinks.dtypes

drinks=pd.read_csv('http://bit.ly/drinksbycountry',dtype={'beer_servings':float}) #Doing conversion while reading from file directly
drinks.dtypes


orders = pd.read_table('http://bit.ly/chiporders')
orders.head()  #Item price is $2.39 
orders.item_price.str.replace('$','').astype(float) #( so need to remove dollar sign and change the datatype)



###Concat
df = pd.DataFrame(np.random.randn(10, 4))

# break it into pieces
pieces = [df[:3], df[3:7], df[7:]]
print(pieces)

pd.concat(pieces)

###pandas series to DATAFRAME 
import pandas as pd
data = {'a': 1, 'b': 2}
pd.Series(data).to_frame()

#####Join
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

###Merge
pd.merge(left,right,on='key')


#Merging
df1=pandas.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//Genders.csv")
df2=pandas.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//Genders.csv")
meanAge = np.mean(df1.Age)
df2.Age = df1.Age.fillna(meanAge)

df_new = pd.merge(df1, df2, how = 'inner', left_index = True, right_index = True) 
print(df_new)

####Join
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
pd.merge(left, right, on='key')




###Append
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
print(df)
s = df.iloc[3]
print(s)
df.append(s, ignore_index=True)


###Combining with conditionals
df1 = pd.DataFrame({'A': [0, 0], 'B': [4, 4]})
df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
df1.combine(df2, lambda s1, s2: s1 if s1.sum() < s2.sum() else s2)

 
df1 = pd.DataFrame([[1, np.nan]])
df2 = pd.DataFrame([[3, 4]])
df1.combine_first(df2)


####Grouping
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                    'D' : np.random.randn(8)})
print(df)

df.groupby('A').sum()

df.groupby(['A','B']).sum()

drinks=pd.read_csv('http://bit.ly/drinksbycountry')
drinks.beer_servings.mean()
drinks.groupby('continent').beer_servings.mean() #now u get it country wise
drinks[drinks.continent == 'Africa'].beer_servings.mean()
drinks.groupby('continent').beer_servings.agg(['count','min','max','mean']) #get all agg funcs at once
drinks.groupby('continent').mean() #for all columns

%matplotlib inline
drinks.groupby('continent').mean().plot(kind='bar') #plot graph too with groupby

#stack
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz','foo', 'foo', 'qux', 'qux'],['one', 'two', 'one', 'two','one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
print(index)

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
print(df)

df2 = df[:4]

print(df2)

#The stack() method “compresses” a level in the DataFrame’s columns.
stacked = df2.stack()
print(stacked)

#With a “stacked” DataFrame or Series (having a MultiIndex as the index), 
#the inverse operation of stack() is unstack(), which by default unstacks the last level:
stacked.unstack()
stacked.unstack(1)
stacked.unstack(0)


###Pivot tables
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                   'B' : ['A', 'B', 'C'] * 4,
                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D' : np.random.randn(12),'E' : np.random.randn(12)})

df.set_index(['A','C'])   ##Having multiple index instead of one
 
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])


##Timeseries
rng = pd.date_range('1/1/2012', periods=100, freq='S')
print(rng)

ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts)

ts.resample('5Min').sum()

#Time zone representation:
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
print(rng)

ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts)

ts_utc = ts.tz_localize('UTC')
print(ts_utc)

#Converting to another time zone:
ts_utc.tz_convert('US/Eastern')


#Date Timestamp
import pandas as pd
ufo=pd.read_csv('http://bit.ly/uforeports')
ufo.head()
ufo.dtypes
ufo.Time.str.slice(-5,-3).astype(int).head()

#or
ufo['Time']= pd.to_datetime(ufo.Time)
ufo.head()
ufo.dtypes #Its become datetime object
ufo['Time']=pd.to_datetime(ufo.Time)
ufo.Time.dt.hour
ufo.Time.dt.weekday_name
ufo.Time.dt.dayofyear
ufo.Time.max() - ufo.Time.min()

ts=pd.to_datetime('1/1/1999')
ufo.loc[ufo.Time > ts,:]

%matplotlib inline
ufo['year']=ufo.Time.dt.year #creating new column YEAR
ufo.year.value_counts().sort_index().plot()


#####Converting between period and timestamp
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
print(prng)

ts = pd.Series(np.random.randn(len(prng)), prng)
print(ts)

ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9

ts.head()

####Categoricals
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
print(df)

df["grade"] = df["raw_grade"].astype("category")
print(df["grade"])


####Rename the categories to more meaningful names (assigning to Series.cat.categories is inplace!).
df["grade"].cat.categories = ["very good", "good", "very bad"]
print(df["grade"])

###Reorder the categories and simultaneously add the missing categories (methods under Series .cat return a new Series by default).
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
print(df["grade"])


#Sorting is per order in the categories, not lexical order.
df.sort_values(by="grade")
or
df.Column_Name.sort_values()
or
df[Column_Name].sort_values(ascending=False)


drinks=pd.read_csv('http://bit.ly/drinksbycountry')
sorted(drinks.continent.unique()) #unique elements in continents column


#CATEGORICAL ==>  memory efficient 
            #(its used like lookup table..it saves memory...string objects consume more memory than int)
            #so will keep string lookup table and the values is stored as integers internally
            #this python automatically does by using category method
            
drinks.memory_usage(deep=True) #currently continent uses 12332 bytes
drinks.continent.memory_usage(deep=True)
drinks['continent']=drinks.continent.astype('category')
drinks.dtypes#it shows now the continent is become category type

drinks.continent.cat.codes.head()  #use cat for category object..now the values are stored as int not strings
drinks.continent.memory_usage(deep=True) #current memory usage is reduced to 824

#####Stack & Unstack ( is used to convert mulitple index into one index)
drinks.groupby('continent').beer_servings.describe()
drinks.groupby('continent').beer_servings.describe().unstack()
drinks.groupby('continent').beer_servings.describe().unstack().stack()
            

#If you need to sort by more than one column
df.sort_values(['A','B'])  #It sorts by first A and then B


#Grouping by a categorical column also shows empty categories.
df.groupby("grade").size()


###MAP (SERIES OBJECT)
train=  pd.read_csv('http://bit.ly/kaggletrain')
train.head()
train['sex_num']=train.Sex.map({'female' : 0,'male' :1}) # Map is like decode function

train.loc[0:4,['Sex','sex_num']]  #by default sex_num will be present in dataframe temp...you can use it at your will

####APPLY (SERIES or DATAFRAME object)
train=  pd.read_csv('http://bit.ly/kaggletrain')
drinks=pd.read_csv('http://bit.ly/drinksbycountry')
train.head()
train['Name_length'] = train.Name.apply(len)
train.loc[0:4,['Name','Name_length']] 

train.Name.str.split(',').head()
train.Name.str.split(',').apply(lambda x: x[0]).head() #Print only the last name (before comma)

drinks.loc[:,'beer_servings':'wine_servings'].apply(max,axis=0)
drinks.loc[:,'beer_servings':'wine_servings'].apply(max,axis=1)

###APPLYMAP (is applied for every element of DATAFRAME(not just one series of col or etc))
drinks=pd.read_csv('http://bit.ly/drinksbycountry')
drinks.loc[:,'beer_servings':'wine_servings'].applymap(float) #converts all INTEGER COLUMNS to float....gives error for string columns



#######################PLOTTING########################
import matplotlib.pyplot as plt
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
print(ts)

ts = ts.cumsum()
ts.plot()


#On a DataFrame, the plot() method is a convenience to plot all of the columns with labels:
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure(); df.plot(); plt.legend(loc='best')

import pandas as pd
data = pd.read_csv('/Users/raghuram.b/Desktop/FreshPython/train.csv')
import matplotlib.pyplot as plt
data.boxplot(column="Survived",by="Sex")
data.hist(column="Survived",by="Sex",bins=20)

####################Convert DF to CSV########################
df.to_csv('foo.csv')





#####Analytics vidya examples


df=pd.read_csv("//users//raghuram.b//Desktop//Savita Projects//Rough worksheets//house.csv",delimiter='\t')
house=pd.DataFrame(df,columns=['House','Region'])
print(house)

df1=pd.read_csv("//users//raghuram.b//Desktop//Savita Projects//Rough worksheets//house_append.csv",delimiter='\t')
house_extra=pd.DataFrame(df1,columns=['House','Region'])
print(house_append)

#Add two datasets (full set of rows)
house=house.append(house_extra)
print(house)

#or
house=pd.concat([house,house_extra],axis=0) #(full set of rows...its diff than merge based on column)
print(house)


#Add new row directly
house.loc[len(house)]=['Redwyne','The Reach']

#combine new dataset of house_new to house
df2=pd.read_csv("//users//raghuram.b//Desktop//Savita Projects//Rough worksheets//house_new.csv",delimiter='\t')
house_new=pd.DataFrame(df2,columns=['House','Region','Religion'])
house=pd.concat([house,house_new],axis=0,sort=True)

#From above combinations there is index repeatation
house=pd.concat([house,house_new],axis=0,ignore_index=True) 

#add new column from new dataset military
df3=pd.read_csv("//users//raghuram.b//Desktop//Savita Projects//Rough worksheets//Military.csv")
military=pd.DataFrame(df3,columns=['Military_Strength'])
house_military=pd.concat([house,military],axis=1)

sLength = len(df)
df['sentiment'] = pd.Series(np.random.randn(sLength), index=df.index)
or
df = df.assign(NewColumnName=df['col1'].values)
or
df['new_column'] = 23 #Every row will have value 23

##Delete a column
del house_military['Religion']

house_military=house_military.dropna(how='any')
house=house_military

df4=pd.read_csv("//users//raghuram.b//Desktop//Savita Projects//Rough worksheets//Candidates.csv",delimiter='\t')
candidates=pd.DataFrame(df4,columns=['House','Name'])

#Merge to get info fro diff datasets but common column
house = pd.merge(candidates,house,on="House",how="left")

#house = pd.merge(candidates,house,left_on="House",right_on="House",how="left")   ==> if name of columns is diff in both datasets

#inner join
house = pd.merge(candidates,house,on="House",how="inner")

#Merging 2 datasets
house = candidates.join(house,how='left',on='House') #By defaut it adds its own suffix like <column name>_X and <column name>_Y

#Merge 2 datasets with custom suffix
house_merged = pd.merge(candidates,house, on='House', how='left', suffixes=('_left', '_right'))
house_merged.drop_duplicates(subset=['House'],keep='first',inplace=True)

#modify values of Datframe (make Robb Stark from North==>south)
house_merged[house_merged.Name == 'Robb Stark']=house_merged[house_merged.Name == 'Robb Stark'].replace("North","xxxxx")
#modify everywhere
house_merged.House.str.replace("a","xxx")

house_merged.iloc[:,[1,6]]

house.replace(to_replace=r'^.*a.*$', value='new', regex=True)



#Need the column in a certain order? The first argument is the position of the column. This will put the column at the begining of the DataFrame.
#Position of column
df.insert(0, 'original_price', full_price)

#conditionals or if else
filtered_data = df[(df.price > 11.99) & (df.topping == 'Pineapple')]

#Mean or std of columns
df['mean'] = df.mean(axis=1)
df.std(axis=0)

# allow plots to appear within the notebook
%matplotlib inline

# visualize the relationship between the features and the response using scatterplots
#plot SEPERATE SCATTER PLOTS TO EACH FEATURES(3 GRAPHS)
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')


####MODIFING/CREATING a new column from existing column
def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)
    return colCoded

#Coding LoanStatus as Y=1, N=0:
print 'Before Coding:'
print pd.value_counts(data["Loan_Status"])
data["Loan_Status_Coded"] = coding(data["Loan_Status"], {'N':0,'Y':1})
print '\nAfter Coding:'
print pd.value_counts(data["Loan_Status_Coded"])
Before Coding:
Y    422
N    192
Name: Loan_Status, dtype: int64

After Coding:
1    422
0    192
Name: Loan_Status_Coded, dtype: int64


##Multi class to binary/two class function
convert = lambda x:1 if x>3 else 0
sf_train['target'] = sf_train['Sentiment'].apply(convert)