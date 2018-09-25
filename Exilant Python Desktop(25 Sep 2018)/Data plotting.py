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

from bokeh.plotting import figure
from bokeh.io import  output_file,show

#prepare some data
x=[1,2,3,4,5]
y=[6,7,8,9,10]

#prepare the output file
output_file("Line.html")

#create a figure object
f=figure()

#create a line plot
f.line(x,y)

#display the plot in the figure object
show(f)


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


#Merging
df1=pandas.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//Genders.csv")
df2=pandas.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//Genders.csv")
meanAge = np.mean(df1.Age)
df2.Age = df1.Age.fillna(meanAge)

df_new = pd.merge(df1, df2, how = 'inner', left_index = True, right_index = True) 
print(df_new)


#	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
#	plt.xlabel('Petal length')
#	plt.ylabel('Petal width')
#	plt.title('Petal Width & Length')
#	plt.show()
 


#
#iris = datasets.load_iris()
#X = iris.data[:, :2]  # we only take the Sepal two features.
#y = iris.target
#C = 1.0  # SVM regularization parameter
# 
## SVC with linear kernel
#svc = svm.SVC(kernel='linear', C=C).fit(X, y)
## LinearSVC (linear kernel)
#lin_svc = svm.LinearSVC(C=C).fit(X, y)
## SVC with RBF kernel
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
## SVC with polynomial (degree 3) kernel
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)



# Packages for creating the graphs
#import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.graph_objs import *
#py.sign_in(YOUR_USER_NAME, YOUR_API_KEY)
# 
# 
#def scatter_graph(x, y, graph_title, x_axis_title, y_axis_title):
#    """
#    Scatter Graph
#    :param x: 
#    :param y: 
#    :param graph_title: 
#    :param x_axis_title: 
#    :param y_axis_title: 
#    :return: 
#    """
#    trace = go.Scatter(
#        x=x,
#        y=y,
#        mode='markers'
#    )
#    layout = go.Layout(
#        title=graph_title,
#        xaxis=dict(title=x_axis_title), yaxis=dict(title=y_axis_title)
#    )
#    data = [trace]
#    fig = Figure(data=data, layout=layout)
#    plot_url = py.plot(fig)
#    
#input_path = '../Inputs/input_data.csv'
#house_price_dataset = pd.read_csv(input_path)
#scatter_graph(house_price_dataset['square_feet'], house_price_dataset['price'],
#              'Square_feet Vs Price', 'Square Feet', 'Price')




