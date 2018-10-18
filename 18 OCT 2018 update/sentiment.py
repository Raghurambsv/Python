import pandas as pd
import numpy as np
import math
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

df = pd.read_csv("//users//raghuram.b//Documents//amazon//data.csv")
df=pd.DataFrame(df,columns=['Sl No.','Review title','Rating','Review Body'])


#Full csv only REVIEW BODY TEXT
text=df['Review Body'].to_string()
testimonial = TextBlob(text)
print("Full csv only REVIEW BODY TEXT",testimonial.sentiment.polarity)
#REVIEW BODY TEXT 0.22469675994657734




#Full csv only REVIEW BODY TEXT (Sentence + Mean)
list=[]
text=df['Review Body'].to_string()
testimonial = TextBlob(text)
for j in testimonial.sentences:
       list.append(j.sentiment.polarity)
print("Full csv (SENTENCE + MEAN)",np.mean(list))
#Full csv (SENTENCE + MEAN) 0.1775705292299454

row=[]
#Per Row of Body text
for i in range(0,len(df['Review Body'])):
    row.append(i)
    row_text=df['Review Body'][i]
    testimonial = TextBlob(row_text)
    df.loc[i,'polarity']=round(testimonial.sentiment.polarity,2)
    df.loc[i,'subjectivty']=round(testimonial.sentiment.subjectivity,2)
#    print("Row level polarity",i,testimonial.sentiment)


#Creating a New column SENTIMENT & sentimentvalues
df = df.assign(sentiment=df['Rating'].values)
df = df.assign(sentimentvalues=df['Rating'].values)
df.rename(columns={'Sl No.':'slno'},inplace=True)



bins = [-1,-0.4,0.25,1]
names = ['Negative', 'Neutral', 'Positive']
df['sentiment'] = pd.cut(df.polarity, bins, labels=names)

df.sentiment.fillna(value='Neutral',inplace=True)

#Assigning numeric values from sentiment column
df.loc[df.sentiment == 'Neutral','sentimentvalues'] = 0
df.loc[df.sentiment == 'Positive','sentimentvalues'] = 1
df.loc[df.sentiment == 'Negative','sentimentvalues'] = 2


sizes=df.sentiment.value_counts().to_dict().values()
labels=df.sentiment.value_counts().to_dict().keys()

df.to_csv('//users//raghuram.b//Desktop//sentiment.csv')

###############
#FULL DATA PLOT
###############
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Polarity', fontsize = 15)
ax.set_ylabel('Data', fontsize = 15)
ax.set_title('Sentiment Analysis', fontsize = 20)
#targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
targets=df.sentiment.value_counts().to_dict().keys()
colors = ['b', 'g', 'r']
for feature, color in zip(targets,colors):
    indicesToKeep = df['sentiment'] == feature
    ax.scatter(df.loc[indicesToKeep, 'polarity']
               , df.loc[indicesToKeep, 'slno']
               , c = color
               , s = 50)
ax.legend(targets)  #Label
ax.grid()
plt.show()


#######################
#Pie chart with slice
#########################
import matplotlib.pyplot as plt
 
# Data to plot
sizes=df.sentiment.value_counts().to_dict().values()
labels=df.sentiment.value_counts().to_dict().keys()
colors = ['gold','yellowgreen', 'lightcoral']
explode = (0, 0.1, 0)  # explode Neutral slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.legend( labels, loc="best")
 
plt.axis('equal')
plt.tight_layout()
plt.show()

#########################
#Doughnut with slice
#########################
import matplotlib.pyplot as plt

# create data
sizes=df.sentiment.value_counts().to_dict().values()
names=df.sentiment.value_counts().to_dict().keys()

# Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.6, color='white')

# Give color names
plt.pie(sizes, labels=names, colors = ['gold','green','coral'],wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

