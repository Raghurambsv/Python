import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)


train  = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')

train.head()

combi=train.append(test,ignore_index=True,sort=False)


#Remove '@user' from the tweets
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 

#The lenght of the text
sum([len(x) for x in combi['tweet']])   

combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*") #Vecotrise func is used to call func with parameters ( just like apply map)


# remove special characters, numbers, punctuations
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

#The lenght of the text
sum([len(x) for x in combi['tweet']]) 


#Remove all words with size less than 3
combi['tidy_tweet']=combi['tidy_tweet'].apply(lambda x:  ' '.join([w for w in x.split() if len(w) > 3]))

#The lenght of the text
sum([len(x) for x in combi['tweet']]) 

#Tokenization (Splitting the string into indivual words)....having list with words seperated for every row
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()


#Stemming (its used for stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word)
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet=tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x])

 

#stitch these tokens back together.
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]= ' '.join(tokenized_tweet[i])
    
combi['tidy_tweet'] = tokenized_tweet    

all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()    

#Words in non racist/sexist tweets or (words in the label '0')
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
    
#Words in Racist/Sexist tweets or (words in the label '1')
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
    
# function to collect hashtags (derive meaning from hashtags)
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0]) #Just for a function you can give a single column of dataframe

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

#Non-Racist/Sexist Tweets top 10 words
a = nltk.FreqDist(HT_regular)  #gives the WORD and its COUNT OF REPETITIONS in dictionary format

#Create dataframe for dictionary
d=pd.DataFrame()
d['Hashtag']=a.keys()
d['Count']=a.values()

# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#Racist/Sexist Tweets top 10 words
b = nltk.FreqDist(HT_negative)
e=pd.DataFrame()
e['Hashtag']=b.keys()
e['Count']=b.values()
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#Just example for One-GRAM and BI-Grams
#ONEgram_vectorizer = CountVectorizer(stop_words='english',token_pattern=r'\b\w+\b', min_df=1)
#analyze = ONEgram_vectorizer.build_analyzer()
#analyze('Bi-grams are cool!') 
#
#bigram_vectorizer = CountVectorizer(ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)
#analyze = bigram_vectorizer.build_analyzer()
#analyze('Bi-grams are cool!') 


#BAG OF WORDS- frequency of words in that line
# [Convert text to numbers matrix]( Using SKLEARN CountVecotorizer)
#It takes into account the count of words present in each line/document
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english',lowercase=True)
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
#print(bow_vectorizer.get_feature_names())
#print(bow.toarray()) 
#print(bow_vectorizer.get_stop_words())
sum([len(x) for x in bow_vectorizer.get_feature_names()])#length of characters of eature_names





#TF-IDF-frequency of words in entire file
#TF-IDF[Convert text to numbers matrix]
#TF-IDF works by penalizing the common words by assigning them lower weights while giving importance to words which are rare in the entire corpus but appear in good numbers in few documents.
#It gives less weightage to common words and more weightage(to words rare but present more in one or few docs)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english',lowercase=True)
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
sum([len(x) for x in tfidf_vectorizer.get_feature_names()]) #length of characters of feature_names


#Logistic regression using BAG OF WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain)

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid, prediction_int) # calculating f1 score of BAG OF WORDS




#Logistic regression using TFIDF
train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)# calculating f1 score of TF-IDF





