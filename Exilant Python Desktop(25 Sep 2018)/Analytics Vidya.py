#(https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29)
#https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
#https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/
#http://www.nltk.org/

import re
result=re.split(r'y','Analytics') #Split at character 'y'
print(result)
result=re.split(r'i','Analytics Vidhiya',maxsplit=1) #split only for first occurence
print(result)
result=re.sub(r'India','the World','AV is largest Analytics community of India')
print(result)
result = re.findall(r'AV', 'AV Analytics Vidhya AV')
print(result)
result = re.search(r'Analytics', 'AV Analytics Vidhya AV')
print(result.group(0))
result = re.match(r'AV', 'AV Analytics Vidhya AV')
print(result.start()) #starting position
print(result.end()) #ending position


#We can combine a regular expression pattern into pattern objects, which can be used for pattern matching. 
#It also helps to search a pattern again without rewriting it.
pattern=re.compile('AV')
result=pattern.findall('AV Analytics Vidhya AV')
print(result)
result2=pattern.findall('AV is largest analytics community of India')
print(result2)


result=re.findall(r'.','AV is largest Analytics community of India') #matches each single character
print(result)

result=re.findall(r'\w','AV #$ # is largest Analytics community of India')#matches each single alphanumeric character
print(result)

result=re.findall(r'\w*','AV is largest Analytics community of India') #extract each word (not character)
print(result)

result=re.findall(r'\w\w','AV is largest Analytics community of India') #return first 2 characters of each word
print(result)

result=re.findall(r'\b\w.','AV is largest Analytics community of India')#Extract consecutive two characters those available at start of word boundary (using “\b“)
print(result)

result=re.findall(r'@\w+.(\w+)','abc.test@gmail.com, xyz@test.in, test.first@analyticsvidhya.com, first.test@rest.biz')
print(result) #grouping and printing data

result=re.findall(r'\d{2}-\d{2}-\d{4}','Amit 34-3456 12-05-2007, XYZ 56-4532 11-11-2011, ABC 67-8945 12-01-2009')
print(result) #extract date

result=re.findall(r'\d{2}-\d{2}-(\d{4})','Amit 34-3456 12-05-2007, XYZ 56-4532 11-11-2011, ABC 67-8945 12-01-2009')
print(result) #extract only year

result=re.findall(r'\b[aeiouAEIOU]\w+','AV is largest Analytics community of India')#Return words starts with alphabets
print(result)

#Validate a phone number (phone number must be of 10 digits and starts with 8 or 9) 
import re
li=['9999999999','999999-999','99999x9999']
for val in li:
 if re.match(r'[8-9]{1}[0-9]{9}',val) and len(val) == 10:
     print('yes')
 else:
     print ('no')
#Split a string with multiple delimiters     
line = 'asdf fjdk;afed,fjek,asdf,foo' # String has multiple delimiters (";",","," ").
result= re.split(r'[;,\s]', line)
print(result)    
#use method re.sub() to replace these multiple delimiters with one as space ” “.
line = 'asdf fjdk;afed,fjek,asdf,foo'
result= re.sub(r'[;,\s]',' ', line)
print(result)  

#Here we need to extract information available between <td> and </td> except the first numerical index.     
str='''
<tr align="center"><td>1</td> <td>Noah</td> <td>Emma</td></tr>
<tr align="center"><td>2</td> <td>Liam</td> <td>Olivia</td></tr>
<tr align="center"><td>3</td> <td>Mason</td> <td>Sophia</td></tr>
<tr align="center"><td>4</td> <td>Jacob</td> <td>Isabella</td></tr>
<tr align="center"><td>5</td> <td>William</td> <td>Ava</td></tr>
<tr align="center"><td>6</td> <td>Ethan</td> <td>Mia</td></tr>
<tr align="center"><td>7</td> <td HTML>Michael</td> <td>Emily</td></tr>'''

result=re.findall(r'<td>\w+</td>\s<td>(\w+)</td>\s<td>(\w+)</td>',str) #have achieved by using ()
print(result)  



import pandas as pd 
train=pd.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//train_E6oV3lV.csv")

train['word_count']= train['tweet'].apply(lambda x: len(str(x).split(" ")))
train[['tweet','word_count']].head()

train['char_count']= train['tweet'].str.len()
train[['tweet','char_count']].head()

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

train['avg_word'] = train['tweet'].apply(lambda x: avg_word(x))
train[['tweet','avg_word']].head()

## to remove the stopwords
#from nltk.corpus import stopwords
#stop = stopwords.words('english')
#
#train['stopwords'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
#train[['tweet','stopwords']].head()


#No of hashtags
train['hastags'] = train['tweet'].apply( lambda x:  len([x for x in x.split() if  x.startswith('#')] ) )
train[['tweet','hastags']].head()

#No of Numerics
train['numerics'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['tweet','numerics']].head()

#No of uppercase   
train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['tweet','upper']].head()

#convert everything to lower case
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['tweet'].head()
   
#Removal of stop words      
#from nltk.corpus import stopwords
#stop = stopwords.words('english')
#train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#train['tweet'].head()

#common word removal  ( gives you all the word frequency arranged in descending order)
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]

#remove top 10 all common words
freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#remove last 10
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()

#from textblob import TextBlob (used for spell correction)
#train['tweet'][:5].apply(lambda x: str(TextBlob(x).correct())) (here just doing for first 5 rows for ex purpose)

#Stemming refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach.
#For this purpose, we will use PorterStemmer from the NLTK library.
from nltk.stem import PorterStemmer
st = PorterStemmer()
train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


#Lemmatization is a more effective option than stemming because it converts the word into its root word, rather than just stripping the suffices.
# It makes use of the vocabulary and does a morphological analysis to obtain the root word.
# Therefore, we usually prefer using lemmatization over stemming.
#from textblob import Word
#train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#train['tweet'].head()



