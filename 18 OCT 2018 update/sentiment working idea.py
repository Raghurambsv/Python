training = [
('Tom Holland is a terrible spiderman.','pos'),
('a terrible Javert (Russell Crowe) ruined Les Miserables for me...','pos'),
('The Dark Knight Rises is the greatest superhero movie ever!','neg'),
('Fantastic Four should have never been made.','pos'),
('Wes Anderson is my favorite director!','neg'),
('Captain America 2 is pretty awesome.','neg'),
('Let\s pretend "Batman and Robin" never happened..','pos'),
]
testing = [
('Superman was never an interesting character.','pos'),
('Fantastic Mr Fox is an awesome film!','neg'),
('Dragonball Evolution is simply terrible!!','pos')
]

from textblob import classifiers
classifier = classifiers.NaiveBayesClassifier(training)

print (classifier.accuracy(testing))
classifier.show_informative_features(5)


blob = TextBlob('the weather is terrible!', classifier=classifier)
print (blob.classify())


classifier.extract_features('Tom Holland is a terrible spiderman')
classifier.labels()
classifier.show_informative_features(5)

#Sentiments at sentence level
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

train = [
    ('I love this sandwich.', 'pos'),
    ('This is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ("My boss is horrible.", "neg")
]
cl = NaiveBayesClassifier(train)
#cl.classify("I feel amazing!")
blob = TextBlob("The beer is good. But the hangover is horrible.", classifier=cl)
for s in blob.sentences:
    print(s)
    print(s.classify())


###WORD CLOUD and its frequency graph
import matplotlib.pyplot as plt     
df=pd.read_csv('//users//raghuram.b//Desktop//product.csv')
df=df['Review Body']
all_words = ' '.join([text for text in df[0].split()]) #just one row data
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


#sentimentvalues=1 #positve
#sentimentvalues=2 #negative
df=pd.read_csv('//users//raghuram.b//Desktop//sentiment.csv')
df=df[df['sentimentvalues']==1]['Review Body']
#df=df[df['sentimentvalues']==2]['Review Body']

all_words = ' '.join([text for text in df])
from wordcloud import WordCloud
#wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#wordcloud.fit_words
#wordcloud.max_words
#wordcloud.stopwords
#wordcloud.process_text


#KMeans clustering
from sklearn.feature_extraction.text import TfidfVectorizer  
#tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3)) 
tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,1))
tfidf_vectorizer = TfidfVectorizer(use_idf=True) 
tfidf_matrix = tfidf_vectorizer.fit_transform(result)  
feature_names = tfidf_vectorizer.get_feature_names() # num phrases  
from sklearn.metrics.pairwise import cosine_similarity  
dist = 1 - cosine_similarity(tfidf_matrix)  
print(dist) 

from sklearn.cluster import KMeans 
num_clusters = 4  
km = KMeans(n_clusters=num_clusters,random_state=10)  
km.fit(tfidf_matrix)  
clusters = km.labels_.tolist()  

cf=pd.DataFrame({'ClusterID' :clusters})
print(cf['ClusterID'].value_counts())

frames=[df['Review Body'],cf['ClusterID']]
new_df=pd.concat(frames,axis=1)

file=pd.DataFrame(new_df)
file.to_csv('//users//raghuram.b//Desktop//cluster_output.csv')


##KNN sort cluster centers by proximity to centroid (Printing output)
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster {} : Words :".format(i))
    for ind in order_centroids[i, :]: 
        print(' %s' % feature_names[ind])
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
#Give your dataset in <result> below
texts = [text for text in result if len(text) > 2]
doc_clean = [clean(doc).split() for doc in texts]
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
ldamodel = models.ldamodel.LdaModel(doc_term_matrix, num_topics=6, id2word = 
dictionary, passes=5)
for topic in ldamodel.show_topics(num_topics=6, formatted=False, num_words=6):
    print("Topic {}: Words: ".format(topic[0]))
    topicwords = [w for (w, val) in topic[1]]
    print(topicwords)

