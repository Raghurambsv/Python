import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)
from textblob import TextBlob, Word
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set(color_codes=True)
import nltk
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re

df = pd.read_csv("//users//raghuram.b//Documents//amazon//data.csv")
df=df.loc[:,['Review Body','Sl No.']]
data=df['Review Body']
data=data[:10]

#gets list of only numeric datatypes column 
import numpy as np
import pandas as pd
df = pd.read_csv('breast-cancer-wisconsin.data')
X = X._get_numeric_data() 
# delete 'Survived', the response vector (Series)
X.drop('Survived', axis=1, inplace=True)
# we drop age for the sake of this example because it contains NaN in some examples
X.drop('Age', axis=1, inplace=True)


#Before Lenght of text
sum([len(x) for x in data]) #160

#Cleaning and Tokenizing data
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)

lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    remove_numbers =  re.sub(r"[0-9]+", "", punc_free)
    normalized = " ".join(lemma.lemmatize(word) for word in remove_numbers.split())
    return normalized

texts = [text for text in data if len(text) > 2]
doc_clean = [clean(doc).split() for doc in texts]

all_words = sum(doc_clean,[])#removing the nested lists and making one list
#dictionary = corpora.Dictionary(doc_clean)
#doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


#Top 10 frequent words
a = nltk.FreqDist(all_words)
d=pd.DataFrame()
d['words']=a.keys()
d['Counts']=a.values()

# selecting top 10 most frequent words     
d = d.nlargest(columns="Counts", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "words", y = "Counts")
ax.set(ylabel = 'Counts')
plt.show()




#Cleaning and Tokenizing data at DATAFRAME LEVEL
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    remove_numbers =  re.sub(r"[0-9]+", "", punc_free)
    normalized = " ".join(lemma.lemmatize(word) for word in remove_numbers.split())
    return normalized

df['clean_body'] = np.vectorize(clean)(df['Review Body']) #Vecotrise func is used to call func with parameters ( just like apply map)






##########F1Score#################
dtc = DecisionTreeClassifier()
# fit
dtc.fit(X_train, y_train)
# predict
y_pred = dtc.predict(X_test)
# f1 score
score = f1_score(y_pred, y_test)
print "Decision Tree F1 score: {:.2f}".format(score)

from sklearn.metrics import f1_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
f1_score(y_true, y_pred, average='macro')  
0.26...#Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
f1_score(y_true, y_pred, average='micro')  
0.33...#Calculate metrics globally by counting the total true positives, false negatives and false positives.
f1_score(y_true, y_pred, average='weighted')  
0.26...#Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
f1_score(y_true, y_pred, average=None)
array([0.8, 0. , 0. ]) #If None, the scores for each target_class are returned.


#######Classification report & Confusion matrix
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# We use a utility to generate artificial classification data.
X, y = make_classification(n_samples=100, n_informative=10, n_classes=3)
sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.5, random_state=0)
for train_idx, test_idx in sss:
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print(f1_score(y_test, y_pred, average="macro"))
    print(precision_score(y_test, y_pred, average="macro"))
    print(recall_score(y_test, y_pred, average="macro"))  

auto_wclf = SVC(kernel='linear', C= 1, class_weight='auto')
auto_wclf.fit(X, y)
auto_weighted_prediction = auto_wclf.predict(X_test)

print 'Accuracy:', accuracy_score(y_test, auto_weighted_prediction)

print 'F1 score:', f1_score(y_test, auto_weighted_prediction,
                            average='weighted')

print 'Recall:', recall_score(y_test, auto_weighted_prediction,
                              average='weighted')

print 'Precision:', precision_score(y_test, auto_weighted_prediction,
                                    average='weighted')

print '\n clasification report:\n', classification_report(y_test,auto_weighted_prediction)

print '\n confussion matrix:\n',confusion_matrix(y_test, auto_weighted_prediction)