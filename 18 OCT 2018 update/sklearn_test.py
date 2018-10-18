# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

df = pd.read_csv("RFA1.csv")
df
df=df.fillna(0)
df.shape
set(df["Disposition"])
from collections import Counter
Counter(df["Disposition"])
import re 
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string=str(string)
    string = re.sub(r"\n", "", string)    
    string = re.sub(r"\r", "", string) 
    string = re.sub(r"[0-9]", "digit", string)
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()
df.columns
from sklearn.model_selection import train_test_split
X = []
for i in range(df.shape[0]):
    X.append(clean_str(df.iloc[i][1]))
y= np.array(df["Disposition"])

y=y.tolist()


# Encoding categorical data
# Encoding the Independent Variable
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
y = np.array(df["Disposition"])
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
model = Pipeline([('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])
    
    
y = np.array(df["Disposition"])
model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])
    
#X_train=int(input(X_train))
#y_train=int(input(y_train))
model.fit(X_train, y_train)


#from sklearn import preprocessing
#lb = preprocessing.LabelBinarizer()
#lb.fit(["a", 2])   
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(pred, y_test)
    
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy_score(y_test, pred)
#save the model
from sklearn.externals import joblib
joblib.dump(model, 'model_question_topic.pkl', compress=1)
question = input( )
model.predict([question])[0]
