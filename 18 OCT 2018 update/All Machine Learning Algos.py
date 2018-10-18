from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#import numpy as np
from statistics import mode
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=20)
#Bagging where the items are bagged as ordered subsets
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(DecisionTreeClassifier(random_state=1))
model.fit(X_train, y_train)
print('Bagging score', model.score(X_test,y_test))
#Random bags and average voring is done in Random forest
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print('Random forest score:', model.score(X_test,y_test))


import numpy as np
from sklearn import preprocessing, model_selection as cross_validation, neighbors, svm
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
cancer = load_breast_cancer()



#Nearest neighbor classifier from skikit learn KNN
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Knn algo",accuracy)

##SVM
clf = svm.SVC()
#Run SVM and calculate confidence
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("Svm algo",confidence)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
#print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Decision algo Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))

#Naive bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
Accuracy=model.score(X_test, y_test)
print("Naive Bayes algo",Accuracy)


from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(X_test, y_test)
print("LogisticRegression algo",logisticRegr.score(X_test, y_test))

####ENSEMBLE for 4 algos (using Voting) (NEED to work on it)
from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(random_state=1)
model3 = neighbors.KNeighborsClassifier()
model4 = svm.SVC(random_state=1)

model1.fit(X_train,y_train)
print(' model 1 accuracy: LogisticRegression',model1.score(X_test, y_test))

model2.fit(X_train,y_train)
print(' model 2 accuracy: DecisionTreeClassifier',model2.score(X_test, y_test))

model3.fit(X_train,y_train)
print(' model 3 accuracy: KNeighbors',model3.score(X_test, y_test))

model4.fit(X_train,y_train)
print(' model 4 accuracy: SVM',model4.score(X_test, y_test))



model = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('knn', model3),('svm', model4)], voting='hard')
model.fit(x_train,y_train)
print('Ensemble model accuracy:',model.score(X_test,y_test))
