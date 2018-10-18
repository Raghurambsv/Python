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
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
#Bagging where the items are bagged as ordered subsets
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
print('Bagging score', model.score(x_test,y_test))
#Random bags and average voring is done in Random forest
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
print('Random forest score:', model.score(x_test,y_test))



import numpy as np
from sklearn import preprocessing, model_selection as cross_validation, neighbors, svm
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
df=pd.read_excel("//users//anaconda//Desktop//iris.xls")
df=pd.DataFrame(df,columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'iris'])
print(df[:4])

#Class needs to be predicted. Remove this from the data frame
X = df.iloc[:,:4]
#Class is what needs to be predicted
y = df.loc[:,'iris']
#Use 20% data for testing and 80% data for training
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
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
print("svm algo",confidence)
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
clf = svm.SVC()
#Run SVM and calculate confidence
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("svm algo",confidence)

from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(X_test, y_test)
print("LogisticRegression algo",logisticRegr.score(X_test, y_test))

####ENSEMBLE for 4 algos (using Voting) (NEED to work on it)
#from sklearn.ensemble import VotingClassifier
#model1 = LogisticRegression(random_state=1)
#model2 = DecisionTreeClassifier(random_state=1)
#model3 = neighbors.KNeighborsClassifier()
#model4 = svm.SVC(random_state=1)
#
#model1.fit(X_train,y_train)
#print(' model 1 accuracy:',model1.score(X_test, y_test))
#
#model2.fit(X_train,y_train)
#print(' model 2 accuracy:',model2.score(X_test, y_test))
#
#model3.fit(X_train,y_train)
#print(' model 3 accuracy:',model3.score(X_test, y_test))
#
#model4.fit(X_train,y_train)
#print(' model 4 accuracy:',model4.score(X_test, y_test))
#
#
#
#model = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('svm', model3),('knn', model4)], voting='hard')
#model.fit(x_train,y_train)
#print('Ensemble model accuracy:',model.score(x_test,y_test))
#