from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target


# Normal Test/split  
#for i in range(1,1000):
#    from sklearn.model_selection import train_test_split
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)
#     
#    # Repeat for KNN with K=5:
#    knn = KNeighborsClassifier(n_neighbors=18)
#    knn.fit(X_train, y_train)
#    y_pred = knn.predict(X_test)
#    Accuracy=metrics.accuracy_score(y_test, y_pred)
##    print(Accuracy)
#    if (Accuracy >= 0.85 and Accuracy <= 0.87):
#        print("The i=",i,metrics.accuracy_score(y_test, y_pred))
#    
# CrossValidation    
#for i in range(1,1000):
#    from sklearn.cross_validation import train_test_split
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)
#     
#    # Repeat for KNN with K=5:
#    knn = KNeighborsClassifier(n_neighbors=18)
#    knn.fit(X_train, y_train)
#    y_pred = knn.predict(X_test)
#    Accuracy=metrics.accuracy_score(y_test, y_pred)
##    print(Accuracy)
#    if (Accuracy >= 0.85 and Accuracy <= 0.87):
#        print("The i=",i,metrics.accuracy_score(y_test, y_pred))    
#
for i in range(1,100):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)
    from sklearn.linear_model import LogisticRegression
    
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()
    
    # fit the model with data
    logreg.fit(X_train, y_train)
    
    # predict the response for new observations
    y_pred = logreg.predict(X_test)
    
    Accuracy=metrics.accuracy_score(y_test, y_pred)
#    print(Accuracy)
    if (Accuracy >= 0.85 and Accuracy <= 0.9):
        print("The i=",i,metrics.accuracy_score(y_test, y_pred))

print("Cross validation starts")        

for i in range(1,100):
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)
    from sklearn.linear_model import LogisticRegression
    
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()
    
    # fit the model with data
    logreg.fit(X_train, y_train)
    
    # predict the response for new observations
    y_pred = logreg.predict(X_test)
    
    Accuracy=metrics.accuracy_score(y_test, y_pred)
#    print(Accuracy)
    if (Accuracy >= 0.85 and Accuracy <= 0.9):
        print("The i=",i,metrics.accuracy_score(y_test, y_pred))