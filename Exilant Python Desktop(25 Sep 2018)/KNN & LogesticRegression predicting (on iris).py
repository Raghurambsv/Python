
# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target



# print the shapes of X and y
print(X.shape)
print(y.shape)


from sklearn.neighbors import KNeighborsClassifier


# "Instantiate" the "estimator"

# "Estimator" is scikit-learn's term for model
# "Instantiate" means "make an instance of"

knn = KNeighborsClassifier(n_neighbors=1)



print(knn)



knn.fit(X, y)



knn.predict([[3, 5, 4, 2]])




X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn.predict(X_new)


#use KNN with neighbours=5 prediction
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
print(knn.predict([[3, 5, 4, 2]]))

X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(knn.predict(X_new))


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response for new observations
logreg.predict(X_new)

