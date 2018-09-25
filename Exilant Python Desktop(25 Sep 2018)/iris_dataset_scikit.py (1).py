

# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)

print(iris.data)

# print integers representing the species of each observation# print 
print(iris.target)

# print the names of the four features
print(iris.feature_names)



# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)

print(iris.target_names)

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target



