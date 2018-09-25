from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
#%matplotlib inline
# read in the iris data
iris = load_iris()
# create X (features) and y (response)
X = iris.data
y = iris.target

#################Manual plot graph way of finding best K value (among 1 to 31)

# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())
# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

#################More efficient parameter tuning using GridSearchCV (using param_grid)
#################(It automatically tell that 13 is the best value using the grid.best_params_ 
#                 rather than you seeing it from earlier plotted graph manually)

from sklearn.model_selection import GridSearchCV

# define the parameter values that should be searched# defin 
k_range = list(range(1, 31))
print(k_range)
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)
print(param_grid)

# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
#You can set n_jobs = -1 to run computations in parallel (if supported by your computer and OS)

# fit the grid with data
grid.fit(X, y)

# view the results as a pandas DataFrame
import pandas as pd
print(pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']])


# examine the first result# examin 
print(grid.cv_results_['params'][0])
print(grid.cv_results_['mean_test_score'][0])

# print the array of mean scores only
grid_mean_scores = grid.cv_results_['mean_test_score']
print(grid_mean_scores)

# plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


#examine the best model with below automated variables by using GridSearchCV 

print(grid.best_score_)
print(grid.best_params_)#It gives first best K=13 value eventhou 13,18 and 20 are best values as per graph
print(grid.best_estimator_) 

#################GridSearchCV can be used with WEIGHT OPTIONS like (Uniform and Distance) to even better tune parameters

# define the parameter values that should be searched
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']
 # Uniform==>All points are treated equal co-efficents
 #distance==> If the points are close to the point ...they have given higher weightage
 
 # create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid) #It judges based on UNIFORM and WEIGHTAGE BY DISTANCE

# instantiate and fit the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(X, y)

# view the results
print(pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]) #Gives 60 lines from 30 input one for each Uniform and Distance

# examine the best model based on UNIFORM and WEIGHTAGE BY DISTANCE
print(grid.best_score_)
print(grid.best_params_) #It gives 13 has best value as output

# train your model using all data and the best known parameters(i.e Neighbour=13 found from last statement but you have to manually give it)
knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(X, y)
print(knn.predict([[3, 5, 4, 2]]))

#or

# shortcut: GridSearchCV automatically refits the best model using all of the data ( It chooses 13 automatically and predicts the model)
grid.predict([[3, 5, 4, 2]])

###############################################Reducing computational expense using RandomizedSearchCV#######################
# Its same as GridSearchCV but reduces the computation by searching  a subset of the parameters
from sklearn.model_selection import RandomizedSearchCV
# specify "parameter distributions" rather than a "parameter grid"
param_dist = dict(n_neighbors=k_range, weights=weight_options)
#Important: Specify a continuous distribution (rather than a list of values) for any continous parameters

# n_iter controls the number of searches ...here instead of 1 to 31 times it tries randomly only 10 RANDOM numbers between 1 and 31 and still give good accuracy
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5, return_train_score=False)
rand.fit(X, y)
print(pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']])

# examine the best model  (It gives approximately close K best value but it computes faster and mostly always gives correct value)
print(rand.best_score_)
print(rand.best_params_)


# run RandomizedSearchCV 20 times (with n_iter=10) and record the best score 
#( you can try the RandomizedSearchCV simply for 20 iterations  you can see all the 20 iterations it will give mostly good output only)
best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, return_train_score=False)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
print(best_scores)
#RandomizedSearchCV is better than GridSearchCV coz it does not try each value from k 1 to 31 instead random n_iter=<INPUT VALUE YOU GIVE>
