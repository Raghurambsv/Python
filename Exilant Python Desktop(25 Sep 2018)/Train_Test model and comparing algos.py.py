
# coding: utf-8

# In[2]:


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


# In[11]:


#Modelling the dataset thru sklearn
from sklearn.neighbors import KNeighborsClassifier
#testing manually with random predict with K=1
knn = KNeighborsClassifier(n_neighbors=1)
print(knn)
knn.fit(X, y)
print(knn.predict([[3, 5, 4, 2]]))
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(knn.predict(X_new))
# compute classification accuracy for the K=1 model for all 150 observations
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred)) # 100% accuracy


# In[10]:


#testing manually with random predict with K=5
knn = KNeighborsClassifier(n_neighbors=5)
# fit the model with data
knn.fit(X, y)
print(knn.predict([[3, 5, 4, 2]]))
# predict the response for new observations
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(knn.predict(X_new))
# compute classification accuracy for the K=5 model for all 150 observations
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred)) # 96.7% accuracy


# In[9]:


#testing manually with random predict and logestic regression algo
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X, y)
print(logreg.predict([[3, 5, 4, 2]]))
# predict the response for new observations
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(logreg.predict(X_new))
# compute classification accuracy for the logistic regression model
# predict the response values for the observations in X (150 observations)
logreg.predict(X)
# store the predicted response values# store 
y_pred = logreg.predict(X)
from sklearn import metrics
print(metrics.accuracy_score(y, y_pred)) #96 percent accuracy


# In[12]:


#Evaluating of algorithms with Train/Test split method (NOT MANUAL WAY)

# Model can be trained and tested on different data
# Response values are known for the testing set, and thus predictions can be evaluated
# Testing accuracy is a better estimate than training accuracy of out-of-sample performance

# print the shapes of X and y
print(X.shape)
print(y.shape)

# STEP 1: split X and y into training and testing sets# STEP  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# print the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)

# print the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)

# STEP 2: train the model on the training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# STEP 3: make predictions on the testing set
y_pred = logreg.predict(X_test)
# compare actual response values (y_test) with predicted response values (y_pred)
print(metrics.accuracy_score(y_test, y_pred)) #95% accuracy


# In[13]:


# Repeat for KNN with K=5:
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))#96.67% accuracy


# In[16]:


# Repeat for KNN with K=1:
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred)) #In using TRAIN/TEST method now K=1 accuracy is reduced to 95% 


# In[18]:


# Can we locate an even better value for K?
# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    
# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')    


# In[19]:


# Making predictions on out-of-sample data
# instantiate the model with the best known parameters ( Anything from K=7 to k=17...for exercise choosing k=11)
knn = KNeighborsClassifier(n_neighbors=11)

# train the model with X and y (not X_train and y_train)
knn.fit(X, y)

# make a prediction for an out-of-sample observation
knn.predict([[3, 5, 4, 2]])

# Downsides of train/test split?
# Provides a high-variance estimate of out-of-sample accuracy
# K-fold cross-validation overcomes this limitation
# But, train/test split is still useful because of its flexibility and speed

