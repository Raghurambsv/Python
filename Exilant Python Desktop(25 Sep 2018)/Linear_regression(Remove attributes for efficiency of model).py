
# coding: utf-8

# In[16]:


# conventional way to import pandas# conve 
import pandas as pd
# read CSV file from the 'data' subdirectory using a relative path
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
# display the first 5 rows
data.head()
# check the shape of the DataFrame (rows, columns)
data.shape

# conventional way to import seaborn
import seaborn as sns
# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# visualize the relationship between the features and the response using scatterplots
#plot SEPERATE SCATTER PLOTS TO EACH FEATURES(3 GRAPHS)
sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales', height=7, aspect=0.7, kind='reg')
# create a Python list of feature names# creat 
feature_cols = ['TV', 'radio', 'newspaper']
# use the list to select a subset of the original DataFrame
X = data[feature_cols]
# equivalent command to do this in one line
X = data[['TV', 'radio', 'newspaper']]
# print the first 5 rows
X.head()
# check the type and shape of X
print(type(X))
print(X.shape)
# select a Series from the DataFrame
y = data['sales']
# equivalent command that works if there are no spaces in the column name
y = data.sales
# print the first 5 values
y.head()
# check the type and shape of y
print(type(y))
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# default split is 75% for training and 25% for testing
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# import model
from sklearn.linear_model import LinearRegression
# instantiate
linreg = LinearRegression()
# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)
# print the intercept and coefficients ( underscore at end is a code standards to depict its algos values)
print(linreg.intercept_)
print(linreg.coef_)
# pair the feature names with the coefficients# pair t 
list(zip(feature_cols, linreg.coef_))
# make predictions on the testing set
y_pred = linreg.predict(X_test)

# Comparing these metrics:
# MAE is the easiest to understand, because it's the average error.
# MSE is more popular than MAE, because MSE "punishes" larger errors.
# RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.

# define true and predicted response values [ROUGH EXAMPLE TAKEN ]
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]

# calculate MAE by hand (Mean Absoulte Error or just MEAN)
print((10 + 0 + 20 + 10)/4.)
# calculate MAE using scikit-learn
from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))

# calculate MSE by hand# calcul  (MEAN SQUARE ERROR -to get positive values so squaring)
print((10**2 + 0**2 + 20**2 + 10**2)/4.)
# calculate MSE using scikit-learn
print(metrics.mean_squared_error(true, pred))

# calculate RMSE by hand (ROOT MEAN SQUARE - to reduce the number by SQUARE ROOT)
import numpy as np
print(np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4.))
# calculate RMSE using scikit-learn
print(np.sqrt(metrics.mean_squared_error(true, pred)))
# Computing the RMSE for our Sales predictions¶
print("consider all 3 features ['TV', 'radio', 'newspaper']",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Feature selection ( To reduce one feature and see if the accuracy is increased without moving away from problem stmt)
# Does Newspaper "belong" in our model? In other words, does it improve the quality of our predictions?
# Let's remove it from the model and check the RMSE!

# (Removing NEWSPAPER attribute coz its graph is mostly not lineraly sperable...so trying to eliminate n see 
#  if it improves the alogs predicting accuracy)

# create a Python list of feature names
feature_cols = ['TV', 'radio']
# use the list to select a subset of the original DataFrame
X = data[feature_cols]
# select a Series from the DataFrame
y = data.sales
# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)
# make predictions on the testing set
y_pred = linreg.predict(X_test)
# compute the RMSE of our predictions
print("After Newspaper feature removal",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# FINAL TAKEAWAY
# ##############
# The RMSE decreased when we removed Newspaper from the model. (Error is something we want to minimize, so a lower number for RMSE is better.)
# Thus, it is unlikely that this feature is useful for predicting Sales, and should be removed from the model.

