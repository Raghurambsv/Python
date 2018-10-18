
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import svm
pd.set_option('max_columns', 30)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def load_train_data(data_name):
    data = pd.read_csv('data/train_' + data_name + '.txt', sep = ' ', header = None)
    data = data[list(range(0, 26))]
    data.rename(columns = {0 : 'unit', 1 : 'cycle'}, inplace = True)
    
    total_cycles = data.groupby(['unit']).agg({'cycle' : 'max'}).reset_index()
    total_cycles.rename(columns = {'cycle' : 'total_cycles'}, inplace = True)
    
    data = data.merge(total_cycles, how = 'left', left_on = 'unit', right_on = 'unit')
    data['RUL'] = data.apply(lambda r: int(r['total_cycles'] - r['cycle']), axis = 1)
    
    return data 


# In[4]:


def load_test_data(data_name):
    data = pd.read_csv('data/test_' + data_name + '.txt', sep = ' ', header = None)
    data = data[list(range(0, 26))]
    data.rename(columns = {0 : 'unit', 1 : 'cycle'}, inplace = True)
    
    total_cycles = data.groupby(['unit']).agg({'cycle' : 'max'}).reset_index()
    total_cycles.rename(columns = {'cycle' : 'total_cycles'}, inplace = True)
    
    data = data.merge(total_cycles, how = 'left', left_on = 'unit', right_on = 'unit')
    
    RUL = pd.read_csv('data/RUL_' + data_name + '.txt', sep = ' ', header = None)
    RUL = RUL[list(range(0, 1))]
    RUL['unit'] = list(range(1, len(RUL) + 1))
    RUL.rename(columns = {0 : 'RUL'}, inplace = True)
    
    data = data.merge(RUL, how = 'left', left_on = 'unit', right_on = 'unit')
    
    data['total_cycles'] = data.apply(lambda r: int(r['total_cycles'] + r['RUL']), axis = 1)
    data['RUL'] = data.apply(lambda r: int(r['total_cycles'] - r['cycle']), axis = 1)
    
    return data
    


# In[5]:


def normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    
    return (arr - mean) / std


# In[6]:


def compute_score(RUL_real, RUL_pred):
    d = RUL_pred - RUL_real
    
    return np.sum(np.exp(d[d >= 0] / 13) - 1) + np.sum(np.exp(-1 * d[d < 0] / 10) - 1)

def MSE(RUL_real, RUL_pred):
    d = RUL_pred - RUL_real
    return np.sqrt(np.sum(d ** 2)) / len(d)

def MAE(RUL_real, RUL_pred):
    return np.mean(np.abs(RUL_pred - RUL_real))


# # Data Preparation

# In[7]:


train_fd3 = load_train_data('FD003')
test_fd3 = load_test_data('FD003')


# In[8]:


features = list(range(2, 26))

train_fd3[features] = train_fd3[features].apply(normalize, axis = 0)
test_fd3[features] = test_fd3[features].apply(normalize, axis = 0)

test_rows = test_fd3.groupby(['unit']).agg({'cycle' : max})
test_rows = test_rows.reset_index()
test_rows = test_rows.merge(test_fd3, how = 'left', left_on = ['unit', 'cycle'],
                                                    right_on = ['unit', 'cycle'])


# In[9]:


y = train_fd3['RUL'].values

X_pred = test_rows[features].dropna(how = 'all', axis = 1)

X = train_fd3[features].dropna(how = 'all', axis = 1)

X = X.as_matrix()
X_pred = X_pred.as_matrix()

y_real = test_rows['RUL'].values



# # Support Vector Regression

# In[20]:


model = svm.SVR()
model.fit(X, y)
y_pred_svm = model.predict(X_pred)

fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (16, 4))

axes[0].scatter(test_rows['cycle'], y_real, color = 'red', label = 'Real RUL', alpha = 0.7, s = 15)
axes[0].scatter(test_rows['cycle'], y_pred_svm, color = 'blue', label = 'Predcited RUL', alpha = 0.7, s = 15)
axes[0].set_xlabel('Cycle')
axes[0].set_ylabel('RUL [cycles]')
axes[0].set_ylim([0, 250])
axes[0].set_title('SVR - Difference')

axes[1].scatter(y_real, y_pred_svm, color = 'green', alpha = 0.7, s = 15)
axes[1].set_xlabel('Real RUL [cycles]')
axes[1].set_ylabel('Predicted RUL [cycles]')
axes[1].set_ylim([0, 250])
axes[1].set_title('SVR - Correlation')

axes[2].hist(np.abs(y_real - y_pred_svm), bins = 20, edgecolor='black')
axes[2].set_xlabel('Absolute error [cycles]')
axes[2].set_ylabel('Freq [-]')
#axes[2].set_ylim([0, 250])
axes[2].set_title('SVR - Result Histogram')

axes[0].legend()

plt.savefig('img/scatter_svm.png')
plt.show()


## In[21]:
#
#plt.scatter(test_rows['cycle'], y_real, color = 'red', label = 'Real RUL', alpha = 0.7, s = 15)
#plt.scatter(test_rows['cycle'], y_pred_svm, color = 'blue', label = 'Predcited RUL', alpha = 0.7, s = 15)
#plt.xlabel('Cycle')
#plt.ylabel('RUL [cycles]')
#plt.title('SVR - Difference')
#plt.show()
#
#
## In[22]:
#
#
#plt.hist(np.abs(y_real - y_pred_svm), bins = 20, edgecolor='black')
#plt.xlabel('Absolute error [cycles]')
#plt.ylabel('Freq [-]')
#plt.title('SVR - Result Histogram')
#plt.savefig('img/hist_svm.png')
#plt.show()
#

# In[23]:


print('Average score per UUT:', compute_score(y_real, y_pred_svm) / 100)
print('Mean squared Error:', MSE(y_real, y_pred_svm))
print('Median difference:', np.median(np.abs(y_real - y_pred_svm)))




from sklearn.ensemble import RandomForestRegressor

# Fit regression model
regr_1 = RandomForestRegressor(max_depth=10,random_state=5)
regr_1.fit(X, y)
# Predict
y_pred_forest = regr_1.predict(X_pred)

fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (16, 4))

axes[0].scatter(test_rows['cycle'], y_real, color = 'red', label = 'Real RUL', alpha = 0.7, s = 15)
axes[0].scatter(test_rows['cycle'], y_pred_forest, color = 'blue', label = 'Predcited RUL', alpha = 0.7, s = 15)
axes[0].set_xlabel('Cycle')
axes[0].set_ylabel('RUL [cycles]')
axes[0].set_ylim([0, 250])
axes[0].set_title('RandomForestRegressor - Diff')

axes[1].scatter(y_real, y_pred_forest, color = 'green', alpha = 0.7, s = 15)
axes[1].set_xlabel('Real RUL [cycles]')
axes[1].set_ylabel('Predicted RUL [cycles]')
axes[1].set_ylim([0, 250])
axes[1].set_title('RFR - Correlation')

axes[2].hist(np.abs(y_real - y_pred_forest), bins = 20, edgecolor='black',color='green')
axes[2].set_xlabel('Absolute error [cycles]')
axes[2].set_ylabel('Freq [-]')
#axes[2].set_ylim([0, 250])
axes[2].set_title('RandomForestRegressor - Histogram')

axes[0].legend()

plt.savefig('img/scatter_svm.png')
plt.show()

#plt.hist(np.abs(y_real - y_1), bins = 20, edgecolor='black',color='green')
#plt.xlabel('Absolute error [cycles]')
#plt.ylabel('Freq [-]')
#plt.title('RandomForestRegressor  - Histogram(Depth = 5)')
#plt.savefig('img/hist_svm.png')
#plt.show()
#
#
#plt.scatter(test_rows['cycle'], y_real, color = 'red', label = 'Real RUL', alpha = 0.7, s = 15)
#plt.scatter(test_rows['cycle'], y_1, color = 'blue', label = 'Predcited RUL', alpha = 0.7, s = 15)
#plt.xlabel('Cycle')
#plt.ylabel('RUL [cycles]')
#plt.title('RandomForestRegressor - Difference')
#plt.show()



