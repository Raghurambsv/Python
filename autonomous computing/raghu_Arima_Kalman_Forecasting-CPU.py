#!/usr/bin/env python
# coding: utf-8

import matplotlib
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
#data = pd.read_csv("stat_train_mindata.csv",index_col=0)
colnames=['cpuutilisation','datetime'] 
#data = pd.read_csv("cpu_usage_log_26_04_2019_18_58.csv",index_col=0,names=colnames,skiprows=[0,1])
data = pd.read_csv("/Users/raghuram.b/Desktop/AutonomicComputing/Raghu/RF+Arima/correctweek.csv",index_col=1,names=colnames,skiprows=[0])
data.head()

data.index

data.index = pd.to_datetime(data.index)

data.head()

data.index

data[pd.isnull(data['cpuutilisation'])]

data.head()

data.plot()

# In[8]:


# =============================Only for testing [Small data]=========
#data=data[0:200]
#data.plot()
# ===================================================================

print("#############################################")
print("No of records considered is =",data.shape[0])
print("#############################################")
 
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data) 
 
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data, freq=20)


# In[9]:


from pmdarima.arima import auto_arima


# In[10]:


stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=20,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True) 


# In[11]:


stepwise_model.aic()


# In[12]:


data.head()


# In[13]:


data.info()


# In[14]:


train = data.loc['2030-05-06 00:00:00':'2030-05-09 18:16:40']


# In[15]:


train.tail()


# In[16]:


test = data.loc['2030-05-09 18:16:45':]


# In[17]:


test.head()


# In[18]:


test.tail()


# In[19]:


len(test)


# In[20]:


len(train)


# In[21]:


stepwise_model.fit(train)


# In[22]:


future_forecast = stepwise_model.predict(n_periods=498)


# In[23]:


future_forecast


# In[24]:


future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])


# In[25]:


future_forecast.head()


# In[26]:


test.head()


# In[27]:


def update(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters.'''
    # Calculate the new parameters
    new_mean = (var2*mean1 + var1*mean2)/(var2+var1)
    new_var = 1/(1/var2 + 1/var1)
    
    return [new_mean, new_var]


# the motion update/predict function
def predict(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters, after motion.'''
    # Calculate the new parameters
    new_mean = mean1 + mean2
    new_var = var1 + var2
    
    return [new_mean, new_var]


# In[28]:


arimaoutputs = future_forecast.values.flatten()
actualoutputs = test.values.flatten()

arima_sig = 4.
actual_sig = 2.
mu = 0.
sig = 10000.


for n in range(len(arimaoutputs)):
    mu, sig = update(mu, sig, arimaoutputs[n], arima_sig)
    print('Update: [{}, {}]'.format(mu, sig))
    mu, sig = predict(mu, sig, actualoutputs[n], actual_sig)
    print('Predict: [{}, {}]'.format(mu, sig))

    
print('\n')
print('Final result: [{}, {}]'.format(mu, sig))



print(test)
print(future_forecast)

actual=test.values.flatten()
predict=future_forecast.values.flatten()
index=future_forecast.index.values.flatten()

# Calculate the absolute errors
errors = abs(predict - actual)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# # Determine Performance Metrics

# In[14]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / actual)
mape = np.nan_to_num(mape)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')





