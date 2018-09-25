######Kaagle work
import pandas as pd
#Training
train=pd.read_csv('http://bit.ly/kaggletrain')
train.head()

feature_cols=['Pclass','Parch']
X_train = train.loc[:,feature_cols]
X_train.shape
Y_train = train.Survived
Y_train.shape

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)

#Testing
test=pd.read_csv('http://bit.ly/kaggletest')
test.head()
feature_cols=['Pclass','Parch']
X_test = test.loc[:,feature_cols]
X_test.shape

#Takes the testing data of 2 columns and predicts the survival values 
Predict_class=logreg.predict(X_test) #This function gives 0's or 1's and that signifies ...survival rate

#Store the predicted results in CSV file
pd.DataFrame({'PassengerId':test.PassengerId,'Survived':Predict_class}).set_index('PassengerId').to_csv('predict.csv')

#Store test dataset as Pickle type file 
#(Pickling way storing files helps for better compact optimized way of storing files and can work background compatible with any python versions
# and you can convert this file to any format with ease and faster
train.to_pickle('train.pkl')
#read it from pickle
pd.read_pickle('train.pkl') 

 #http://dataaspirant.com/2017/02/13/save-scikit-learn-models-with-python-pickle/
 #import pickle
#
## pickle list object
#
#numbers_list = [1, 2, 3, 4, 5]
#list_pickle_path = 'list_pickle.pkl'
#
## Create an variable to pickle and open it in write mode
#list_pickle = open(list_pickle_path, 'wb')
#pickle.dump(numbers_list, list_pickle)
#list_pickle.close()

## unpickling the list object
# 
## Need to open the pickled list object into read mode
# 
#list_pickle_path = 'list_pickle.pkl'
#list_unpickle = open(list_pickle_path, 'r')
# 
## load the unpickle object into a variable
#numbers_list = pickle.load(list_unpickle)
# 
#print "Numbers List :: ", numbers_list


###########For any given DATASET you need to create TRAINING and TESTING dataset (Follow below method)################
import pandas as pd
ufo=pd.read_csv('http://bit.ly/uforeports')

ufo.sample(n=5)# random sample of rows but gives 5 rows (each time u run ...diff 5 rows)
ufo.sample(n=5,random_state=32) #gives random 5 rows but wen you give random_state=<number>...how much ever times u run you get same 5 rows

Training_Set=ufo.sample(frac=0.75,random_state=99)#gives 75% of ufo dataset

Testing_set=ufo.loc[~ufo.index.isin(Training_Set.index),:] #to generate the rest 25% as Testing_Set


