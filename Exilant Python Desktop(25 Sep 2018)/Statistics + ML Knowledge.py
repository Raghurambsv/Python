Python library
##############
NumPy ==> n-dimensional arrays,linear algebra functions,fourier transforms
Scipy ==> advanced version of NUMPY with high level science
Matplotlib=> Plotting variety of graphs
Scikit Learn==> Machine learning and statistical modelling(classification,regression and clustering)
Seaborn ==> Attractive plotting
Bokeh==> Interactive plotting on browser pages and Dashboards
Blaze ==> used to extract data from SPARK,MONGODB,PyTABLES and plot with bokeh
Requests==> for accessing web (Beautiful soup is another library) [ONLY ONE WEB PAGE]
Scrapy==> useful to access web and its subsequent web pages{MULTIPLE PAGES IN WEB LINK}

from sklearn.feature_extraction.text import CountVectorizer==>We call **vectorization** the general process of turning a collection of
                                     text documents(rows of data) into numerical feature vectors. 
                                     This specific strategy (tokenization, counting and normalization) is called 
                                     the **Bag of Words** or "Bag of n-grams" representation. 
                                     Documents are described by word occurrences while completely ignoring the relative position
                                     information of the words in the document.

pandas==> for data munging and data manipulations
#######
Pandas has 2 parts
.Series --> like array 
.Dataframes --> is similar to excel workbook

GRAPHS TO PLOT [https://www.analyticsvidhya.com/blog/2015/05/data-visualization-resource/]
##############
CONTINOUS VARIABLE (like series values of age)==> Histogram
CATEGORICAL VARIABLE(like many categories and there respective frequency)==> Barchart or Piechart(for 2 to 3 categories)....Linegraph if you need to plot quantitative of those 2 or 3 category variables
SCATTER PLOT ==> for plotting 2 continous variables (like Height vs Weight)
                we can also use it to plot 3 variables using varying size bubble (ex: cost,revenue,(bubble size for low,med,high)) 
GEOSPATIAL MAP==> is nothing but scatter plot laid out on geographical map                


scatter plot is best used for linear progression (means where there is a something constant in the data values)
Ex: price/sqft ratio is same for all datapoints
    (if there no constant ratio than it will deviate from the LINEAR LINE...thats called Noise..Liner can be +ve or -ve too)
    
Bar graph: Is better used to plot a cumulative values (it plots 2D data)   

Histogram: (Plots 1D data) its like Age in x-axis  and y-Axis(is simply how many belong to that age group plotted or frequency of age)

Pie-Chart: is best used for relative data (like how much percent each team scored in poltics)

Variance = sum of all squares (mean - indivual_items) / No of items 
[ If the variance value is small it tells the data values in the set are almost close to each other]
[If the variance is large it tells that the data values have extreme (low/high)values present in set]
standard devition = squareroot of Variance

Machine Learning Videos
#######################
https://www.youtube.com/channel/UCnVzApLJE2ljPZSeQylSEyg --Machine learning videos
https://www.youtube.com/watch?v=RlQuVL6-qe8
https://www.youtube.com/watch?v=MEG35RDD7RA&hd=1
http://work.caltech.edu/telecourse.html
https://www.dataschool.io/15-hours-of-expert-machine-learning-videos/
https://www.youtube.com/watch?v=0pP4EwWJgIU ==> How to choose which model through Train_And_Test model ( This guy follow all his video lessons 
                                                                                                        and at end of every video he go thru links shared by him )

WHOLE MACHINE LEARNING IS ABOUT TAKE ANY RAW_DATA/NON_LINEARLY_SEPERABLE_DATA TO AND MAKE DATA LINEALLY SEPERABLE
(you can also say ML is used create models from dataset in such a way that it can GENERALIZE IN PREDICTING THE FUTURE DATASET)
OR
MACHINE LEARNING IS ALL ABOUT MAKING ASSUMPTIONS AND REACH THE RESULT 
(or its a process of reaching to a accurate result with as much less assumtions as possible)

Supervised Learning (https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)
###################
This algorithm consist of a target / outcome variable (or dependent variable) 
which is to be predicted from a given set of predictors (independent variables).
 Using these set of variables, we generate a function that map inputs to desired outputs.
 The training process continues until the model achieves a desired level of accuracy on the training data.
 Ex: Y=mx+c (predicting Y is the task )
a)Classification : Is supervised learning in which the target/outcome is categorical in nature....
like take data and classify them into labels ( like photo images ==> classify male or female)
(Student ==> pass or fail)

b)Regression : Is supervised learning in which the target/outcome is continous (like from photo images==> predict age ( which will be 16 or 22 or 22.5 or 23...so on )
               (Student ==> Predict percentage of marks)
or 
regression is like for some input you create output and later for New_Input you PREDICT the New_Output

Entropy: means the homogenity of the values...entropy is measured b/w '0' and '1'....'0' means all values are same ...'1' means more values distinct

EXAMPLES OF SUPERVISED LEARNING:Algorithms
===========================================
Ex1: You can use Email whether its  (SPAM/NOT_SPAM)
     Input : History of emails come to your inbox
     Output: Is YES/NO(like binary o/p)...whether the future email is SPAM or NOT_SPAM
     
Ex2: Bank Credit Score for customers
     Input: History of customers (transactions he has done,loan taken,loan paid on time,his salary,his age,perm hosuse or in rented) 
     Output: Is YES/NO...whether the future unknown customer comes based on the KnowledgeBaseOfBank to give LOAN/NOT_GIVE_LOAN
     Analysis: Each customer is rated based on many attributes as told in INPUT above with weightage attached to each
               attribute finally using this is to compare with History KnowledgeBaseOfBank ...just see if the New_Customer 
               has score higher than the threshold setup by bank from the History.
               
Ex3: In coin and vending machine example...here you would have 4 labels given initially(1re,2re,5re and 10rs)    
     you will have training dataset with above labels...testing dataset/outofsample dataset entries you need to classify them into 4 pre-defined labels           
     
Regression(Logistic,lasso,ridge,elastic),Random forest
Decision Trees: each level you go down you come close to the solution ( or,xor,binary decision trees)
                Its simply like 20 questions ( each question you come closer to the answer)
                for a group of decision trees u need to take a call then use RANDOM FOREST algorithm
                
KNN algorithm - It helps for maps of real estate(Area/price) ...new area value(not available in previous map)
                to find out(you can calculate by the nearest marked value in map with the price)
               (Its like have a base map first...n slowly when properties comes up need to update map manually or automate it by ML)
               It can be used for both classification and regression problems.
               usually you run a for loop of values of 'K' from range (1 to 31) and see where we get peak performance
               If we choose low value of K ==>  Complexity is low 
                                                but bias will be more (Away from expected BULLS EYE middle point)
                                                or low variance from mean value (All arrow points will be closer)
                                                or OverFitting problem
               If we choose high value of K ==> Complexity will be high
                                                or bias will be less (Much closer to expected BULLS EYE point)
                                                or High Variance from mean value (But the distance between outliers and expected points will be more)
                                                   (All arrow points are most likely been scattered or have huge diff from min to max points perspective)
                                                or UnderFitting problem
               
Ensemble(Grouping) Learning algo: Is for boosting
                       ( Learn something from one subset data ...than learn something in another subset...so on n compile one complete rules 
                        from all the diff subset ...eventually it will become a formula)
                      
Bayes rule : It is a classification technique based on Bayes’ theorem with an assumption of independence between predictors. 
             In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier would consider all of these properties to independently contribute to the probability that this fruit is an apple.
             This algorithm is mostly used in text classification and with problems having multiple classes.


Unsupervised Learning (https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)
######################
HAS PYTHON ALGORITHM & CODE SAMPLE ==>(https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)
                                      (https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)
                                      http://dataaspirant.com/2014/09/19/supervised-and-unsupervised-learning/
In this algorithm, we do not have any target or outcome variable to predict / estimate. 
It is used for clustering population in different groups, which is widely used for
segmenting customers in different groups for specific intervention. 
its like taking data and making sense of it or you describe the data 
EXAMPLES OF UNSUPERVISED LEARNING: Apriori algorithm, K-means,Hidden Markov models

(There wont be any pre-labels to data ..u need to classify in terms of many clusters n finally come up with label to them )

Example: You want to build a vending machine ....so the machine should be able to classify the denominations of the coins inserted into it
         (usually ur approach would be about size and weight of the coin)...so you will be given a dataset(without any labels)...
         you draw a graph out ofthe given lableless dataset 
         using the parameter size ==> (1rs,2rs & 10rs) and  (5rs)
         Using  weight ==> wen you use weight parameter along with size (1re,2re,5re and 10rs) you come with four diff categories 
         ..you will see a somewat pattern in terms of clusters (usually here in example i am taking only
                                                                                                     4 denominations coins 1re,2re,5re and 10rs)
         so u will get  a graph and visibly you can see few datapoints are together and few little away from it
         the datapoints which are close to each other will be one type of coin(say 5rs coin)
         and datapoints slightly away from the group might be in between of denominations due to the coin usage and rub of little weight
         so from the graph u sud be able to predict there 4 major clusters and little noises in it
         later on looking at it closely you can see the first cluster belongs to 1 rs for light weight ....and last one is 10 rs coz of weight
         SO YOU CAN SAY FROM SOME DATA YOU DERIVED A SLIGHTLY MEANINGFUL OUTPUT...THIS IS CALLED UNSUPERVISED LEARNING
         YOU LITERALLY FIGURED OUT SOME PATTERN FROM RAW DATA


K-means clustering algorithm : it randomly picks the points and those points start collecting the points closer to it.
                               and again in next parse you repeat the same with diff points and like wise derive some meaningful dataset. 
                               
                               
                               

Reinforcement learning (https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)
######################
Its like strengthing your analysis or improvising
Using this algorithm, the machine is trained to make specific decisions. 
It works this way: the machine is exposed to an environment where it trains itself continually using trial and error.
This machine learns from past experience and tries to capture the best possible knowledge to make accurate business decisions.
Ex: Take models and then create polices..take policies and making rules
    Or like you give a high price to product even thou its useful to ppl they wont buy coz of the price(IPHONE) 
    so u wont get expected sales even u know the product is world class and everyone needs
    so reinforcement way of doing is adding more features for the price, give freebies with it like gold coin or reduce the price by giving 
    discounts etc etc...this way of reinforcement you will increase the goal of selling all your IPHONES (may be slight comprimise of 
                                                                                                       price...but do take another approach 
                                                                                                 and increase accuracy for revenue generation)
Example of Reinforcement Learning: Markov Decision Process

EXAMPLE: In terms of coin example you will use reinforcement algo to increase the accuracy of the identifaction of the coin.







#######################CLASSIFICATION & Regression Algorithms(Mostly used for SUPERVISED MAHINE LEARNING)##########################
#In classification, the idea is to predict the target class by analysis the training dataset.
#Use the training dataset to get better boundary conditions which could be used to determine each target class.
#Once the boundary conditions determined, the next task is to predict the target class


.Linear classifiers      --> Logistic regression,Naive Bayes classifier,Fisher’s linear discriminant
.Support vector machines(SVM)-->Least squares support vector machines
.Kernel estimation -->k-nearest neighbor
.Decision trees-->Random forests
.Neural networks
.Learning vector quantization
.Quadratic classifiers

.Logistic Regression
.Linear regression
.polynomial regression
.SVM for regression
.Regression(Logistic,lasso,ridge,elastic)

Applications of Classification algos examples
=============================================
Email spam classification
Bank customers loan pay bank willingness prediction.
Cancer tumour cells identification.
Sentiment analysis.
Drugs classification



#######################CLUSTERING Algorithms(Mostly used for UNSUPERVISED MAHINE LEARNING)##########################
#In clustering the idea is not to predict the target class as like classification , 
#it’s more ever trying to group the similar kind of things by considering the most satisfied condition
#or involves grouping data into categories based on some measure of inherent similarity or distance.
#or all the items in the same group should be similar and no two different group items should not be similar


Unsupervised Linear clustering algorithm
=======================================
k-means clustering algorithm
Fuzzy c-means clustering algorithm
Hierarchical clustering algorithm
Gaussian(EM) clustering algorithm
Quality threshold clustering algorithm

Unsupervised Non-linear clustering algorithm  (https://sites.google.com/site/dataclusteringalgorithms/home)
======================================= (http://scikit-learn.org/stable/modules/clustering.html)
MST based clustering algorithm
kernel k-means clustering algorithm
Density-based clustering algorithm

Applications of clustering algos examples
=============================================
Recommender systems
Anomaly detection
Human genetic clustering
Genom Sequence analysis
Analysis of antimicrobial activity
Grouping of shopping items
Search result grouping
Slippy map optimization
Crime analysis
Climatology





Here is the list of commonly used machine learning algorithms. These algorithms can be applied to almost any data problem:
#########################################################################################################################
NOTE:  linear regression is a regression algorithm whereas logistic regression is a classification algorithm.
       Model Equalation metrics ==>REGRESSION problems uses RootMeanSquaredError
                                   CLASSIFICATION problems use ACCRUACY


Linear Regression   ==>  Linear regression is generally a REGRESSION PROBLEM (i.e it takes 'x' inputs and gives continous output of 'Y' values)
(Regression type of problems)     for ex: Output is 1.15 rounded to 1 and 3.3 is rounded to 3
                       It uses ROOT MEAN SQUARE usually.
                       Regression is the task of predicting a continuous quantity.
                       A regression algorithm may predict a discrete value, but the discrete value in the form of an integer quantity.
                       Ex: Regression : Is supervised learning in which the target/outcome is continous 
                       (like from photo images==> predict age ( which will be 16 or 22 or 22.5 or 23...so on )
                       or
                       For example, a house may be predicted to sell for a specific dollar value, 
                       It has output variable as SALES VALUES FIGURES perhaps in the range of $100,000 to $200,000
                       or in the range of $50,000 to 75,000 or in range of $200,000 to $30,000...
                       
( when all datapoints fall on the line in scatterplot it helps to take business decision better..
                       Formula: y=mx+c where m is slope of line ...c is the y intercept)
   Y=MX+C for perfectly linear data...but in real world its highly unlikely to be like that 
   It will be more of like Y= b0 +b1(x) + b2(x) +b3(x).....bn(x) 
    where "b" is co-efficent here ....or you can simply say that its like weightages put for each attributes/features for the problem set.
    Linear regression is like sum of squared means (i.e finding the best fit line)..if all points fall perfectly on line ..accuracy is penultimate
    If it falls far from line...Linear regression accuraccy is not efficient.
                     Pros: linear regression doesnt need any TUNNING as like in KNN to find the best "K" accuracy...so its runs faster for 
                           larger datasets too.
                     Disadvantage: Its unlikely to produce best  predict accuracy or its accuracy prediction is not that good...coz it expects the data to be exactly lineraly seperable
                                   if its somewat partially linerally seperable then its accuracy is not bankable:
                                       
Logistic Regression ==>      Logistic is a CLASSIFICATION PROBLEM...coz it says either SPAM/NOT_SPAM or FRAUD or NOT_FRAUD (http://dataaspirant.com/2017/03/02/how-logistic-regression-model-works/)
                             “Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic function” 
(Classification type of problems)   It gives the answer in terms of binary probability like 0/1 or yes/No ..you use log function to plot the graph
                            ( log is best mathematical term chosen which serves the purpose....This popular logistic function to give out the probabilty  is the SOFTMAX/SIGMOID FUNCTION )..Softmax function output values are always in the range of (0, 1). The sum of the output values will always equal to the 1.)
                            Linear regression assigns values to the response variable whereas logistic assigns probability.
                            Linear regression is used for numeric values prediction whereas 
                            Logistic is generally used for classification like default/non_default,fraud/no_fraud or 2 or more output labels but discreet in nature
                            Classification is the task of predicting a discrete class label.
                            A classification algorithm may predict a continuous value, but the continuous value is in the form of a probability for a class label.
                            Ex: Iris Dataset ( output is only of 3 categories )
       IMPORTANT: Classification predictions can be evaluated using accuracy, whereas regression predictions uses RMSE
                  In some cases, a classification problem can be converted to a regression problem. 
                  For example, a label can be converted into a continuous range.  
                  
                  
Multinominal Logistic Regression ==>  If the target class is more than > 2 (then its called Multinominal Logistic Regression) http://dataaspirant.com/2017/03/14/multinomial-logistic-regression-model-works-machine-learning/ 
  
                                     SIGMOID function: used in the logistic regression model for binary classification.(-1 to 1) 
                                                      (sigmoid function take any range real number and returns the output value which falls in range of -1 to 1)
                                     
                                     SOFTMAX function: used in the logistic regression model for multiclassification. (0 to 1 probabilities) 
                                                      Softmax function calculates the probabilities distribution of the event over ‘n’ different events. In general way of saying, this function will calculate the probabilities of each target class over all possible target classes. Later the calculated probabilities will be helpful for determining the target class for the given inputs.
                                                      The main advantage of using Softmax is the output probabilities range. The range will 0 to 1, and the sum of all the probabilities will be equal to one.
                                     
                                     Example usecases: .Predicting the Iris flower species type (3 different species)
                                                       .Evaluating the acceptability of car using its given features( Good,vGood, Acc, unAcc)
                                                       .Predicting the animal category using the given animal features( Dog, Cat, Tiger, Lion )

Polynomial Regression ==>
( when variance has extreme values(or more outliers)...or if linear regression cant predict we use this)
  so this time straight line wont be there ...it will be more of a curve to catch all the datapoints
   y = delta1 + delta2 * x + delta3 * x(square) + delta4*x(cube)............
  with polynomial way if u try to create more than curve and try to connect all dots it will lead to OVER_FITTING problem
  if use connect too low datapoints it leads to UNDER_FITTING problem ...so always use JUST_RIGHT polynominal
  so JUST_RIGHT one is done by checking how much is degree of polynominal needed)
To overcome UNDERFITTING we can basically add new parameters to our model so that the model complexity increases,
and thus reducing errors and move closer to JUST right
To Overcome OVERFITTING reduce the no of parameters to reduce the complexity of the model.
                          
Ridge Regression ==> if you plot it for complex model with lot of parameters and still difficult to get the JUST_RIGHT graph 
                     you can thing of reducing the co-efficents of all the parameters on a ratio( or if two related variables are there
                     it retains one and reduces other one close to zero)...so that all the rest of data comes slightly closer
                     to the line from earlier extremes

Lasso Regression ==> It is slightly better than Ridge coz ridge reduces the datapoints close to line but still it doesnt help
                    Entirely to rule out few parameters to reduce complexity...but LASSO-REGRESSION formula will give a output where
                    most of the datavalues comes close to line entirely and only few data points will left ...so that we can make wise 
                    decision from there ...so we can call this algo as FEATURE SELECTION
                    It is generally used when we have more number of features, because it automatically does feature selection.
Explanation to read below:
#Now that you have a basic understanding of ridge and lasso regression, let’s think of an example where we have a large dataset,
#lets say it has 10,000 features. And we know that some of the independent features are correlated with other independent features.
# Then think, which regression would you use, Rigde or Lasso?
#Let’s discuss it one by one. If we apply ridge regression to it, it will retain all of the features but will shrink the coefficients. But the problem is that model will still remain complex as there are 10,000 features, thus may lead to poor model performance.
#Instead of ridge what if we apply lasso regression to this problem. The main problem with lasso regression is when we have correlated variables, it retains only one variable and sets other correlated variables to zero. That will possibly lead to some loss of information resulting in lower accuracy in our model.
#Then what is the solution for this problem? Actually we have another type of regression, known as elastic net regression,
# which is basically a hybrid of ridge and lasso regression.                    
            
All algorithms in ML does following steps (Import...Instantiate...FIT....Predict )
        
ElasticNet Regression ==> Hybrid of Ridge & Lasso
Decision Tree(explained above)
Support Vector Machine ==> Supervised ML uses SupportVectorMachine algo and Unsupervised ML uses SupportVectorMachineClustering
######################
          There are 2 kinds of SVM classifiers:
                    Linear SVM Classifier --> Where data is lineraly seperable
                                              LinearSVC() is an SVC for Classification that uses only linear kernel. In LinearSVC(), we don’t pass value of kernel, since it’s specifically for linear classification.
                    Non-Linear SVM Classifier--> Where Kernel_method(RBF kernel,Polynomial kernel... etc)is used to convert Non-Linear data to Linear data
                                                It moves the data into diff DIMENSION and than maximises the differences between the two planes.
                                                By default kernel parameter uses “rbf” as its value but we can pass values like “poly”, “linear”, “sigmoid” or callable function.
           Advantages of SVM Classifier:
                        Svm classifier mostly used in addressing multi-classification problems
                        SVMs are effective when the number of features is quite large.
                        It works effectively even if the number of features are greater than the number of samples.
                        Non-Linear data can also be classified using customized hyperplanes built by using kernel trick.
                        It is a robust model to solve prediction problems since it maximizes margin.
          Disadvantages of SVM Classifier:
                        The biggest limitation of Support Vector Machine is the choice of the kernel. The wrong choice of the kernel can lead to an increase in error percentage.
                        With a greater number of samples, it starts giving poor performances.
                        SVMs have good generalization performance but they can be extremely slow in the test phase.
                        SVMs have high algorithmic complexity and extensive memory requirements due to the use of quadratic programming.                                     

           SVM Applications:
                        (a)SVMS are a byproduct of Neural Network. They are widely applied to pattern classification and regression problems. Here are some of its applications:
                        (b)Facial expression classification: SVMs can be used to classify facial expressions. It uses statistical models of shape and SVMs.
                        (c)Speech recognition: SVMs are used to accept keywords and reject non-keywords them and build a model to recognize speech.
                        (d)Handwritten digit recognition: Support vector classifiers can be applied to the recognition of isolated handwritten digits optically scanned.
                        (e)Text Categorization: In information retrieval and then categorization of data using labels can be done by SVM.    



Naive Bayes ==>Bayes’ Theorem with an assumption of independence among predictors/features  (https://dataaspirant.com/2017/02/06/naive-bayes-classifier-machine-learning/)
############ 
            It works on conditional probability. Conditional probability is the probability that something will happen, given that something else has already occurred. Using the conditional probability, we can calculate the probability of an event using its prior knowledge.
           (wen used for text analytics problems....Ex: Ham/Spam ==> It calculates how much SPAMINESS or HAMINESS the future mail consists of and depending upon 
             how which is greater it classifes the mail as SPAM/HAM)not accurate but gives results faster for big dataset 
            Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.
            What are the Pros and Cons of Naive Bayes?
                    Pros:
                    It is easy and fast to predict class of test data set. It also perform well in multi class prediction
                    When assumption of independence holds, a Naive Bayes classifier performs better compare to other models like logistic regression and you need less training data.
                    It perform well in case of categorical input variables compared to numerical variable(s). For numerical variable, normal distribution is assumed (bell curve, which is a strong assumption).
                    
                    Cons:
                    If categorical variable has a category (in test data set), which was not observed in training data set, then model will assign a 0 (zero) probability and will be unable to make a prediction. This is often known as “Zero Frequency”. To solve this, we can use the smoothing technique. One of the simplest smoothing techniques is called Laplace estimation.
                    On the other side naive Bayes is also known as a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.
                    Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.
                    Naive Bayes can learn individual features importance but can’t determine the relationship among features.
                    
             Applications of Naive Bayes Algorithms
                            Real time Prediction: Naive Bayes is an eager learning classifier and it is sure fast. Thus, it could be used for making predictions in real time.
                            Multi class Prediction: This algorithm is also well known for multi class prediction feature. Here we can predict the probability of multiple classes of target variable.
                            Text classification/ Spam Filtering/ Sentiment Analysis: Naive Bayes classifiers mostly used in text classification (due to better result in multi class problems and independence rule) have higher success rate as compared to other algorithms. As a result, it is widely used in Spam filtering (identify spam e-mail) and Sentiment Analysis (in social media analysis, to identify positive and negative customer sentiments)
                            Recommendation System: Naive Bayes Classifier and Collaborative Filtering together builds a Recommendation System that uses machine learning and data mining techniques to filter unseen information and predict whether a user would like a given resource or not
                             
                            
            How to build a basic model using Naive Bayes in Python?
                            Again, scikit learn (python library) will help here to build a Naive Bayes model in Python. There are three types of Naive Bayes model under scikit learn library:
                            
                            Gaussian: It is used in classification and it assumes that features follow a normal distribution.
                            
                            Multinomial: It is used for discrete counts. For example, let’s say,  we have a text classification problem. Here we can consider bernoulli trials which is one step further and instead of “word occurring in the document”, we have “count how often word occurs in the document”, you can think of it as “number of times outcome number x_i is observed over the n trials”.
                            
                            Bernoulli: The binomial model is useful if your feature vectors are binary (i.e. zeros and ones). One application would be text classification with ‘bag of words’ model where the 1s & 0s are “word occurs in the document” and “word does not occur in the document” respectively.
                            
         Tips to improve the power of Naive Bayes Model
                            Here are some tips for improving power of Naive Bayes Model:
                            If continuous features do not have normal distribution, we should use transformation or different methods to convert it in normal distribution.
                            If test data set has zero frequency issue, apply smoothing techniques “Laplace Correction” to predict the class of test data set.
                            Remove correlated features, as the highly correlated features are voted twice in the model and it can lead to over inflating importance.
                            Naive Bayes classifiers has limited options for parameter tuning like alpha=1 for smoothing, fit_prior=[True|False] to learn class prior probabilities or not and some other options (look at detail here). I would recommend to focus on your  pre-processing of data and the feature selection.
                            You might think to apply some classifier combination technique like ensembling, bagging and boosting but these methods would not help. Actually, “ensembling, boosting, bagging” won’t help since their purpose is to reduce variance. Naive Bayes has no variance to minimize.                



Decision Tree ====>  Decision tree classifier is a classification model which creates set of rules from the training dataset.
                     Later the created rules used to predict the target class.
                     The trained decision tree can use for both classification and regression problems.
                     Complexity-wise decision tree is logarithmic in the number observation in the training dataset.
                     But problem is you have to keep building the tree for every new observation ( so best suited if the model doesnt change frequently)
                     Graphviz is one of the visualization libray which is best suited to visualize Decision Tree
       

kNN
K-Means
Random Forest
Dimensionality Reduction Algorithms ==> Its better used when you need to choose SIGNIFICANT PREDICTORS from lot of predictors labels
Gradient Boosting algorithms
GBM
XGBoost
LightGBM
CatBoost

Bagging algorithms:
##############

Bagging meta-estimator
Random forest

Boosting algorithms:
###############
AdaBoost
GBM
XGBM
Light GBM
CatBoost

Clustering algorithms
#####################
K-Means
Special clustering
men-shift


Dimensionality reduction (Reducing the number of random variables to consider)
(Its better used when you need to choose SIGNIFICANT PREDICTORS from lot of predictors labels)
########################
PCA
feature selection
non-negative matrix factorization

Feature Selection (Data collection==>Feature transformation==>Feature selection)   http://dataaspirant.com/2018/01/15/feature-selection-techniques-r/
#################
 Feature transformation is to transform the already existed features into other forms. Suppose using the logarithmic function to convert normal features to logarithmic features.
 Feature selection is to select the best features out of already existed features
 
 .Coorelation   ==>Correlations is the number between +1 and -1 that represnts the strength and direct relationship between two variables
                   Correlations that are closer to (+1 & -1) are better able to predict accuracy
                   While plotting correlations, we always assume that the features and dependent variable are numeric.
  .Random forests are based on decision trees and use bagging to come up with a model over the best feature to be used.
  
This is why feature selection is used as it can improve the performance of the model. This is by removing predictors with chance or negative influence and provide faster and more cost-effective implementations by the decrease in the number of features going into the model.
To decide on the number of features to choose, one should come up with a number such that neither too few nor too many features are being used in the model.
In case of a large number of features (say hundreds or thousands), a more simplistic approach can be a cutoff score such as only the top 20 or top 25 features or the features such as the combined importance score crosses a threshold of 80% or 90% of the total importance score.


Model selection
###############
.DummyClassifier ==> Need to research

.Train_and_Test split ==> usually 20-40% as testing dataset and remaining is traing dataset
                         Pros: Faster running and Simpler to examine the detailed results of the testing process(SIMPLE MODEL)
                         Cons: Not that efficent in predicting the OUT_OF_SAMPLE datasets coz (80\20 is done only once and model is built)
                               which gives HIGH VARIANCE IN ACCURACY( tried by playing with RANDOM_STATE change of values...always gives diff accruacy)
                               Ideal model sud give same accuracy across diff random_state but Train/Test algo accruacy varies more
                               More Variance in predicting accuracy from algo .....it turns out less efficent in predicting out_of_Sample data
                              
                         
.k-cross validation   ==> it does the same thing as Train_and_Test split but a "K-times" coz it splits the dataset into folds
                       each fold is treated as testing set against the rest_of_union_of_fold_datasets(Training set) and this is repeated K-Times
                       each "k-time" the testing set is diff fold and ALL THE AVERAGE OF THE RESULTS IS USED FOR PREDICTING ACCRURACY.
                       (usual recomendation is K=10 times )
                       Pros: Cross validation eliminates the VARIANCE range to low (compared to Train/Test algo coz it uses only 1 trial(80% trainig set and 20% testing set with one RANDOM_STATE value))
                             It gives better estimate of OUT_OF_SAMPLE datasets (So the predicting accruacy is high)
                             So inshort you can say K-CROSS VALIDATION algo makes us feel more confident on telling OUT_OF_SAMPLE dataset
                             cross validation helps better for selecting models,tunning parametrs(taking out few features like newspaper)
                             
                       Cons: It takes K-times more time to execute than Train_Test_Split.
                             It takes more time to execute or computationally expensive (Coz its COMPLEX MODEL)
                             
.Gridsearch ==> GridSearchCV (grid search cross validation) 
                For k-CrossValidation you need to manually try out all the NEIBOURING_K_VALUES count from(1 to 31) for KNN algo
                to see which would be the best NEIBOURING_K_VALUES 
                Instead you can use GridSearchCV can give you directly that Best 'K' value
                You can further tune GridSearchCV by using with WEIGHT OPTIONS like (Uniform and Distance) 
                 # Uniform==>All points are treated equal co-efficents
                 #distance==> If the points are close to the point ...they have given higher weightage  

.RandomizedSearchCV ==> Its same as GridSearchCV but reduces the computation by searching  a subset of the parameters
                        here instead of 1 to 31 times it tries randomly only 10 RANDOM numbers between 1 and 31 and still give good accuracy                 
                 
.metrics ==> Classification accuracy: percentage of correct predictions     


            Model evaluation procedures
            ===========================
            .Training and testing on the same data(Rewards overly complex models that "overfit" the training data and won't necessarily generalize)
            .Train/test split(Split the dataset into two pieces, so that the model can be trained and tested on different data
                              Better estimate of out-of-sample performance, but still a "high variance" estimate)
            .K-fold cross-validation(Systematically create "K" train/test splits and average the results together
                                     Even better estimate of out-of-sample performance..Runs "K" times slower than train/test split)                      
                                   
            What is the purpose of model evaluation, and what are some common evaluation procedures?
            What is the usage of classification accuracy, and what are its limitations?
            How does a confusion matrix describe the performance of a classifier?
            What metrics can be computed from a confusion matrix?
            How can you adjust classifier performance by changing the classification threshold?
            What is the purpose of an ROC curve?
            How does Area Under the Curve (AUC) differ from classification accuracy?   
            
            Which metrics should you focus on?

            Choice of metric depends on your business objective
            Spam filter (positive class is "spam"): Optimize for precision or specificity because false negatives (spam goes to the inbox) are more acceptable than false positives (non-spam is caught by the spam filter)
            Fraudulent transaction detector (positive class is "fraud"): Optimize for sensitivity because false positives (normal transactions that are flagged as possible fraud) are more acceptable than false negatives (fraudulent transactions that are not detected)
            
             
            Adjusting the classification threshold (say define 0.5>probability only than report or 0.3>probabilty higher )

.TUNNING PARAMETERS ==> Remove 'stop_words' ,use n_grams,max_df,min_df

####Different datasets repository 
http://archive.ics.uci.edu/ml/datasets.html
http://scikit-learn.org/stable/datasets/ (reading about datasets)


SKLEARN library in python for all algos
#######################################
Each row is an observation (also known as: sample, example, instance, record)
Each column is a feature (also known as: predictor, attribute, independent variable, input, regressor, covariate)
Each value we are predicting is the response (also known as: target, outcome, label, dependent variable)

Requirements for working with data in scikit-learn
==================================================
Features and response are separate objects
Features and response should be numeric
Features and response should be NumPy arrays
Features and response should have specific shapes

classifier is like takes some data as input and gives out a output (which is also called LABELS)its like a set of rules 
algorithms are like intelligence which makes use of classifier as Input/Output


We call **vectorization** the general process of turning a collection of text documents into numerical feature vectors.
                                                