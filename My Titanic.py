#Imports
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')


#Importing dataset
os.chdir('C:\\Users\\Owner\\Desktop\\Machine Learning Practice\\Titanic ML')
df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 1) EDA
print(len(df)) #891 entries
print (df.head(5))
print(df.columns)
print(df.info()) #pasId, Ticket

df.isnull().sum()
test.isnull().sum()
# Feature exploration and Visualizations

# SURVIVED

#Visualization
f, ax  = plt.subplots(1,2, figsize = (12,6))
df['Survived'].value_counts().plot.pie(explode=[0,0.1], ax=ax[0],autopct='%1.1f%%',\
     shadow = True, colors = ['coral', 'skyblue']) #autopct converts to pctage , #legend=True if wanted
#ax[0].legend(loc = 'upper left', frameon = False, ncol=2)
ax[0].set_ylabel('')
ax[0].set_title('Survived')
sns.countplot(data = df, x = df.Survived,\
              ax = ax[1], palette = ['coral', 'skyblue'])
ax[1].set_title('Survived')
plt.show()

# SURVIVED BY SEX
df.groupby(['Survived', 'Sex'])['Survived'].count()

#Visualization
f, ax  = plt.subplots(1,2, figsize = (12,6))
sns.set_style('darkgrid')
df[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0],\
     legend = True, color = 'skyblue')
ax[0].set_ylabel('Percentage')
ax[0].set_xlabel('')
ax[0].legend(frameon = False)
ax[0].set_title('Survived by Gender')
sns.countplot('Sex',hue='Survived',data=df,ax=ax[1], palette = ['coral', 'skyblue'])
ax[1].set_title('Survived v Perished by Gender')
ax[1].legend(frameon = False, labels = ['Perished', 'Survived'])
plt.show()

# SEX IS AN IMPORTANT VARIABLE #

# PCLASS

pclass1_fare = df[df['Pclass']==1]['Fare'].mean() #$84
pclass2_fare = df[df['Pclass']==2]['Fare'].mean() #$20
pclass3_fare = df[df['Pclass']==3]['Fare'].mean() #$14

# SURVIVED BY PCLASS

#Cross tab
pd.crosstab(df.Pclass,df.Survived,margins=True)
#Survival rate as expected - highest pclass1, mid class2, lowest class3


#Visualization
f, ax  = plt.subplots(1,2, figsize = (12,6))
df['Pclass'].value_counts().plot.pie(explode=[0,0.1,0.1], ax=ax[0],autopct='%1.1f%%',\
     shadow = True, colormap = 'Accent') #autopct converts to pctage , #legend=True if wanted
#ax[0].legend(loc = 'upper left', frameon = False, ncol=2)
ax[0].set_ylabel('')
ax[0].set_title('Class Distribution')
sns.countplot('Pclass', data = df, hue = 'Survived', ax = ax[1], palette = ['coral', 'skyblue'])
ax[1].set_xlabel('Class')
ax[1].set_title('Survival by Class')
plt.show()

# SURVIVED BY PCLASS AND SEX

pd.crosstab([df.Sex,df.Survived],df.Pclass,margins=True)
#Class 1 and 2 almost all female survived
#Men had a signifincatly lower chance of survival

#Visualization
sns.factorplot('Pclass','Survived',hue='Sex',data=df)
plt.show()

######### PCLASS SEEMS TO BE AN IMPORTANT VARIABLE ###########

# AGE

df['Age'].min() #0.42
df['Age'].max() #80

len(df[df.Age < 3]) #24

df['Age'].mean() #29.7
df['Age'].median() #28

#Visualization
f, ax = plt.subplots(1,2, figsize = (12,6))
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', split = True,\
               data = df, ax = ax[0], palette = ['coral', 'skyblue'] )
ax[0].set_title('Survival by Pclass and Age')
ax[0].legend(frameon = False)
ax[0].set_yticks(range(0,110,10))
sns.violinplot(x = 'Sex', y = 'Age', split = True, hue = 'Survived', data = df,\
               ax = ax[1], palette = ['coral', 'skyblue'])
ax[1].set_title('Survival by Sex and Age')
ax[1].legend(frameon = False)
ax[1].set_yticks(range(0,110,10))
plt.show()

#Dealing with NaN values in age columns -- 177 missing values.
#Extracting Title from the Name Col and appending it to train dataset (extracting any letter comb followed by a dot)
df['Title'] = df.Name.str.extract('([A-Za-z]+)\.') #regex ([]+)\. 1 or more, escape
test['Title'] = test.Name.str.extract('([A-Za-z]+)\.')
#Crosstabulation of all titles by sex (T for transpose)
title_ct = pd.crosstab(df.Title, df.Sex).T

df.Title.unique()
#Replacing misspelled and misc titles
df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer',\
                        'Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr',\
                        'Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],\
                        inplace=True)
test['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer',\
                        'Col','Rev','Capt','Sir','Don', 'Dona'],['Miss','Miss','Miss','Mr',\
                        'Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mrs'],\
                        inplace=True)                                                        
                                                                 
#Average age by initials
df.groupby('Title')['Age'].mean()

## Assigning the NaN Values with the Ceil values of the mean ages
df.loc[(df.Age.isnull())&(df.Title=='Mr'),'Age']= math.ceil(df[df.Title == 'Mr']['Age'].mean())
df.loc[(df.Age.isnull())&(df.Title=='Mrs'),'Age']= math.ceil(df[df.Title == 'Mrs']['Age'].mean())
df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age']= math.ceil(df[df.Title == 'Master']['Age'].mean())
df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age']= math.ceil(df[df.Title == 'Miss']['Age'].mean())
df.loc[(df.Age.isnull())&(df.Title=='Other'),'Age']= math.ceil(df[df.Title == 'Other']['Age'].mean())

test.loc[(test.Age.isnull())&(test.Title=='Mr'),'Age']= math.ceil(test[df.Title == 'Mr']['Age'].mean())
test.loc[(test.Age.isnull())&(test.Title=='Mrs'),'Age']= math.ceil(test[test.Title == 'Mrs']['Age'].mean())
test.loc[(test.Age.isnull())&(test.Title=='Master'),'Age']= math.ceil(test[test.Title == 'Master']['Age'].mean())
test.loc[(test.Age.isnull())&(test.Title=='Miss'),'Age']= math.ceil(test[test.Title == 'Miss']['Age'].mean())
test.loc[(test.Age.isnull())&(test.Title=='Other'),'Age']= math.ceil(test[test.Title == 'Other']['Age'].mean())

#Any more null values
df.Age.isnull().any() #false
test.Age.isnull().any()

#Visualization of survival by age (bins of 5)
f, ax = plt.subplots(1,2, figsize = (12,6))
df[df['Survived']==0].Age.plot.hist(bins = 20, ax = ax[0], color = 'coral', edgecolor = 'black')
ax[0].set_title('Perished')
ax[0].set_xlabel('Age bins')
ax[0].set_xticks(range(0,85,5))
df[df['Survived']==1].Age.plot.hist(ax=ax[1], bins = 20, color = 'skyblue', edgecolor = 'black')
ax[1].set_title('Survived')
ax[1].set_xlabel('Age bins')
ax[1].set_xticks(range(0,85,5))
plt.show()
#High survival for children

#Factorplot of Titles by P-class #col specifies args like Facetgrid
sns.factorplot('Pclass','Survived', col = 'Title', data = df)
plt.ylabel('Survived')
plt.show()


# EMBARKED

pd.crosstab([df.Embarked,df.Pclass], [df.Sex,df.Survived], margins = True)

#Visualization
sns.factorplot('Embarked', 'Survived', data = df)
fig = plt.gcf() #get current figure
fig.set_size_inches(5,3)
plt.show()

#Port C had a highest chance of survival
f, ax = plt.subplots(2,2, figsize = (16,8))
sns.countplot('Embarked', data = df, ax = ax[0,0])
ax[0,0].set_title('Number of Passengers Embarked')
sns.countplot('Embarked', hue = 'Sex', data = df, ax = ax[0,1])
ax[0,1].set_title('Male-Female Split for Each Port')
sns.countplot('Embarked', hue = 'Survived', data = df, ax = ax[1,0])
ax[1,0].set_title('Survival for Each Port')
sns.countplot('Embarked', hue = 'Pclass', data = df, ax = ax[1,1])
ax[1,1].set_title('Pclass for Each Port')

#Pclass survival by gender for each port
sns.factorplot('Pclass', 'Survived',hue = 'Sex', col = 'Embarked', data = df)
plt.show()

#Filling Embarked Nan (WHY AFTER THE VISUALIZATIONS?)
#Max passengers embarked in port S, thus fill in with S
df['Embarked'].fillna('S',inplace=True)

#check
df.Embarked.isnull().any() #False

# SIBSP

#Sib - number of siblings on the ship
#Sp - husband/wife
pd.crosstab(df['SibSp'], df['Survived'], margins = True)

#Visualization
sns.factorplot('SibSp','Survived',data=df)
plt.title('SibSp vs Survived')
plt.show()


pd.crosstab(df.SibSp, df.Pclass)

# PARCH

#Parents children

pd.crosstab(df.Parch, df.Pclass)

#Visualization
sns.factorplot('Parch', 'Survived', data = df)
plt.title('Survival by number of Par/Ch')
plt.show()

# FARE

df.Fare.max() #$512
df.Fare.min() #$0
df.Fare.mean() #$32

f, ax = plt.subplots(1,3, figsize = (14,7))
sns.distplot(df[df['Pclass']==1].Fare, ax = ax[0])
ax[0].set_title('Fares in Pclass1')
sns.distplot(df[df['Pclass']==2].Fare, ax = ax[1])
ax[1].set_title('Fares in Pclass2')
sns.distplot(df[df['Pclass']==3].Fare, ax = ax[2])
ax[2].set_title('Fares in Pclass3')

###### CORRELATION MATRIX
#Only Numeric values, no high correlations - no multicolinearity
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.show()

##################################################### FEATURE ENGINEERING 
# Trying to append the datasets to apply it to both
appended_df = df.append(test, sort = True )

# Fare Range
appended_df.Fare.min() #0
appended_df.Fare.max() #512

appended_df['Fare_Range'] = pd.qcut(appended_df['Fare'], 4)

# Age Bands
appended_df['Age_group'] = 0
appended_df.loc[appended_df['Age'] <= 16, 'Age_group'] = 'Minor'
appended_df.loc[(appended_df['Age'] > 16) & (appended_df['Age'] <= 32), 'Age_group'] = 'Young Adult'
appended_df.loc[(appended_df['Age'] > 32) & (appended_df['Age'] <= 48), 'Age_group'] = 'Middle Aged'
appended_df.loc[(appended_df['Age'] > 48) & (appended_df['Age'] <= 64), 'Age_group'] = 'Getting Old'
appended_df.loc[appended_df['Age'] > 64, 'Age_group'] = 'Old'

# Family/ Alone
appended_df['Family Size'] = appended_df['Parch'] + appended_df['SibSp']
appended_df['isAlone'] = 0
appended_df.loc[appended_df['Family Size'] == 0, 'isAlone'] = 1

#Correlation matrix
sns.heatmap(appended_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.show()

# Dropping irrelevant columns
appended_df.drop(['Age', 'Cabin', 'Fare', 'Name', 'Parch', 'Pclass', 'PassengerId',\
                 'SibSp', 'Ticket', 'Family Size'], axis = 1, inplace = True)
    
#Not sure if need to drop Family Size 
   
appended_df.head(5)

#################################################################### ONE HOT ENCODING, PIPELINE, 

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer #make_col_trans(OneHotEncoder(), [cols], remainder = passthrough)
from sklearn.pipeline import make_pipeline #make_pipeline(column_trans, ML model)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics #accuracy measure
from sklearn.svm import SVC #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.metrics import confusion_matrix #for confusion matrix

#Split back data to Train, Test
train = appended_df.loc[appended_df['Survived'].notna(), :]
test = appended_df.loc[appended_df['Survived'].isna(), appended_df.columns != 'Survived']

#Grab target Variable
X = train.drop('Survived', axis = 1)
y = train['Survived']

#train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Encoding data
column_trans = make_column_transformer((OneHotEncoder(), ['Embarked', 'Sex', 'Title',\
                                                         'Fare_Range', 'Age_group']),\
                                       remainder = 'passthrough')
sc_column_trans = make_column_transformer((OneHotEncoder(), ['Embarked', 'Sex', 'Title',\
                                                         'Fare_Range', 'Age_group']),\
                                          (StandardScaler(), ['Fare_Range']), remainder = 'passthrough')
#LOGISTIC REGRESSION
logReg = LogisticRegression()
pipe = make_pipeline(column_trans, logReg)
pipe.fit(X_train, y_train)
pred1 = pipe.predict(X_test)
print('The accuracy of Logistic Regression is:', metrics.accuracy_score(pred1, y_test))

#SVM
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
accuracy = []

for kernel in kernels:
    svc = SVC(kernel = kernel, random_state = 42)
    pipe = make_pipeline(column_trans, svc)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    accuracy.append(metrics.accuracy_score(pred, y_test))

print(accuracy) #linear and rbf give the highest accuracy

for kernel in kernels:
    svc = SVC(kernel = kernel, random_state = 42)
    pipe = make_pipeline(sc_column_trans, svc)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    accuracy.append(metrics.accuracy_score(pred, y_test))

print(accuracy)

#DECISION TREE 
criterion = ['gini', 'entropy']
accuracy = []

for crit in criterion:
    dtc = DecisionTreeClassifier(criterion = crit, random_state = 42)
    pipe = make_pipeline(column_trans, dtc)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    accuracy.append(metrics.accuracy_score(pred, y_test))

print(accuracy) #entropy

#RANDOM FOREST
rfc = RandomForestClassifier(n_estimators = 100, random_state = 42)
pipe = make_pipeline(column_trans, rfc)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print('The accuracy of Random Forest is:', metrics.accuracy_score(pred, y_test))

#KNN
neighbors = list(range(1,11,1))
accuracy = []

for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors = n)
    pipe = make_pipeline(column_trans, knn)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    accuracy.append(metrics.accuracy_score(pred, y_test))
    
print(accuracy) #neighbors = 5 highest accuracy

#NAIVE BAYES

nb = GaussianNB()
pipe = make_pipeline(column_trans, nb)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print('The accuracy of Naive Bayes is:', metrics.accuracy_score(pred, y_test))

# CROSS VALIDATION OF ALL MODELS
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_predict
means = []
accuracy = []
std = []
kfold = KFold(n_splits = 10, random_state = 42)
models = [make_pipeline(column_trans,LogisticRegression()),make_pipeline(column_trans, SVC(kernel = 'linear')),\
          make_pipeline(column_trans, SVC(kernel = 'rbf')), make_pipeline(column_trans, DecisionTreeClassifier(criterion = 'entropy')),\
          make_pipeline(column_trans, RandomForestClassifier(n_estimators = 100)),\
          make_pipeline(column_trans, KNeighborsClassifier(n_neighbors = 5)), make_pipeline(column_trans, GaussianNB())]
classifiers = ['LogReg', 'SVC(linear)', 'SCV(rbf)', 'DecisionTree', 'RandForest',\
               'KNN', 'NaiveBayes']

for model in models:
    model = model
    cv_result = cross_val_score(model, X, y, cv= kfold, scoring = 'accuracy')
    cv_result = cv_result
    means.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)

models_df = pd.DataFrame({"CV Mean":means, "Std":std}, index  = classifiers)

#Visualization #eexcluding Naive Bayes because of outlier ~38% on one kfold. ########WHY?
f, ax = plt.subplots(figsize = (12, 6))
box = pd.DataFrame(accuracy[0:6], index = classifiers[0:6])
box.T.boxplot()
plt.show()

models_df['CV Mean'].plot.barh(width = 0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()

#Let's make confusion matrix to see correct/incorrect classifications by each model
f, ax = plt.subplots(3,3,figsize = (12,10))
pred = cross_val_predict(make_pipeline(column_trans,LogisticRegression()), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[0,0], annot = True, fmt='2.0f')
ax[0,0].set_title('Matrix of LogReg')
pred = cross_val_predict(make_pipeline(column_trans, SVC(kernel = 'linear')), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[0,1], annot = True, fmt='2.0f')
ax[0,1].set_title('Matrix of Linear SVC')
pred = cross_val_predict(make_pipeline(column_trans, SVC(kernel = 'rbf')), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[0,2], annot = True, fmt='2.0f')
ax[0,2].set_title('Matrix of RBF SVC')
pred = cross_val_predict(make_pipeline(column_trans, DecisionTreeClassifier(criterion = 'entropy')), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[1,0], annot = True, fmt='2.0f')
ax[1,0].set_title('Matrix of Decision Tree')
pred = cross_val_predict( make_pipeline(column_trans, RandomForestClassifier(n_estimators = 100)), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[1,1], annot = True, fmt='2.0f')
ax[1,1].set_title('Matrix of Random Forest')
pred = cross_val_predict(make_pipeline(column_trans, KNeighborsClassifier(n_neighbors = 5)), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[1,2], annot = True, fmt='2.0f')
ax[1,2].set_title('Matrix of KNN')
pred = cross_val_predict( make_pipeline(column_trans, GaussianNB()), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[2,0], annot = True, fmt='2.0f')
ax[2,0].set_title('Matrix of Naive Bayes')
plt.show()
########################Parameter Tuning

#Shows available params when using gridsearch with pipeline
#make_pipeline(column_trans,DecisionTreeClassifier()).get_params().keys()

#SVC
from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1] #Penalty parameter
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] #Parameter for non-linear hyperplanes
kernel=['rbf','linear']
param = {'svc__C':C, 'svc__kernel' : kernel, 'svc__gamma' : gamma, 'svc__random_state': [42]}
gd = GridSearchCV(estimator = make_pipeline(column_trans,SVC()) , param_grid = param, verbose = True)
gd.fit(X,y)
gd.best_params_
gd.best_score_
# C = 0.3, gamma = 0.7, rbf, random state 42

#Decision Tree fro practice
mss = list(range(10, 500, 20))
max_depth = list(range(1,20,2))
criterion=['gini','entropy']
param = {'decisiontreeclassifier__min_samples_split':mss,'decisiontreeclassifier__max_depth':max_depth,\
         'decisiontreeclassifier__criterion':criterion, 'decisiontreeclassifier__random_state': [42]}
gd = GridSearchCV(estimator = make_pipeline(column_trans,DecisionTreeClassifier()) , param_grid = param, verbose = True)
gd.fit(X,y)
gd.best_params_
gd.best_score_
#entropy, max depth 3, min samples 10

#Logistic Regression
penalty = ['l1', 'l2']
C = np.logspace(-3,3,7)
param = {'logisticregression__penalty':penalty, 'logisticregression__C':C, 'logisticregression__random_state':[42]}
gd = GridSearchCV(estimator = make_pipeline(column_trans,LogisticRegression()) , param_grid = param, verbose = True)
gd.fit(X,y)
print(gd.best_params_)
print(gd.best_score_)
#0.1, l2

X_train.dtypes
y_train.dtypes













#Let's make confusion matrix to see correct/incorrect classifications by each model
f, ax = plt.subplots(3,3,figsize = (12,10))
pred = cross_val_predict(make_pipeline(column_trans,LogisticRegression()), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[0,0], annot = True, fmt='2.0f')
ax[0,0].set_title('Matrix of LogReg')
pred = cross_val_predict(make_pipeline(column_trans, SVC(kernel = 'linear')), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[0,1], annot = True, fmt='2.0f')
ax[0,1].set_title('Matrix of Linear SVC')
pred = cross_val_predict(make_pipeline(column_trans, SVC(kernel = 'rbf')), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[0,2], annot = True, fmt='2.0f')
ax[0,2].set_title('Matrix of RBF SVC')
pred = cross_val_predict(make_pipeline(column_trans, DecisionTreeClassifier(criterion = 'entropy')), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[1,0], annot = True, fmt='2.0f')
ax[1,0].set_title('Matrix of Decision Tree')
pred = cross_val_predict( make_pipeline(column_trans, RandomForestClassifier(n_estimators = 100)), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[1,1], annot = True, fmt='2.0f')
ax[1,1].set_title('Matrix of Random Forest')
pred = cross_val_predict(make_pipeline(column_trans, KNeighborsClassifier(n_neighbors = 5)), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[1,2], annot = True, fmt='2.0f')
ax[1,2].set_title('Matrix of KNN')
pred = cross_val_predict( make_pipeline(column_trans, GaussianNB()), X, y, cv = 10)
sns.heatmap(confusion_matrix(pred, y), ax = ax[2,0], annot = True, fmt='2.0f')
ax[2,0].set_title('Matrix of Naive Bayes')
plt.show()







