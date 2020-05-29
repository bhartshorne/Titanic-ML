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
df.head(5)
df.columns
df.info() #pasId, Ticket

#Remove trivial columns
df = df.drop('PassengerId', axis = 1)
df = df.drop('Ticket', axis = 1)

df.isnull().sum()

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

#Crosstabulation of all titles by sex (T for transpose)
title_ct = pd.crosstab(df.Title, df.Sex).T

df.Title.unique()
#Replacing misspelled and misc titles
df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer',\
                        'Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr',\
                        'Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],\
                        inplace=True)

#Average age by initials
df.groupby('Title')['Age'].mean()

## Assigning the NaN Values with the Ceil values of the mean ages
df.loc[(df.Age.isnull())&(df.Title=='Mr'),'Age']= math.ceil(df[df.Title == 'Mr']['Age'].mean())
df.loc[(df.Age.isnull())&(df.Title=='Mrs'),'Age']= math.ceil(df[df.Title == 'Mrs']['Age'].mean())
df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age']= math.ceil(df[df.Title == 'Master']['Age'].mean())
df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age']= math.ceil(df[df.Title == 'Miss']['Age'].mean())
df.loc[(df.Age.isnull())&(df.Title=='Other'),'Age']= math.ceil(df[df.Title == 'Other']['Age'].mean())

#Any more null values
df.Age.isnull().any() #false

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
f,ax = plt.subplots(1,2, figsize = (12,6))
sns.barplot('SibSp', 'Survived', data = df, ax=ax[0])
ax[0].set_title('Survival by number of Sib/Sp')
sns.factorplot('SibSp', 'Survived', data = df, ax = ax[1])
ax[1].set_title('Survival by number of Sib/Sp')
plt.close(2)
plt.show()

pd.crosstab(df.SibSp, df.Pclass)

# PARCH

#Parents children

pd.crosstab(df.Parch, df.Pclass)

#Visualization
f,ax = plt.subplots(1,2, figsize = (12,6))
sns.barplot('Parch', 'Survived', data = df, ax=ax[0])
ax[0].set_title('Survival by number of Par/Ch')
sns.factorplot('Parch', 'Survived', data = df, ax = ax[1])
ax[1].set_title('Survival by number of Par/Ch')
plt.close(2)
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


# 2) FEATURE ENGINEERING -- not all features are important. WE cna get or add new features
#                           by observing or extracting info from other features.


# Age_group -- age is a continuous variable which is a problem in Machine Learning Models.
#             We need to convert continuous values into categorical values.
#             Can use binning or normalisation.

df['Age_group'] = 0
df.loc[df['Age'] <= 16, 'Age_group'] = 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age_group'] = 1
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age_group'] = 2
df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age_group'] = 3
df.loc[df['Age'] > 64, 'Age_group'] = 4
df.head(2)

#Number of passengers in each age group
df.Age_group.value_counts()

#Visualization
sns.set( rc = {"lines.linewidth":3})
sns.factorplot('Age_group', 'Survived', col = 'Pclass', data = df)

# Family size and Alone
df['Family_size'] = 0
df['Family_size'] = df['SibSp'] + df['Parch']
df['Alone'] = 0
df.loc[df['Family_size'] == 0,'Alone'] = 1
df.head(10)

#Visualization
f,ax = plt.subplots(1,2,figsize = (12,6))
sns.factorplot('Family_size', 'Survived', data=df, ax = ax[0])
ax[0].set_title('Survival by Family Size')
sns.factorplot('Alone', 'Survived', data = df, ax = ax[1])
ax[1].set_title('Survival (Alone of Not)')
plt.close(2)
plt.close(3)
plt.show()

#Important feature, examine more
sns.factorplot('Alone','Survived',hue = 'Sex', col = 'Pclass', data = df)
plt.show()

# Fare_range - splitting into 4 equally spaced ranges using pd.qcut
df['Fare_range'] = pd.qcut(df['Fare'], 4)

df.groupby(df['Fare_range']).Survived.mean() #as price range and survived are correlated

#Converting to singleton values
df['Fare_cat'] = 0
df.loc[df['Fare'] <= 7.91, 'Fare_cat'] = 0
df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare_cat'] = 1
df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare_cat'] = 2
df.loc[(df['Fare'] > 31) & (df['Fare'] <= 513), 'Fare_cat'] = 3
df.head(2)

#Visualization
sns.factorplot('Fare_cat', 'Survived', hue = 'Sex', data = df)
plt.show()

#Converting String Values to Numeric -- cannot pass strings to machine learning models
#Dummy variables -- easier code below

df['Sex'].replace(['male', 'female'], [0,1], inplace = True)
df['Embarked'].replace(['S','C','Q'], [0,1,2], inplace = True)
df['Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

'''
Dropping UnNeeded Features
Name--> We don't need name feature as it cannot be converted into any categorical value.
Age--> We have the Age_group feature, so no need of this.
Ticket--> It is any random string that cannot be categorised.
Fare--> We have the Fare_cat feature, so unneeded
Cabin--> A lot of NaN values and also many passengers have multiple cabins. So this is a useless feature.
Fare_Range--> We have the fare_cat feature.
PassengerId--> Cannot be categorised.'''
df.drop(['Name', 'Age','Fare', 'Cabin', 'Fare_range'], axis = 1, inplace = True)

#Visualization
sns.heatmap(df.corr(), annot = True, cmap = 'RdYlGn', linewidth=0.2,\
             annot_kws={'size':20})
f = plt.gcf()
f.set_size_inches(15,12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
#High corr between parch and famsize, sibsp and famsize. High negative between alone and famsize
# Why not removing some? Multicolinearity?


# 3) PREDICTIVE MODELING -- using clasification algorithms

#importing all the required ML packages - bad practice to import here?
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

#Splitting the data
train,test=train_test_split(df,test_size=0.3,random_state=0,stratify=df['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=df[df.columns[1:]]
Y=df['Survived']


#Radial SVM (rbf kernel)
model = svm.SVC(C = 1.0)
model.fit(train_X,train_Y)
pred1 = model.predict(test_X)
print('Accuracy for rbf SVM is ', metrics.accuracy_score(pred1,test_Y))

#Radial SVM (rbf kernel)
model = svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(train_X,train_Y)
pred2 = model.predict(test_X)
print('Accuracy for Linear SVM is ', metrics.accuracy_score(pred2,test_Y))

#Logistic Regression
model = LogisticRegression()
model.fit(train_X, train_Y)
pred3 = model.predict(test_X)
print('Accuracy for Logistic Regression is ', metrics.accuracy_score(pred3,test_Y))

#Decision Tree
model = DecisionTreeClassifier()
model.fit(train_X,train_Y)
pred4 = model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(pred4,test_Y))

#K-NearestNeighbor
model = KNeighborsClassifier()
model.fit(train_X, train_Y)
pred5 = model.predict(test_X)
print('The accuracy of KNN is',metrics.accuracy_score(pred5, test_Y))

#Experimenting with different neighbors values
a_index = list(range(1,11))
a = pd.Series()
x = list(range(0,11))
for i in list(range(1,11)):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(train_X, train_Y)
    pred = model.predict(test_X)
    a = a.append(pd.Series(metrics.accuracy_score(pred, test_Y)))
plt.plot(a_index, a)
plt.xticks(x)
fig = plt.gcf()
fig.set_size_inches(12,6)
plt.show()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())
# 9 neighbors gives the most accurace result

#Naive Bayes
model = GaussianNB()
model.fit(train_X,train_Y)
pred6 = model.predict(test_X)
print('The accuracy of the NaiveBayes is',metrics.accuracy_score(pred6,test_Y))

#Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y)
pred7 = model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(pred7,test_Y))

'''IMPORTANT NOTE -- accuracy is not the only factor that determines robustness of 
a model. Training and testing data changes, the accuracy will change too, it' called
model variance. To overcome this we use a generalized model - cross validation'''

### CROSS VALIDATION
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits = 10, random_state = 22)
cv_mean = []
accuracy = []
std = []
classifiers = ['Linear Svm','Radial Svm','Logistic Regression',\
               'KNN','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),\
        KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),\
        RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model, X,Y, cv = kfold, scoring = 'accuracy')
    cv_result = cv_result
    cv_mean.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2 = pd.DataFrame({'CV Mean':cv_mean,'Std':std}, index = classifiers)
new_models_dataframe2

#Visualization of accuracies
plt.subplots(figsize = (12,6))
box = pd.DataFrame(accuracy, index=classifiers)
box.T.boxplot()

#Visualization of Average CV Mean accuracy
new_models_dataframe2['CV Mean'].plot.barh(width = 0.8)
plt.title('Average CV Mean Accuracy')
fig = plt.gcf()
fig.set_size_inches(8,5)
plt.show()

# Confusion Matrices for all models
f,ax=plt.subplots(3,3,figsize=(12,10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for rbf-SVM')
y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Linear-SVM')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
ax[0,2].set_title('Matrix for KNN')
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Random-Forests')
y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
ax[2,0].set_title('Matrix for Naive Bayes')
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()
 
# Hyper-Parameter tuning -- tune to change learning rate of algorithm get better model
# Tuning parameters for 2 best classifiers - SVM and Random Forest

#SVM Grid Search
from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd = GridSearchCV(estimator = svm.SVC(), param_grid = hyper, verbose = True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)

#Random Forests
n_estimators = range(100, 1000, 100)
hyper = {'n_estimators':n_estimators}
gd = GridSearchCV(estimator = RandomForestClassifier(random_state = 0), param_grid = hyper,\
                  verbose = True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)
#Best scores: SVM - 82.82% (C = 0.5, gamma = 0.1), RFC - 81.8% (n_estimators = 900)

### ENSEMBLING - Inrease accuracy by combining models. 

# Voting Classifier
from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], 
                       voting='soft').fit(train_X,train_Y)
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))
cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())

#Bagging 
#KNN
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(base_estimator = KNeighborsClassifier(n_neighbors = 3), random_state = 0,\
                          n_estimators = 700)
model.fit(train_X,train_Y)
pred = model.predict(test_X)
print('The accuracy for bagged KNN is:', metrics.accuracy_score(test_Y, pred))
result = cross_val_score(model, X, Y, cv =10, scoring = 'accuracy')
print('The cross validated score for bagged Decision Tree is:', result.mean())

#Bagging
#Decision Tree
model = BaggingClassifier(base_estimator = DecisionTreeClassifier(), random_state = 0,\
                          n_estimators = 100)
model.fit(train_X,train_Y)
pred = model.predict(test_X)
print('The accuracy for bagged Decision Tree is:', metrics.accuracy_score(pred, test_Y))
res = cross_val_score(model, X, Y, cv = 10, scoring = 'accuracy')
print('The cross validated score for bagged Decision Tree is:', res.mean())

# Boosting -- iterative approach to imporove the accuracy
# Adaboost
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators = 200, random_state = 0, learning_rate = 0.1)
result = cross_val_score(ada, X, Y, scoring = 'accuracy', cv = 10)
print('The cross validated score of AdaBoost is:', result.mean())

#Stochastic Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())

#XGBoost
import xgboost as xg
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result=cross_val_score(xgboost,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())

#Highest accuracy - AdaBoost
#Hyper-parameter tuning for AdaBoost
n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)

#Takes forever to run - best model is 83.16% accuracy with n_estimators = 200, learning_rate = 0.05

#Confusion matrix for Best Model
ada = AdaBoostClassifier(n_estimators = 200, random_state = 0, learning_rate = 0.05)
result = cross_val_predict(ada, X, Y, cv = 10)
sns.heatmap(confusion_matrix(Y, result), cmap = 'winter', annot = True, fmt = '2.0f')
plt.show()

#Feature Importance in different models
f, ax = plt.subplots(2,2, figsize = (15,12))
model = RandomForestClassifier(n_estimators = 500, random_state = 0)
model.fit(X,Y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending = True).plot.barh(width = 0.8, ax = ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')
model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')
model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')
model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')
ax[1,1].set_title('Feature Importance in XgBoost')
plt.show()


#Submission of AdaBoostClassifier
ada.fit(train_X, train_Y)

test2 = pd.read_csv('test.csv')
pred = ada.predict(test)
sub = test2[['PassengerId']]
sub['Survived'] = pred

sub['Survived'] = sub['Survived'].astype('int32')

sub.to_csv('Submission.csv', index = False)
