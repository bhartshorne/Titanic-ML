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
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 1) EDA

print(len(train)) #891 entries
train.head(5)
train.columns
train.info() #pasId, Ticket

#Remove trivial columns
train = train.drop('PassengerId', axis = 1)
train = train.drop('Ticket', axis = 1)

train.isnull().sum()

# Feature exploration and Visualizations

# SURVIVED
#Visualization
f, ax  = plt.subplots(1,2, figsize = (12,6))
train['Survived'].value_counts().plot.pie(explode=[0,0.1], ax=ax[0],autopct='%1.1f%%',\
     shadow = True, colors = ['coral', 'skyblue']) #autopct converts to pctage , #legend=True if wanted
#ax[0].legend(loc = 'upper left', frameon = False, ncol=2)
ax[0].set_ylabel('')
ax[0].set_title('Survived')
sns.countplot(data = train, x = train.Survived,\
              ax = ax[1], palette = ['coral', 'skyblue'])
ax[1].set_title('Survived')
plt.show()

# SURVIVED BY SEX
train.groupby(['Survived', 'Sex'])['Survived'].count()

#Visualization
f, ax  = plt.subplots(1,2, figsize = (12,6))
sns.set_style('darkgrid')
train[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0],\
     legend = True, color = 'skyblue')
ax[0].set_ylabel('Percentage')
ax[0].set_xlabel('')
ax[0].legend(frameon = False)
ax[0].set_title('Survived by Gender')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1], palette = ['coral', 'skyblue'])
ax[1].set_title('Survived v Perished by Gender')
ax[1].legend(frameon = False, labels = ['Perished', 'Survived'])
plt.show()

# SEX IS AN IMPORTANT VARIABLE #


# PCLASS
pclass1_fare = train[train['Pclass']==1]['Fare'].mean() #$84
pclass2_fare = train[train['Pclass']==2]['Fare'].mean() #$20
pclass3_fare = train[train['Pclass']==3]['Fare'].mean() #$14

# SURVIVED BY PCLASS
#Cross tab
pd.crosstab(train.Pclass,train.Survived,margins=True)
#Survival rate as expected - highest pclass1, mid class2, lowest class3


#Visualization
f, ax  = plt.subplots(1,2, figsize = (12,6))
train['Pclass'].value_counts().plot.pie(explode=[0,0.1,0.1], ax=ax[0],autopct='%1.1f%%',\
     shadow = True, colormap = 'Accent') #autopct converts to pctage , #legend=True if wanted
#ax[0].legend(loc = 'upper left', frameon = False, ncol=2)
ax[0].set_ylabel('')
ax[0].set_title('Class Distribution')
sns.countplot('Pclass', data = train, hue = 'Survived', ax = ax[1], palette = ['coral', 'skyblue'])
ax[1].set_xlabel('Class')
ax[1].set_title('Survival by Class')
plt.show()

# SURVIVED BY PCLASS AND SEX
pd.crosstab([train.Sex,train.Survived],train.Pclass,margins=True)
#Class 1 and 2 almost all female survived
#Men had a signifincatly lower chance of survival

#Visualization
sns.factorplot('Pclass','Survived',hue='Sex',data=train)
plt.show()

######### PCLASS SEEMS TO BE AN IMPORTANT VARIABLE ###########

# AGE
train['Age'].min() #0.42
train['Age'].max() #80

len(train[train.Age < 3]) #24

train['Age'].mean() #29.7
train['Age'].median() #28

#Visualization
f, ax = plt.subplots(1,2, figsize = (12,6))
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', split = True,\
               data = train, ax = ax[0], palette = ['coral', 'skyblue'] )
ax[0].set_title('Survival by Pclass and Age')
ax[0].legend(frameon = False)
ax[0].set_yticks(range(0,110,10))
sns.violinplot(x = 'Sex', y = 'Age', split = True, hue = 'Survived', data = train,\
               ax = ax[1], palette = ['coral', 'skyblue'])
ax[1].set_title('Survival by Sex and Age')
ax[1].legend(frameon = False)
ax[1].set_yticks(range(0,110,10))
plt.show()

#Dealing with NaN values in age columns -- 177 missing values.
#Extracting Title from the Name Col and appending it to train dataset (extracting any letter comb followed by a dot)
train['Title'] = train.Name.str.extract('([A-Za-z]+)\.') #regex ([]+)\. 1 or more, escape

#Crosstabulation of all titles by sex (T for transpose)
title_ct = pd.crosstab(train.Title, train.Sex).T

train.Title.unique()
#Replacing misspelled and misc titles
train['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer',\
                        'Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr',\
                        'Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],\
                        inplace=True)

#Average age by initials 
train.groupby('Title')['Age'].mean()

## Assigning the NaN Values with the Ceil values of the mean ages
train.loc[(train.Age.isnull())&(train.Title=='Mr'),'Age']= math.ceil(train[train.Title == 'Mr']['Age'].mean())
train.loc[(train.Age.isnull())&(train.Title=='Mrs'),'Age']= math.ceil(train[train.Title == 'Mrs']['Age'].mean())
train.loc[(train.Age.isnull())&(train.Title=='Master'),'Age']= math.ceil(train[train.Title == 'Master']['Age'].mean())
train.loc[(train.Age.isnull())&(train.Title=='Miss'),'Age']= math.ceil(train[train.Title == 'Miss']['Age'].mean())
train.loc[(train.Age.isnull())&(train.Title=='Other'),'Age']= math.ceil(train[train.Title == 'Other']['Age'].mean())

#Any more null values
train.Age.isnull().any() #false

#Visualization of survival by age (bins of 5)
f, ax = plt.subplots(1,2, figsize = (12,6))
train[train['Survived']==0].Age.plot.hist(bins = 20, ax = ax[0], color = 'coral', edgecolor = 'black')
ax[0].set_title('Perished')
ax[0].set_xlabel('Age bins')
ax[0].set_xticks(range(0,85,5))
train[train['Survived']==1].Age.plot.hist(ax=ax[1], bins = 20, color = 'skyblue', edgecolor = 'black')
ax[1].set_title('Survived')
ax[1].set_xlabel('Age bins')
ax[1].set_xticks(range(0,85,5))
plt.show()
#High survival for children 

#Factorplot of Titles by P-class #col specifies args like Facetgrid
sns.factorplot('Pclass','Survived', col = 'Title', data = train)
plt.ylabel('Survived')
plt.show()


# EMBARKED
pd.crosstab([train.Embarked,train.Pclass], [train.Sex,train.Survived], margins = True)

#Visualization
sns.factorplot('Embarked', 'Survived', data = train)
fig = plt.gcf() #get current figure
fig.set_size_inches(5,3)
plt.show()

#Port C had a highest chance of survival
f, ax = plt.subplots(2,2, figsize = (16,8))
sns.countplot('Embarked', data = train, ax = ax[0,0])
ax[0,0].set_title('Number of Passengers Embarked')
sns.countplot('Embarked', hue = 'Sex', data = train, ax = ax[0,1])
ax[0,1].set_title('Male-Female Split for Each Port')
sns.countplot('Embarked', hue = 'Survived', data = train, ax = ax[1,0])
ax[1,0].set_title('Survival for Each Port')
sns.countplot('Embarked', hue = 'Pclass', data = train, ax = ax[1,1])
ax[1,1].set_title('Pclass for Each Port')

#Pclass survival by gender for each port
sns.factorplot('Pclass', 'Survived',hue = 'Sex', col = 'Embarked', data = train)
plt.show()

#Filling Embarked Nan (WHY AFTER THE VISUALIZATIONS?)
#Max passengers embarked in port S, thus fill in with S
train['Embarked'].fillna('S',inplace=True)

#check
train.Embarked.isnull().any() #False


# SIBSP

#Sib - number of siblings on the ship
#Sp - husband/wife
pd.crosstab(train['SibSp'], train['Survived'], margins = True)

#Visualization
f,ax = plt.subplots(1,2, figsize = (12,6))
sns.barplot('SibSp', 'Survived', data = train, ax=ax[0])
ax[0].set_title('Survival by number of Sib/Sp')
sns.factorplot('SibSp', 'Survived', data = train, ax = ax[1])
ax[1].set_title('Survival by number of Sib/Sp')
plt.close(2)
plt.show()

pd.crosstab(train.SibSp, train.Pclass)

# PARCH
#Parents children

pd.crosstab(train.Parch, train.Pclass)

#Visualization
f,ax = plt.subplots(1,2, figsize = (12,6))
sns.barplot('Parch', 'Survived', data = train, ax=ax[0])
ax[0].set_title('Survival by number of Par/Ch')
sns.factorplot('Parch', 'Survived', data = train, ax = ax[1])
ax[1].set_title('Survival by number of Par/Ch')
plt.close(2)
plt.show()

# FARE

train.Fare.max() #$512
train.Fare.min() #$0
train.Fare.mean() #$32

f, ax = plt.subplots(1,3, figsize = (14,7))
sns.distplot(train[train['Pclass']==1].Fare, ax = ax[0])
ax[0].set_title('Fares in Pclass1')
sns.distplot(train[train['Pclass']==2].Fare, ax = ax[1])
ax[1].set_title('Fares in Pclass2')
sns.distplot(train[train['Pclass']==3].Fare, ax = ax[2])
ax[2].set_title('Fares in Pclass3')

###### CORRELATION MATRIX
#Only Numeric values, no high correlations - no multicolinearity
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.show()

# 2) FEATURE ENGINEERING -- not all features are important. WE cna get or add new features
#                           by observing or extracting info from other features. 

# Age_band -- age is a continuous variable which is a problem in Machine Learning Models.
#             We need to convert continuous values into categorical values.
#             Can use binning or normalisation.

train['Age_group'] = 0
train.loc[train['Age'] <= 16, 'Age_band'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age_group'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age_group'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age_group'] = 3
train.loc[train['Age'] > 64, 'Age_group'] = 4
train.head(2)

#Number of passengers in each age group
train.Age_group.value_counts()

#Visualization
sns.set( rc = {"lines.linewidth":3})
sns.factorplot('Age_group', 'Survived', col = 'Pclass', data = train)

# Family size and Alone 
train['Family_size'] = 0
train['Family_size'] = train['SibSp'] + train['Parch']
train['Alone'] = 0
train.loc[train['Family_size'] == 0,'Alone'] = 1
train.head(10)

#Visualization
f,ax = plt.subplots(1,2,figsize = (12,6))
sns.factorplot('Family_size', 'Survived', data=train, ax = ax[0])
ax[0].set_title('Survival by Family Size')
sns.factorplot('Alone', 'Survived', data = train, ax = ax[1])
ax[1].set_title('Survival (Alone of Not)')
plt.close(2)
plt.close(3)
plt.show()

#Important feature, examine more
sns.factorplot('Alone','Survived',hue = 'Sex', col = 'Pclass', data = train)
plt.show()

# Fare_range - splitting into 4 equally spaced ranges using pd.qcut
train['Fare_range'] = pd.qcut(train['Fare'], 4)

train.groupby(train['Fare_range']).Survived.mean() #as price range and survived are correlated

#Converting to singleton values
train['Fare_cat'] = 0
train.loc[train['Fare'] <= 7.91, 'Fare_cat'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare_cat'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare_cat'] = 2
train.loc[(train['Fare'] > 31) & (train['Fare'] <= 513), 'Fare_cat'] = 3
train.head(2)

#Visualization 
sns.factorplot('Fare_cat', 'Survived', hue = 'Sex', data = train)
plt.show()






















