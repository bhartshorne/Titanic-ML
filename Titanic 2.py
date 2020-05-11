import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.model_selection import cross_val_score
import os

try:
    os.chdir(r'C:\Users\Billy Hansen\Desktop\Kaggle Practice\Titanic')
except:
    # Enter Brett's path here.
    doggydoggy

# importing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.shape)
print(test.shape)


# Check Target Variable distribution
train.Survived.value_counts(normalize = True)

# Check Relationships to target variable

# Percentage of total by gender
train.Sex.value_counts(normalize = True)

# Percentage of survival by gender
train['Survived'].groupby(train['Sex']).mean() # Looks like Sex is a very important variable

# Percentage of total by PClass
train.Pclass.value_counts(normalize = True)

# Percentage survived by PClass
train['Survived'].groupby(train['Pclass']).mean() # Another Useful Variable

# Percentage of total by embarked
train.Embarked.value_counts(normalize=True)

# Percentage survived by embarked

'''Does embarked really matter or is this difference random?
Intuitivelly it seemms like it wouldn't matter much which port you took off from,
especially because you are assitrain['Survived'].groupby(train['Embarked']).mean()
gned to a cabin anyway.

Let's look at some correlation stats for our ordinal variables. 
'''

train[['Survived', 'Parch', 'SibSp']].corr(method='spearman') # Not a strong correlation.

'''Looks like ther are slight, positive correlations. Now for feature engineering.
So not to repeat steps I'm going to join both datasets and then break them apart when I'm done.'''

df = train.append(test, sort=True)

# Drop Passenger ID
df = df.drop(columns = ['PassengerId'])

# Function to split and re-group name column
def name_func(row):
    if row['Name'].split(' ')[1] == 'Mr.':
        return 'Mr.'
    elif row['Name'].split(' ')[1] == 'Mrs.':
        return 'Mrs.'
    elif row['Name'].split(' ')[1] ==  'Miss.':
        return 'Miss.'
    elif row['Name'].split(' ')[1] ==  'Dr.':
        return 'Dr.'
    else:
        return 'Other'


df['Name Split'] = df.apply(name_func, axis=1)

# Look at Name Split stats
df['Name Split'].value_counts(normalize=True)

df['Survived'].groupby(df['Name Split']).mean() # looks interested but maybe too correlated with Sex variable?

# Drop Name Column
df = df.drop(columns = ['Name'])

# Check for Nulls
df.isnull().sum()

# We'll fill embarked with the mode of the column
df.Embarked.value_counts()
df.Embarked = df.Embarked.fillna('S')

# Look at Fare Column
df.Fare.describe()

# Replace null with mean
df.Fare = df.Fare.fillna(df.Fare.mean())

# Look at Age Column
df.Age.describe()
df.Age.isnull().sum()

# # # # Let's fill age NAs with a ML model

# Drop irrelevant columns for this model - could redo this once I have useful data for cabin and ticket
age = df.drop(columns = ['Cabin', 'Survived', 'Ticket'])

# Pclass should be a categorical variable
age['Pclass'] = age['Pclass'].astype('object')

# Create Dummies
age = pd.get_dummies(age, columns = ['Name Split', 'Sex', 'Pclass', 'Embarked'], drop_first = True)

# Create train and predict datasets
age_train = age[age['Age'].notnull()]
age_test = age[age['Age'].isnull()]
age_test = age_test.drop(columns = ['Age'])
len(age_train)
len(age_test)

# Grab target vector
age_X = age_train.drop(columns = ['Age'])
age_y = age_train['Age']

# Create Random Forest model
rf = RandomForestRegressor(n_estimators=100)

# Evaluate model using five fold cross validation
scores = cross_val_score(rf, age_X, age_y, cv=5, scoring='neg_mean_squared_error')

# Mean score with 95% confidence interval
print("RF MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# Create Adaboost Model
ad = AdaBoostRegressor(n_estimators=100, learning_rate=1)

# Evaluate model using five fold cross validation
scores = cross_val_score(ad, age_X, age_y, cv=5, scoring='neg_mean_squared_error')

# Mean score with 95% confidence interval
print("Adaboost MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# I'll use RF for now - train model on all data
rf.fit(age_X, age_y)

# Make preds
age_preds = np.around(rf.predict(age_test))

# Replace age nulls in original dataframe
df.loc[df['Age'].isnull(), 'Age'] = age_preds

df.isnull().sum()
