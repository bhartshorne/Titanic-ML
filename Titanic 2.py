import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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
train.columns.values

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

'''Now for some feature engineering.
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

# Let's fill NAs with a model after we prep the data some more.


# Pclass should be categorical because it's not a nominal variable
df.Pclass = df.Pclass.astype('object')

# Create Dummy Variables for categorical variables
df = pd.get_dummies(df, columns = ['Name Split', 'Sex', 'Embarked', 'Pclass'], drop_first=True)

df.dtypes



