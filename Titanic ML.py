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

# looking at size of data
print('Train shape:', train.shape)

# looking at first and last 5 lines of data
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 12)
print(train.head())
print(train.tail())

# looking for missing values

print(train.isnull().sum())
print(train.isnull().sum().sort_values(ascending=False))

# different way to check this

train.info()

# plotting missing variables

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='plasma')
plt.show()

# Let's look at a coorelation matrix for all of our numeric variables
corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(10, 9)
sns.heatmap(corr, mask=mask, vmax=.6, square=True, annot=True);

# there are missing values for cabin, age and embarked


# there is 1 less variable in the test data. let's compare the differences

print('Train columns:', train.columns.tolist())
print('Test columns:', test.columns.tolist())

# survived is not part of the test data


# analyzing the variable survived


# Exploring data

# Countplot for 'Survived' variable
sns.countplot(train['Survived'])
plt.show()
# Let's calculate the mean of our target
round(np.mean(train['Survived']), 2)

# chosing a variable

sns.countplot(x='Survived', hue='Sex', data=train)
plt.show()

# it looks like men are more likely to die!

# reviewing features

sns.countplot(train['Pclass'])
plt.show()

# clean category, 3 possibilities

sns.countplot(train['Name'])
# plt.show() ran this, so many options, easier to look at list
print(train.Name.value_counts())

# figure this out, what to do with this feature???

# age

train['Age'].hist(bins=50, color='blue')

# Function to split and re-group name column
def name_func(row):
    if row['Name'].split(' ')[1] == 'Mr.':
        return 'Mr.'
    elif row['Name'].split(' ')[1] == 'Mrs.':
        return 'Mrs.'
    elif row['Name'].split(' ')[1] ==  'Miss.':
        return 'Miss'
    else:
        return 'Other'


train['Name Split'] = train.apply(name_func, axis=1)

# plot name column
sns.countplot(train['Name Split'])


sns.countplot(x='Survived', hue='Name Split', data=train)
plt.show()


'''
Next steps:

More EDA
Feature Engineering 
Prep for ML

Before we meet again you can study and try to split the data. 
'''



