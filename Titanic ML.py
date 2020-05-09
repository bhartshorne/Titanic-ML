import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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
print(train.isnull().sum().sort_values(ascending = False))

# different way to check this

train.info()

#plotting missing variables

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'plasma')
plt.show()

# there are missing values for cabin, age and embarked


# there is 1 less variable in the test data. let's compare the differences

print('Train columns:', train.columns.tolist())
print('Test columns:', test.columns.tolist())

# survived is not part of the test data


# analyzing the variable survived



#Exploring data

# Countplot for 'Survived' variable
sns.countplot(train['Survived'])
plt.show()
# Let's calculate the mean of our target
round(np.mean(train['Survived']), 2)

# chosing a variable

sns.countplot(x = 'Survived', hue = 'Sex', data = train)
plt.show()

# it looks like men are more likely to die!

# reviewing features

sns.countplot(train['Pclass'])
plt.show()

# clean category, 3 possibilities

#sns.countplot(train['Name'])
#plt.show() ran this, so many options, easier to look at list
print(train.Name.value_counts())

#figure this out, what to do with this feature???

# age

train['Age'].hist(bins = 50, color = 'blue')







