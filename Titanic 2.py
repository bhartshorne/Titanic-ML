import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

try:
    os.chdir(r'C:\Users\Billy Hansen\Desktop\Kaggle Practice\Titanic')
except:
    os.chdir(r'C:\Users\Owner\Desktop\Machine Learning Practice\Titanic ML')

# importing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.shape)
print(test.shape)

# Check Target Variable distribution
train.Survived.value_counts(normalize=True)

# Check Relationships to target variable

# Percentage of total by gender
train.Sex.value_counts(normalize=True)

# Percentage of survival by gender
train['Survived'].groupby(train['Sex']).mean()  # Looks like Sex is a very important variable

# Percentage of total by PClass
train.Pclass.value_counts(normalize=True)

# Percentage survived by PClass
train['Survived'].groupby(train['Pclass']).mean()  # Another Useful Variable

# Percentage of total by embarked
train.Embarked.value_counts(normalize=True)

# Percentage survived by embarked
train['Survived'].groupby(train['Embarked']).mean()

'''Does embarked really matter or is this difference random?
Intuitivelly it seemms like it wouldn't matter much which port you took off from,
especially because you are assigned to a cabin anyway.

Let's look at some correlation stats for our ordinal variables. 
'''

train[['Survived', 'Parch', 'SibSp']].corr(method='spearman')  # Not a strong correlation.

'''Looks like ther are slight, positive correlations. Now for feature engineering.
So not to repeat steps I'm going to join both datasets and then break them apart when I'm done.'''

df = train.append(test, sort=True)

# Drop Passenger ID
df = df.drop(columns=['PassengerId'])


# Function to split and re-group name column
def name_func(row):
    if row['Name'].split(' ')[1] == 'Mr.':
        return 'Mr.'
    elif row['Name'].split(' ')[1] == 'Mrs.':
        return 'Mrs.'
    elif row['Name'].split(' ')[1] == 'Miss.':
        return 'Miss.'
    elif row['Name'].split(' ')[1] == 'Dr.':
        return 'Dr.'
    else:
        return 'Other'


df['Name Split'] = df.apply(name_func, axis=1)

# Look at Name Split stats
df['Name Split'].value_counts(normalize=True)

df['Survived'].groupby(df['Name Split']).mean()  # looks interested but maybe too correlated with Sex variable?

# Drop Name Column
df = df.drop(columns=['Name'])

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
age = df.drop(columns=['Cabin', 'Survived', 'Ticket'])

# Pclass should be a categorical variable
age['Pclass'] = age['Pclass'].astype('object')

# Create Dummies
age = pd.get_dummies(age, columns=['Name Split', 'Sex', 'Pclass', 'Embarked'], drop_first=True)

# Create train and predict datasets
age_train = age[age['Age'].notnull()]
age_test = age[age['Age'].isnull()]
age_test = age_test.drop(columns=['Age'])
len(age_train)
len(age_test)

# Grab target vector
age_X = age_train.drop(columns=['Age'])
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

# Should we use the cabin column?
df.Cabin.isnull().sum() / len(df)

'''Because 77% of the values are null, I'm going to drop this column.
I'd be interested to see how others filled this column effectively, and if it had value after doing so.'''

df = df.drop(columns=['Cabin'])

df.isnull().sum()  # Nulls are all clean.

'''# I'm also going to drop the ticket number column for now. I'm don't have
a clear idea how this column could be engineered into something useful.
'''

df = df.drop(columns=['Ticket'])

# Let's create dummy variables.
df_ready = pd.get_dummies(df, columns=['Embarked', 'Pclass', 'Sex', 'Name Split'], drop_first=True)
# Split back into training and testing datasets.
train = df_ready[~pd.isnull(df_ready['Survived'])]
test = df_ready[pd.isnull(df_ready['Survived'])].drop(columns=['Survived'])

# Grab Target
X = train.drop(columns=['Survived'])
y = train['Survived']

'''For some models we'll want to use feature scaling, and for our
tree based models we won't want to. I'll created scaled, and unscaled versions of the data.'''

sc = StandardScaler()
Xs = sc.fit_transform(X)
testS = sc.transform(test)

# Start with logistic regression with scaled data and 10 fold cross validation.
lr = LogisticRegression(solver='lbfgs')
scores = cross_val_score(lr, Xs, y, cv=10, scoring='accuracy')
print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Support Vector Machine with 10 fold cross validation
svm = SVC(gamma='scale')
scores = cross_val_score(svm, Xs, y, cv=10, scoring='accuracy')
print("Simple Support Vector Machine Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Grid Search for Logistic Regression - We'll split the data for this step
X_train_S, X_test_S, y_train_S, y_test_S = train_test_split(Xs, y, test_size=.15, random_state=12)


lr = LogisticRegression(solver='liblinear')
params = {'C': [.5, 1, 1.5], 'max_iter': [100, 50, 150]}

grid_lr = GridSearchCV(lr,
                       params,
                       scoring='accuracy',
                       cv=10)

grid_lr.fit(X_train_S, y_train_S)

preds = grid_lr.predict(X_test_S)

print(grid_lr.best_params_)
print("Accuracy For Logistic Regression: ", round(accuracy_score(y_test_S, preds), 2))

# Grid Search for SVM
svm = SVC(gamma='scale')
params = {'C': [.5, 1, 1.5, 2], 'kernel': ['rbf', 'linear']}

grid_svm = GridSearchCV(svm,
                       params,
                       scoring='accuracy',
                       cv=10)

grid_svm.fit(X_train_S, y_train_S)

preds = grid_svm.predict(X_test_S)

print(grid_svm.best_params_)
print("Accuracy For SVM: ", round(accuracy_score(y_test_S, preds), 2))

# Let's Try XGBoost
from xgboost import XGBClassifier

# Split Unscaled Data
# Grid Search for Logistic Regression - We'll split the data for this step
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=12)

# For XGBoost and other tree based models we'll use unscaled data - We'll grid search right away
XG = XGBClassifier(objective='binary:logistic')

params = {'eta':[.1, .3, .5], 'max_depth': [4, 6, 8]}

grid_XGB = GridSearchCV(XG,
                       params,
                       scoring='accuracy',
                       cv=10)

grid_XGB.fit(X_train, y_train)

preds = grid_XGB.predict(X_test)


print(grid_XGB.best_params_)
print("Accuracy For XGBoost: ", round(accuracy_score(y_test, preds), 2))

# Let's Try XGBoost
# Split Unscaled Data
# Grid Search for Logistic Regression - We'll split the data for this step
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=12)

# For XGBoost and other tree based models we'll use unscaled data - We'll grid search right away
XG = XGBClassifier(objective='binary:logistic')

params = {'eta':[.1, .3, .5], 'max_depth': [4, 6, 8]}

grid_XGB = GridSearchCV(XG,
                       params,
                       scoring='accuracy',
                       cv=10)

grid_XGB.fit(X_train, y_train)

preds = grid_XGB.predict(X_test)


print(grid_XGB.best_params_)
print("Accuracy For XGBoost: ", round(accuracy_score(y_test, preds), 2))


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300)

params = {'criterion': ['gini', 'entropy'], 'min_samples_split': [.5, 2, 5,], 'min_samples_leaf': [1, 2, 5]}

grid_RF = GridSearchCV(rf,
                       params,
                       scoring='accuracy',
                       cv=10)

grid_RF.fit(X_train, y_train)

preds = grid_RF.predict(X_test)


print(grid_RF.best_params_)
print("Accuracy For RF: ", round(accuracy_score(y_test, preds), 2))


# Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

preds = gnb.fit(X_train, y_train).predict(X_test)
print("Accuracy For Naive Bayes is: ", round(accuracy_score(y_test, preds), 2))

# Make first submission - Best model right now is the grid search logistic regression

final_preds = grid_lr.predict(test)

test = pd.read_csv('test.csv')
sub = test[['PassengerId']]
sub['Survived'] = final_preds

sub['Survived'] = sub['Survived'].astype('int32')

sub.to_csv('Sub2.csv', index=False)
