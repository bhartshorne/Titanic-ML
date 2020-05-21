import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
try:
    os.chdir(r'C:\Users\Billy Hansen\Desktop\Kaggle Practice\Titanic')
except:
    os.chdir(r'C:\Users\harts\Documents\Titanic ML')

# importing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
