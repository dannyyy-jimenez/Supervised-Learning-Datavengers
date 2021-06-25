from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
plt.style.use('ggplot')


churn_original = pd.read_csv('data/churn.csv')
churn_train_original = pd.read_csv('data/churn_train.csv')
churn_test_original = pd.read_csv('data/churn_test.csv')

print(churn_original.head())
print(churn_test_original.head())
print(churn_train_original.head())
print(churn_original.info())

