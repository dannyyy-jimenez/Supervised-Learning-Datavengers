from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
import pandas as pd
import numpy as np
import CLFScores


def Model(X, y, fit=True, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = DecisionTreeClassifier(**kwargs)

    if fit:
        clf.fit(X_train, y_train)
    else:
        return clf

    return clf, X_test, y_test

def Optimize(X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

    optimizers = {'max_depth': [3, 5, None], 'min_samples_split': [2, 10], 'max_features': ['sqrt', 'log2', 2, 5, None]}
    grid = GridSearchCV(DecisionTreeClassifier(**kwargs), optimizers, verbose=3)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


if __name__ == "__main__":
    X = pd.read_csv('./data/churn_train.csv')
    pass
