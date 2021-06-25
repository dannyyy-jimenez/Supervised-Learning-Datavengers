import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CLFScores
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score


def Model(X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = LogisticRegression(**kwargs)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test


def Optimize(X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)
    optimizers = {'penalty': ['l2', 'elasticnet', 'none'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
    grid = GridSearchCV(LogisticRegression(max_iter=10000, **kwargs), optimizers, verbose=3)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


if __name__ == "__main__":
    pass
