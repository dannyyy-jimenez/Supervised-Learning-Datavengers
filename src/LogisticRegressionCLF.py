import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_score,  GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
#import CLFScores
plt.style.use("ggplot")
kfold = KFold(n_splits=10)

def Model(X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = LogisticRegression(**kwargs)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test

def Scores(estimator, X_test, y_test, **kwargs):
    accuracy = np.mean(cross_val_score(estimator, X, y, scoring='accuracy', **kwargs))
    recall = np.mean(cross_val_score(estimator, X, y, scoring='recall', **kwargs))
    precision = np.mean(cross_val_score(estimator, X, y, scoring='precision', **kwargs))
    print(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}")
    return accuracy, precision, recall

def Optimize(X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)
    optimizers = {'penalty': ['l2', 'elasticnet', 'none'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
    grid = GridSearchCV(LogisticRegression(max_iter=10000, **kwargs), optimizers, verbose=3)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


if __name__ == "__main__":
    X = pd.read_csv('/home/vurimindi-ubuntu/Desktop/gsd/casestudy/Supervised-Learning-Datavengers/data/churn.csv')
    X['city'] = X['city'].apply(lambda x: np.random.choice([0, 1, 2]))
    X['phone'] = X['phone'].apply(lambda x: np.random.choice([0, 1]))
    X['luxury_car_user'] = X['luxury_car_user'].apply(lambda x: np.random.choice([0, 1]))
    X.drop(columns=['last_trip_date', 'signup_date'], inplace=True)
    X = X[X.notna().all(axis=1)]
    y = np.random.choice([False, True], size=X.shape[0])

    model, X_test, y_test = Model(X, y, random_state=42)

    accuracy, precision, recall = Scores(model, X_test, y_test)

    best_model, best_params = Optimize(X, y, random_state=42)
    Scores(clf, X_test, y_test, cv=5)
    CLFScores.PlotConfusionMatrix(best_model, X_test, y_test)
    CLFScores.PlotRocCurve(best_model, X_test, y_test, name="Decision Tree")
    CLFScores.PermutationImportance(best_model, X.columns, X_test, y_test)
