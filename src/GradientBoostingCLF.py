from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import CLFScores
from sklearn.ensemble import GradientBoostingClassifier


def Model(X, y, fit=True, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = GradientBoostingClassifier(**kwargs)

    if fit:
        clf.fit(X_train, y_train)
    else:
        return clf
    return clf, X_test, y_test


def Optimize(X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

    optimizers = {'learning_rate': [0.1, 0.4, 1], 'subsample': [0.2, 1], 'max_depth': [1, 3], 'min_samples_split': [2, 10], 'max_features': ['sqrt', 'log2', None]}
    grid = GridSearchCV(GradientBoostingClassifier(**kwargs), optimizers, verbose=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


if __name__ == "__main__":
    X = pd.read_csv('../data/churn_train.csv')
    X['city'] = X['city'].apply(lambda x: np.random.choice([0, 1, 2]))
    X['phone'] = X['phone'].apply(lambda x: np.random.choice([0, 1]))
    X['luxury_car_user'] = X['luxury_car_user'].apply(lambda x: np.random.choice([0, 1]))
    X.drop(columns=['last_trip_date', 'signup_date'], inplace=True)
    X = X[X.notna().all(axis=1)]
    y = np.random.choice([False, True], size=X.shape[0])

    model, X_test, y_test = Model(X, y, random_state=42)

    unfitted_model = Model(X, y, False, random_state=42)

    accuracy, precision, recall = CLFScores.Scores(unfitted_model, X, y)

    best_model, best_params = Optimize(X, y, random_state=42)
    unfitted_best_model = Model(X, y, False, **best_params)

    CLFScores.Scores(unfitted_model, X_test, y_test)
    CLFScores.PlotConfusionMatrix(model, X_test, y_test)
    CLFScores.PlotRocCurve(model, X_test, y_test, name="Gradient Boosting Tree")

    CLFScores.PermutationImportance(model, X.columns, X_test, y_test)
    pass
