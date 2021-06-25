from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
import pandas as pd
import numpy as np
import CLFScores


def Model(X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(X_train, y_train)

    return clf, X_test, y_test


def Scores(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = recall_score(y_test, predictions)
    recall = precision_score(y_test, predictions)

    print(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}")

    return accuracy, precision, recall


def Optimize(X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

    optimizers = {'max_depth': [3, 5, None], 'min_samples_split': [2, 10], 'max_features': ['sqrt', 'log2', 2, 5, None]}
    grid = GridSearchCV(DecisionTreeClassifier(**kwargs), optimizers, verbose=3)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


if __name__ == "__main__":
    X = pd.read_csv('./data/churn_train.csv')
    X['city'] = X['city'].apply(lambda x: np.random.choice([0, 1, 2]))
    X['phone'] = X['phone'].apply(lambda x: np.random.choice([0, 1]))
    X['luxury_car_user'] = X['luxury_car_user'].apply(lambda x: np.random.choice([0, 1]))
    X.drop(columns=['last_trip_date', 'signup_date'], inplace=True)
    X = X[X.notna().all(axis=1)]
    y = np.random.choice([False, True], size=X.shape[0])

    model, X_test, y_test = Model(X, y, random_state=42)

    accuracy, precision, recall = Scores(model, X_test, y_test)

    best_model, best_params = Optimize(X, y, random_state=42)

    Scores(best_model, X_test, y_test)
    CLFScores.PlotConfusionMatrix(best_model, X_test, y_test)
    CLFScores.PlotRocCurve(best_model, X_test, y_test, name="Decision Tree")

    CLFScores.PermutationImportance(best_model, X.columns, X_test, y_test)
    pass
