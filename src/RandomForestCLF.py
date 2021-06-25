from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
import pandas as pd
import numpy as np
import CLFScores
from sklearn.ensemble import RandomForestClassifier

def Model(X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = RandomForestClassifier(**kwargs)
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
    
    ###Random Hyperparameter Grid
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 100)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2, 15, num = 3)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    
  

    # Create the random grid
    random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier(bootstrap = True, oob_score=True)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, cv = 5, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    
    return rf_random.best_estimator_, rf_random.best_params_

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
