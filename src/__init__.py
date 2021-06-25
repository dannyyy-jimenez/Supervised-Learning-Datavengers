import pandas as pd
from DataCleaning import Data
import DecisionTreeCLF
import LogisticRegressionCLF
import GradientBoostingCLF
import RandomForestCLF
import CLFScores

X_test, y_test = Data('churn_test')
X_train, y_train = Data('churn_train')

X_train


def DecisionTree(X, y, **kwargs):
    model, X_test, y_test = DecisionTreeCLF.Model(X, y, **kwargs)
    unfitted_model = DecisionTreeCLF.Model(X, y, False, **kwargs)
    accuracy, precision, recall = CLFScores.Scores(unfitted_model, X, y)
    best_model, best_params = DecisionTreeCLF.Optimize(X, y, **kwargs)
    unfitted_best_model = DecisionTreeCLF.Model(X, y, False, **best_params)

    CLFScores.Scores(unfitted_model, X_test, y_test)
    CLFScores.PlotConfusionMatrix(model, X_test, y_test, filename="Decision-Tree")
    CLFScores.PlotRocCurve(model, X_test, y_test, filename="Decision-Tree", name="Decision Tree Tree")

    CLFScores.Scores(unfitted_best_model, X_test, y_test)
    CLFScores.PlotConfusionMatrix(best_model, X_test, y_test, filename="Decision-Tree-Best")
    CLFScores.PlotRocCurve(best_model, X_test, y_test, filename="Decision-Tree-Best", name="Decision Tree Tree Best")

    CLFScores.PermutationImportance(best_model, X_test.columns, X_test, y_test, filename="Decision-Tree-Best")

    return best_model


def LogisticRegression(X, y, **kwargs):
    model, X_test, y_test = LogisticRegressionCLF.Model(X, y, **kwargs)
    unfitted_model = LogisticRegressionCLF.Model(X, y, False, **kwargs)
    accuracy, precision, recall = CLFScores.Scores(unfitted_model, X, y)
    best_model, best_params = LogisticRegressionCLF.Optimize(X, y, **kwargs)
    unfitted_best_model = LogisticRegressionCLF.Model(X, y, False, **best_params)

    CLFScores.Scores(unfitted_model, X_test, y_test)
    CLFScores.PlotConfusionMatrix(model, X_test, y_test, filename="Logistic-Regression")
    CLFScores.PlotRocCurve(model, X_test, y_test, filename="Logistic-Regression", name="Logistic Regression")

    CLFScores.Scores(unfitted_best_model, X_test, y_test)
    CLFScores.PlotConfusionMatrix(best_model, X_test, y_test, filename="Decision-Tree-Best")
    CLFScores.PlotRocCurve(best_model, X_test, y_test, filename="Logistic-Regression-Best", name="Logistic-Regression Best")

    CLFScores.PermutationImportance(best_model, X_test.columns, X_test, y_test, filename="Logistic-Regression-Best")

    return best_model


def GradientBoosting(X, y, **kwargs):
    model, X_test, y_test = GradientBoostingCLF.Model(X, y, **kwargs)
    unfitted_model = GradientBoostingCLF.Model(X, y, False, **kwargs)
    accuracy, precision, recall = CLFScores.Scores(unfitted_model, X, y)
    best_model, best_params = GradientBoostingCLF.Optimize(X, y, **kwargs)
    unfitted_best_model = GradientBoostingCLF.Model(X, y, False, **best_params)

    CLFScores.Scores(unfitted_model, X_test, y_test)
    CLFScores.PlotConfusionMatrix(model, X_test, y_test, filename="Gradient-Boosting")
    CLFScores.PlotRocCurve(model, X_test, y_test, filename="Gradient Boosting", name="Gradient Boosting")

    CLFScores.Scores(unfitted_best_model, X_test, y_test)
    CLFScores.PlotConfusionMatrix(best_model, X_test, y_test, filename="Gradient-Boosting-Best")
    CLFScores.PlotRocCurve(best_model, X_test, y_test, filename="Gradient-Boosting-Best", name="Gradient Boosting Best")

    CLFScores.PermutationImportance(best_model, X_test.columns, X_test, y_test, filename="Gradient-Boosting-Best")

    return best_model


def RandomForest(X, y, **kwargs):
    model, X_test, y_test = RandomForestCLF.Model(X, y, **kwargs)
    unfitted_model = RandomForestCLF.Model(X, y, False, **kwargs)
    accuracy, precision, recall = CLFScores.Scores(unfitted_model, X, y)
    best_model, best_params = RandomForestCLF.Optimize(X, y, **kwargs)
    unfitted_best_model = RandomForestCLF.Model(X, y, False, **best_params)

    CLFScores.Scores(unfitted_model, X_test, y_test)
    CLFScores.PlotConfusionMatrix(model, X_test, y_test, filename="Random-Forest")
    CLFScores.PlotRocCurve(model, X_test, y_test, filename="Random-Forest", name="Random Forest")

    CLFScores.Scores(unfitted_best_model, X_test, y_test)
    CLFScores.PlotConfusionMatrix(best_model, X_test, y_test, filename="Random-Forest-Best")
    CLFScores.PlotRocCurve(best_model, X_test, y_test, filename="Random-Forest-Best", name="Random Forest Best")

    CLFScores.PermutationImportance(best_model, X_test.columns, X_test, y_test, filename="Random-Forest-Best")

    return best_model

# <editor-fold> Decision Tree

decision_tree_best_model = DecisionTree(X_train, y_train, random_state=45)
CLFScores.FinalScores(decision_tree_best_model, X_test, y_test)

# </editor-fold>


# <editor-fold> Random Forest

random_forest_best_model = RandomForest(X_train, y_train, random_state=45)
CLFScores.FinalScores(random_forest_best_model, X_test, y_test)

# </editor-fold>

# <editor-fold> Gradient Boosting

gradient_boosting_best_model = GradientBoosting(X_train, y_train, random_state=45)
CLFScores.FinalScores(gradient_boosting_best_model, X_test, y_test)

# </editor-fold>

# <editor-fold> LogisticRegression

logistic_regression_best_model = LogisticRegression(X_train, y_train, random_state=45)
CLFScores.FinalScores(logistic_regression_best_model, X_test, y_test)

# </editor-fold>
