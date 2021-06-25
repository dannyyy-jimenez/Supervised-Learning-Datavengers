import pandas as pd
from DataCleaning import Data
import DecisionTreeCLF
import CLFScores

X, y = Data()

# model, X_test, y_test = DecisionTreeCLF.Model(X, y, random_state=42)
# accuracy, precision, recall = DecisionTreeCLF.Scores(model, X_test, y_test)
# best_model, best_params = DecisionTreeCLF.Optimize(X, y, random_state=42)
# DecisionTreeCLF.Scores(best_model, X_test, y_test)
# DecisionTreeCLF.ConfusionMatrix(best_model, X_test, y_test)
# CLFScores.PlotConfusionMatrix(best_model, X_test, y_test)
