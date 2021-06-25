import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


def PlotRocCurve(model, X_test, y_test, filename=None, **kwargs):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    plot_roc_curve(model, X_test, y_test, ax=ax, **kwargs)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(f'../images/{filename}-roc')

    pass


def PlotConfusionMatrix(model, X_test, y_test, filename=None):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    plot_confusion_matrix(model, X_test, y_test, ax=ax)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(f'../images/{filename}-confusion_matrix')

    pass


def PermutationImportance(model, feature_names, X_test, y_test, filename=None, **kwargs):
    r = permutation_importance(model, X_test, y_test, n_repeats=30, **kwargs)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{feature_names[i]:<8}"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}")

    fig, ax = plt.subplots(1, figsize=(8, 8))
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    feat_importances.nlargest(10).plot(kind='barh', ax=ax)

    if filename is not None:
        fig.savefig(f'../images/{filename}-permutation_importance')

    pass


def Scores(estimator, X, y, **kwargs):
    accuracy = np.mean(cross_val_score(estimator, X, y, scoring='accuracy', **kwargs))
    recall = np.mean(cross_val_score(estimator, X, y, scoring='recall', **kwargs))
    precision = np.mean(cross_val_score(estimator, X, y, scoring='precision', **kwargs))

    print(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}")

    return accuracy, recall, precision


if __name__ == "__main__":
    pass
