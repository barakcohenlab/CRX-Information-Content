"""
Author: Ryan Friedman (@rfriedman22)
Email: ryan.friedman@wustl.edu
"""

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_curve, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def calc_tpr_precision_fbeta(true_labels, positive_probs, xaxis, positive_cutoff=None, f_beta=1.0, f_avg="binary"):
    """Calculate the TPR and Precision for predictions and interpolate them to the x-axis (i.e. FPR and Recall)

    Parameters
    ----------
    true_labels : list-like
        True class labels of the data, 0 for negatives and 1 for positives.
    positive_probs : list-like
        Predicted probability that each example belongs to the positives.
    xaxis : list-like
        Represents FPR and Recall. Used to interpolate the calculated TPR and Precision to make plotting easier.
    positive_cutoff : float or None
        Cutoff for separating positives from negatives to binarize the probabilities (used for F-score). Anything
        greater than the cutoff is a positive, anything less than or equal to the cutoff is a negative. If None, simply take the argmax for each row.
    f_beta : float
        Beta value for the F-score, default is F1 score.
    f_avg : str
        Method for averaging F-score for classes, default is binary.

    Returns
    -------
    tpr : np.array
        The TPR interpolated to xaxis.
    precision: np.array
        The precision interpolated to xaxis.
    f_score : float
        The F-beta score

    """
    fpr, tpr, _ = roc_curve(true_labels, positive_probs)
    tpr = np.interp(xaxis, fpr, tpr)
    # Make sure TPR still starts at zero
    tpr[0] = 0.0
    precision, recall, _ = precision_recall_curve(true_labels, positive_probs)
    # If no cutoff is specified, the label corresponds to the one with the highest probability/score.
    if positive_cutoff is None:
        pred_labels = positive_probs.argmax(axis=1)
    # Otherwise, the labels are true if the probability/score is above the cutoff and false otherwise.
    else:
        pred_labels = (positive_probs > positive_cutoff).astype(int)
    f_score = fbeta_score(true_labels, pred_labels, beta=f_beta, average=f_avg)
    precision = np.interp(xaxis, recall[::-1], precision[::-1])
    return tpr, precision, f_score


def train_estimate_variance(classifier, cv, features, labels, xaxis, positive_cutoff=None, f_beta=1.0, f_avg="binary"):
    """Compute the ROC and PR curves for each fold of the stratified cross-validation object. These can then be used
    to estimate the SD of model performance. Then, fit the model to all available data. Assumes hyperparameters are
    already learned.

    Parameters
    ----------
    classifier : An sklearn classifier with hyperparameters set.
    cv : sklearn.model_selection.StratifiedKFold
        The object used to perform cross-validation.
    features : matrix-like
        Training data features, shape = [n_examples, n_features]
    labels : list-like
        Class labels of the data. Positives are 1, negatives are 0.
    xaxis : list-like
        Represents FPR and Recall. Used to interpolate the calculated TPR and Precision to make plotting easier.
    positive_cutoff : float or None
        Cutoff for separating positives from negatives to binarize the probabilities (used for F-score). Anything
        greater than the cutoff is a positive, anything less than or equal to the cutoff is a negative. If None, simply
        take the argmax for each row.
    f_beta : float
        Beta value for the F-score, default is F1 score.
    f_avg : str
        Method for averaging F-score for classes, default is binary.

    Returns
    -------
    classifier : The classifier fit to all training data.
    tpr_cv : matrix-like
        The TPR calculated for each fold of the CV, interpolated along xaxis.
    precision_cv : matrix-like
        The precision calculated for each fold of the CV, interpolated along xaxis.
    f_cv : array-like
        The F-beta score calculated for each fold of the CV.
    """
    tpr_cv = []
    precision_cv = []
    f_cv = []
    features = pd.DataFrame(features)

    for train_idx, val_idx in cv.split(features, labels):
        # If the kernel is precomputed, then features is a Gram matrix and we need to extract a sub-matrix,
        # not just the rows.
        if hasattr(classifier, "kernel") and classifier.kernel == "precomputed":
            train_features = features.iloc[train_idx, train_idx]
            val_features = features.iloc[val_idx, train_idx]
        else:
            train_features = features.iloc[train_idx]
            val_features = features.iloc[val_idx]
        classifier.fit(train_features, labels[train_idx])
        # If the model has a decision function, use that to predict. Otherwise, predict probabilities.
        if hasattr(classifier, "decision_function"):
            preds = classifier.decision_function(val_features)
        else:
            preds = classifier.predict_proba(val_features)[:, 1]
        truth = labels[val_idx]

        tpr, precision, f_score = calc_tpr_precision_fbeta(truth, preds, xaxis, positive_cutoff=positive_cutoff,
                                                           f_beta=f_beta, f_avg=f_avg)
        tpr_cv.append(tpr)
        precision_cv.append(precision)
        f_cv.append(f_score)

    # Fit the model to all training data
    classifier.fit(features, labels)

    return classifier, tpr_cv, precision_cv, f_cv


def grid_search_hyperparams(classifier, n_folds, param_grid, scoring, features, labels, xaxis, positive_cutoff=None,
                            f_beta=1.0, f_avg="binary"):
    """Perform grid search to select optimal hyperparameters for the classifier. Then, use the same cross-validation
    split to estimate training variance. Finally, use all available data to fit the final model.

    Parameters
    ----------
    classifier : An sklearn classifier.
    n_folds : int
        The number of folds to use in cross-validation for grid search.
    param_grid : dict or list of dicts
        The grid of hyperparameters to try.
    scoring : str or callable
        The scoring function to use to assess hyperparameter performance.
    features : matrix-like
        Training data features, shape = [n_examples, n_features]
    labels : list-like
        Class labels of the data. Positives are 1, negatives are 0.
    xaxis : list-like
        Represents FPR and Recall. Used to interpolate the calculated TPR and Precision to make plotting easier.
    positive_cutoff : float or None
        Cutoff for separating positives from negatives to binarize the probabilities (used for F-score). Anything
        greater than the cutoff is a positive, anything less than or equal to the cutoff is a negative. If None, simply
        take the argmax for each row.
    f_beta : float
        Beta value for the F-score, default is F1 score.
    f_avg : str
        Method for averaging F-score for classes, default is binary.

    Returns
    -------
    classifier : The classifier fit to all training data.
    tpr_cv : matrix-like
        The TPR calculated for each fold of the CV, interpolated along xaxis.
    precision_cv : matrix-like
        The precision calculated for each fold of the CV, interpolated along xaxis.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True)
    grid_searcher = GridSearchCV(classifier, param_grid, scoring, cv=cv, return_train_score=True)
    grid_searcher.fit(features, labels)

    # Get the best model
    classifier = grid_searcher.best_estimator_
    classifier, tpr_cv, precision_cv, f_cv = train_estimate_variance(classifier, cv, features, labels, xaxis,
                                                                     positive_cutoff=positive_cutoff, f_beta=f_beta,
                                                                     f_avg=f_avg)
    return classifier, tpr_cv, precision_cv


def fdr(pvalues, name_prefix=None):
    """Correct for multiple hypotheses using Benjamini-Hochberg FDR and return q-values for each observation. Ties
    are assigned the largest possible rank.

    Parameters
    ----------
    pvalues : pd.Series or pd.DataFrame
        Each row is the p-value for an observation. If pvalues is a DataFrame, each column is a different condition.
        FDR is performed separately on each column.
    name_prefix : str, list, or None
        Prefix(es) to use for name(s) of the q-values. `_qvalue` is appended to the prefixes If a str, then pvalues
        must be a Series; if list-like, then pvalue must be a DataFrame. If None or a datatype mismatch, simply take
        the old names and append `_qvalue` to the names.

    Returns
    -------
    qvalues : Same as pvalues
        The FDR-corrected q-values.

    """
    n_measured = pvalues.notna().sum()
    ranks = pvalues.rank(method="max")
    qvalues = pvalues * n_measured / ranks
    suffix = "_qvalue"

    # Define the name of qvalues
    if type(pvalues) is pd.Series:
        if type(name_prefix) is str:
            name_prefix += suffix
        else:
            name_prefix = pvalues.name + suffix
        qvalues.name = name_prefix

    elif type(pvalues) is pd.DataFrame:
        if type(name_prefix) is not list and type(name_prefix) is not np.array:
            name_prefix = pvalues.columns

        name_prefix = [i + suffix for i in name_prefix]
        qvalues.columns = name_prefix
    else:
        raise Exception(f"Error, pvalues is not a valid data type (this should never happen), it is a {type(pvalues)}")
    return qvalues
