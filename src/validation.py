import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold


def make_folds(y, n_repeats=2, seed=7):
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def cv_scores(make_model, X, y, folds):
    """Accuracy on every fold. Use the same folds for all models so they can be compared."""
    scores = []
    for train_idx, valid_idx in folds:
        model = make_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        scores.append(accuracy_score(y[valid_idx], model.predict(X.iloc[valid_idx])))
    return np.array(scores)


def compare(scores, baseline):
    """Paired difference to a baseline, in percentage points."""
    diff = (scores - baseline) * 100
    return diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
