"""Random search for the ensemble members. Settings are picked on one set of folds and checked on another."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from make_submission import GROUPS
from src.data import load_all
from src.features import build_features, build_species_table, build_vocab
from src.models import SEED, make_ensemble
from src.validation import compare, cv_scores, make_folds

HGB_SPACE = {
    "max_iter": [200, 300, 500, 800],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "max_leaf_nodes": [4, 6, 8, 12],
    "min_samples_leaf": [20, 40, 80, 150],
    "l2_regularization": [1.0, 5.0, 20.0],
    "max_features": [0.2, 0.4, 0.7, 1.0],
}
TREES_SPACE = {
    "min_samples_leaf": [1, 2, 3, 5, 8],
    "max_features": [0.1, 0.2, 0.3, 0.5],
}
LOGREG_C = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def hgb(**params):
    return HistGradientBoostingClassifier(random_state=SEED, **params)


def trees(**params):
    return ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1, **params)


def logreg(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=SEED))


def search(name, candidates, make_model, X, y, folds):
    best_params, best_score = None, -1
    for params in candidates:
        score = cv_scores(lambda: make_model(**params), X, y, folds).mean()
        print(f"{name} {score * 100:.2f} {params}", flush=True)
        if score > best_score:
            best_params, best_score = params, score
    print(f"best {name}: {best_score * 100:.2f} {best_params}\n", flush=True)
    return best_params


def main():
    train, test, chart, _ = load_all()
    y = np.array([int(b["player_won"]) for b in train])
    species = build_species_table(train + test)
    X = build_features(train, species, chart, build_vocab(train), GROUPS)

    pick_folds = make_folds(y, n_repeats=1, seed=7)
    best_c = search("logreg", [{"C": c} for c in LOGREG_C], logreg, X, y, pick_folds)
    best_trees = search("trees", list(ParameterSampler(TREES_SPACE, 8, random_state=SEED)), trees, X, y, pick_folds)
    best_hgb = search("hgb", list(ParameterSampler(HGB_SPACE, 20, random_state=SEED)), hgb, X, y, pick_folds)

    def tuned_ensemble():
        return VotingClassifier(
            [("logreg", logreg(**best_c)), ("hgb", hgb(**best_hgb)), ("trees", trees(**best_trees))],
            voting="soft",
        )

    check_folds = make_folds(y, n_repeats=2, seed=2026)
    current = cv_scores(make_ensemble, X, y, check_folds)
    print(f"current ensemble {current.mean() * 100:.2f}")
    for name, make_model in (
        ("tuned logreg", lambda: logreg(**best_c)),
        ("tuned hgb", lambda: hgb(**best_hgb)),
        ("tuned trees", lambda: trees(**best_trees)),
        ("tuned ensemble", tuned_ensemble),
    ):
        scores = cv_scores(make_model, X, y, check_folds)
        diff, err = compare(scores, current)
        print(f"{name} {scores.mean() * 100:.2f}   {diff:+.2f} +/- {err:.2f}", flush=True)


if __name__ == "__main__":
    main()
