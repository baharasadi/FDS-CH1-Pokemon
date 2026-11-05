from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def make_logreg(C=0.05):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=SEED))


def make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=8,
        min_samples_leaf=40,
        l2_regularization=5.0,
        random_state=SEED,
    )


def make_extra_trees():
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.3,
        random_state=SEED,
        n_jobs=-1,
    )


def make_ensemble():
    return VotingClassifier(
        [("logreg", make_logreg()), ("hgb", make_hgb()), ("trees", make_extra_trees())],
        voting="soft",
    )


MODELS = {
    "logreg": make_logreg,
    "hgb": make_hgb,
    "extra_trees": make_extra_trees,
    "ensemble": make_ensemble,
}
