"""Compare feature groups and models with repeated 5-fold CV on the training set."""
import numpy as np

from make_submission import GROUPS
from src.data import load_all
from src.features import FEATURE_GROUPS, build_features, build_species_table, build_vocab, select_columns
from src.models import make_hgb, make_logreg, MODELS
from src.validation import compare, cv_scores, make_folds

ROUND_TWO = ["vanish", "status", "hazards", "momentum"]
ROUND_ONE = [g for g in FEATURE_GROUPS if g not in ROUND_TWO]


def report(name, scores, baseline=None):
    line = f"{name:<40} {scores.mean() * 100:6.2f}"
    if baseline is not None:
        diff, err = compare(scores, baseline)
        line += f"   {diff:+5.2f} +/- {err:.2f}"
    print(line, flush=True)


def forward_selection(X, y, folds, start, candidates, baseline):
    """Add the group that helps hgb most until none of them improves the score."""
    chosen = []
    best = baseline
    remaining = list(candidates)
    while remaining:
        trials = {g: cv_scores(make_hgb, X[select_columns(X, start + chosen + [g])], y, folds) for g in remaining}
        group = max(trials, key=lambda g: trials[g].mean())
        if trials[group].mean() <= best.mean():
            break
        chosen.append(group)
        remaining.remove(group)
        best = trials[group]
        report(" + ".join(chosen), best, baseline)
    return chosen


def round_one(train, y, species, chart, vocab):
    X = build_features(train, species, chart, vocab, ROUND_ONE)
    print("features:", X.shape[1], "missing values:", int(X.isna().sum().sum()))
    folds = make_folds(y)

    print("\nOne group at a time on top of the base features")
    base = {}
    for model_name, make_model in (("logreg", make_logreg), ("hgb", make_hgb)):
        base[model_name] = cv_scores(make_model, X[select_columns(X, ["base"])], y, folds)
        report(f"{model_name}: base", base[model_name])
        for group in ROUND_ONE:
            if group != "base":
                scores = cv_scores(make_model, X[select_columns(X, ["base", group])], y, folds)
                report(f"{model_name}: base + {group}", scores, base[model_name])

    print("\nForward selection with hgb")
    candidates = [g for g in ROUND_ONE if g != "base"]
    chosen = ["base"] + forward_selection(X, y, folds, ["base"], candidates, base["hgb"])
    print("chosen groups:", chosen)

    print("\nModels on the chosen groups")
    X_chosen = X[select_columns(X, chosen)]
    for model_name, make_model in MODELS.items():
        report(f"{model_name} ({X_chosen.shape[1]} features)", cv_scores(make_model, X_chosen, y, folds), base["hgb"])
    print("\nModels on all groups")
    for model_name, make_model in MODELS.items():
        report(f"{model_name} ({X.shape[1]} features)", cv_scores(make_model, X, y, folds), base["hgb"])

    # the groups were picked on the folds above, so check again on folds not used for picking
    print("\nConfirmation on new folds")
    new_folds = make_folds(y, seed=2026)
    fresh_base = cv_scores(make_logreg, X[select_columns(X, ["base"])], y, new_folds)
    report("logreg base", fresh_base)
    for model_name, make_model in MODELS.items():
        report(f"{model_name} chosen", cv_scores(make_model, X_chosen, y, new_folds), fresh_base)


def round_two(train, y, species, chart, vocab):
    X = build_features(train, species, chart, vocab, GROUPS + ROUND_TWO)
    folds = make_folds(y)
    current = X[select_columns(X, GROUPS)]

    print("\nRound two: new groups on top of the first selection")
    for model_name, make_model in (("hgb", make_hgb), ("logreg", make_logreg)):
        base = cv_scores(make_model, current, y, folds)
        report(f"{model_name}: current", base)
        for group in ROUND_TWO:
            scores = cv_scores(make_model, X[select_columns(X, GROUPS + [group])], y, folds)
            report(f"{model_name}: + {group}", scores, base)

    print("\nForward selection with hgb")
    base = cv_scores(make_hgb, current, y, folds)
    chosen = forward_selection(X, y, folds, GROUPS, ROUND_TWO, base)
    print("chosen groups:", chosen)

    print("\nConfirmation on new folds")
    new_folds = make_folds(y, n_repeats=3, seed=99)
    for model_name, make_model in (("ensemble", MODELS["ensemble"]), ("hgb", make_hgb), ("logreg", make_logreg)):
        base = cv_scores(make_model, current, y, new_folds)
        report(f"{model_name}: current", base)
        scores = cv_scores(make_model, X[select_columns(X, GROUPS + chosen)], y, new_folds)
        report(f"{model_name}: + chosen", scores, base)


def main():
    train, test, chart, _ = load_all()
    y = np.array([int(b["player_won"]) for b in train])
    species = build_species_table(train + test)
    vocab = build_vocab(train)
    round_one(train, y, species, chart, vocab)
    round_two(train, y, species, chart, vocab)


if __name__ == "__main__":
    main()
