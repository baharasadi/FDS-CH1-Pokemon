"""Train the final models on the whole training set and write the submission files."""
from pathlib import Path

import numpy as np

from src.data import load_all
from src.features import build_features, build_species_table, build_vocab, select_columns
from src.models import MODELS

# first round of forward selection in run_cv.py
GROUPS = ["base", "board", "move_bag", "ko", "species", "boosts", "matchup", "trend", "team"]
# second round of forward selection in run_cv.py
EXTRA_GROUPS = ["hazards", "status"]

# file name: (model, feature groups)
FINAL_MODELS = {
    "logreg": ("logreg", GROUPS),
    "hgb_v2": ("hgb", GROUPS + EXTRA_GROUPS),
    "ensemble_v2": ("ensemble", GROUPS + EXTRA_GROUPS),
}


def main(out_dir="submissions"):
    train, test, chart, sample = load_all()
    y = np.array([int(b["player_won"]) for b in train])
    species = build_species_table(train + test)
    vocab = build_vocab(train)
    X_train = build_features(train, species, chart, vocab, GROUPS + EXTRA_GROUPS)
    X_test = build_features(test, species, chart, vocab, GROUPS + EXTRA_GROUPS)
    assert list(X_train.columns) == list(X_test.columns)

    Path(out_dir).mkdir(exist_ok=True)
    test_ids = [b["battle_id"] for b in test]
    for name, (model_name, groups) in FINAL_MODELS.items():
        columns = select_columns(X_train, groups)
        model = MODELS[model_name]()
        model.fit(X_train[columns], y)
        predictions = dict(zip(test_ids, model.predict(X_test[columns])))
        submission = sample.copy()
        submission["player_won"] = submission["battle_id"].map(predictions).astype(int)
        path = Path(out_dir) / f"submission_{name}.csv"
        submission.to_csv(path, index=False)
        print(path, "rows:", len(submission), "win rate:", round(submission["player_won"].mean(), 3))


if __name__ == "__main__":
    main()
