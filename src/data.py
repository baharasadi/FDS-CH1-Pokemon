import json
from pathlib import Path

import pandas as pd

DATA_DIRS = [
    Path("/kaggle/input/competitions/fds-pokemon-battles-prediction-2026"),
    Path("/kaggle/input/fds-pokemon-battles-prediction-2026"),
    Path("fds-pokemon-battles-prediction-2026"),
]


def find_data_dir():
    for path in DATA_DIRS:
        if (path / "train.jsonl").exists():
            return path
    raise FileNotFoundError("competition data not found")


def load_battles(path):
    battles = []
    with open(path) as f:
        for line in f:
            battles.append(json.loads(line))
    return battles


def load_type_chart(path):
    chart = pd.read_csv(path)
    return {
        (row.attacking_type, row.defending_type): float(row.multiplier)
        for row in chart.itertuples(index=False)
    }


def load_all():
    data_dir = find_data_dir()
    train = load_battles(data_dir / "train.jsonl")
    test = load_battles(data_dir / "test.jsonl")
    chart = load_type_chart(data_dir / "type_chart_gen2.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, chart, sample
