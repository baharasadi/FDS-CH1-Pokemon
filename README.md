# Pokemon battle prediction (FDS Kaggle challenge)

Predicts whether Player 1 wins a Gen 2 OU battle from their team, the opponent's lead and the first 30 turns.

## Setup

    ./setup_venv.sh

Put the competition files in `fds-pokemon-battles-prediction-2026/`. On Kaggle the input folder is found automatically.

## Usage

Run these from the repository root.

    python make_submission.py   # write the three final submissions to submissions/
    python run_cv.py            # compare feature groups and models with repeated 5-fold CV
    python tune.py              # random search for the model settings

## Final submissions

| File | Model | Features | Public score |
|---|---|---|---|
| `submission_ensemble_v2.csv` | logistic regression, gradient boosting and extra trees, soft voting | first and second round | 0.69400 |
| `submission_logreg.csv` | logistic regression | first round | 0.68800 |
| `submission_hgb_v2.csv` | gradient boosting | first and second round | 0.69133 |

## Code

- `src/data.py`: loading the battles and the type chart
- `src/features.py`: feature groups
- `src/models.py`: model definitions
- `src/validation.py`: cross-validation helpers
