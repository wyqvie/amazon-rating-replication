# Amazon Reviews — Rating Prediction with Sentiment (Replication Package)

Reproducible code for predicting product ratings from Amazon reviews using engineered metadata and sentiment features.

## Project structure
```
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── reviews.csv            # place dataset here (see columns below)
├── outputs/                   # results will be saved here
├── scripts/
│   ├── run_fast.sh            # LR + RF (quick)
│   └── run_full.sh            # LR + RF + XGBoost
└── src/
    ├── preprocess.py
    ├── utils.py
    ├── train_eval.py          # main pipeline (with/without sentiment)
    ├── train_xgb_text.py      # optional TF‑IDF + XGBoost text model (baseline)
    └── evaluate_text_model.py # evaluate saved text model
```

## Dataset
Use the Datafiniti Amazon reviews CSV from Kaggle. Required columns:
- `reviews.rating` *(float/int; target)*
- `reviews.numHelpful` *(int)*
- `reviews.doRecommend` *(bool/int; 0/1)*
- `review_length` *(int)*
- `sentiment_score` *(float)*
- `sentiment_category` *(str)*
- (optional) `reviews.text` *(str; for TF‑IDF baseline)*

Place your file at `data/reviews.csv` or pass `--csv`.

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick run (tabular features: LR + RF)
```bash
python -m src.train_eval --csv data/reviews.csv --fast
```

## Full run (adds XGBoost tabular)
```bash
python -m src.train_eval --csv data/reviews.csv --full
```

## Optional: Text-only baseline (TF‑IDF + XGBoost)
```bash
python -m src.train_xgb_text --csv data/reviews.csv --model outputs/xgb_text.pkl
python -m src.evaluate_text_model --csv data/reviews.csv --model outputs/xgb_text.pkl
```

## Outputs
- `outputs/rating_prediction_results.csv` — RMSE, MAE, R² (with/without sentiment)
- `outputs/actual_vs_predicted.png` — best model scatter
- `outputs/feature_importance.png` — feature importances or LR coefficients
- `outputs/methodology_diagram.png` — methodology figure

## Notes
- Fixed 80/20 split with seed for reproducibility.
- Pipelines avoid data leakage; One-Hot for sentiment category.
- `--subsample N` is available for quicker local runs.

