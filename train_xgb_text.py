import argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import xgboost as xgb

def main(csv, model_path, seed):
    df = pd.read_csv(csv)
    if 'reviews.text' not in df.columns or 'reviews.rating' not in df.columns:
        raise ValueError('Need reviews.text and reviews.rating columns for text baseline.')
    X_text = df['reviews.text'].fillna('')
    y = df['reviews.rating']
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2)
    X = vec.fit_transform(X_text)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    mdl = xgb.XGBRegressor(random_state=seed, n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.9, colsample_bytree=0.9, n_jobs=-1)
    mdl.fit(Xtr, ytr)
    yhat = mdl.predict(Xte)
    print('Text baseline — RMSE:', round(root_mean_squared_error(yte, yhat),4), ' MAE:', round(mean_absolute_error(yte, yhat),4),
          ' R2:', round(r2_score(yte, yhat),4))
    joblib.dump((mdl, vec), model_path)
    print('Saved model to', model_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model', default='outputs/xgb_text.pkl')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args.csv, args.model, args.seed)
