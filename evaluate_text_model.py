import argparse, joblib, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

def main(csv, model_path):
    df = pd.read_csv(csv)
    mdl, vec = joblib.load(model_path)
    X = vec.transform(df['reviews.text'].fillna(''))
    y = df['reviews.rating']
    yhat = mdl.predict(X)
    print('Eval text model — RMSE:', round(root_mean_squared_error(y, yhat),4),
          ' MAE:', round(mean_absolute_error(y, yhat),4),
          ' R2:', round(r2_score(y, yhat),4))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model', required=True)
    args = ap.parse_args()
    main(args.csv, args.model)
