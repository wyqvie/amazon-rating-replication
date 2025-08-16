import pandas as pd
import argparse

REQ = ['reviews.rating','reviews.numHelpful','reviews.doRecommend',
       'review_length','sentiment_score','sentiment_category']

def preprocess(csv_in: str, csv_out: str):
    df = pd.read_csv(csv_in)
    missing = [c for c in REQ if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df = df.copy()
    df['reviews.doRecommend'] = df['reviews.doRecommend'].astype(int)
    # Basic sanity
    df = df.dropna(subset=['reviews.rating'])
    df.to_csv(csv_out, index=False)
    print(f'Saved cleaned file to {csv_out} | shape={df.shape}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    preprocess(args.csv, args.out)
