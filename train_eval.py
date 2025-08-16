import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

from .utils import make_preprocessor, get_feature_names

REQ = ['reviews.rating','reviews.numHelpful','reviews.doRecommend',
       'review_length','sentiment_score','sentiment_category']

def evaluate(y_true, y_pred):
    return {
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def run(csv_path, outdir, seed, fast, full, subsample, no_fig):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    miss = [c for c in REQ if c not in df.columns]
    if miss:
        raise ValueError(f'Missing required columns: {miss}')
    df = df.copy()
    df['reviews.doRecommend'] = df['reviews.doRecommend'].astype(int)
    if subsample and subsample < len(df):
        df = df.sample(n=subsample, random_state=seed).reset_index(drop=True)

    X_no = df[['reviews.numHelpful','reviews.doRecommend','review_length']]
    X_yes = df[['reviews.numHelpful','reviews.doRecommend','review_length','sentiment_score','sentiment_category']]
    y = df['reviews.rating']

    Xtr_w, Xte_w, ytr, yte = train_test_split(X_yes, y, test_size=0.2, random_state=seed)
    Xtr_wo, Xte_wo, _, _ = train_test_split(X_no, y, test_size=0.2, random_state=seed)

    models = {'Linear Regression': LinearRegression(),
              'Random Forest': RandomForestRegressor(random_state=seed, n_estimators=100, n_jobs=-1)}
    if full and XGB_OK:
        models['XGBoost'] = xgb.XGBRegressor(random_state=seed, n_estimators=200, learning_rate=0.1,
                                             max_depth=6, subsample=0.9, colsample_bytree=0.9, n_jobs=-1)

    rows, pipes = [], {}

    for name, mdl in models.items():
        # without sentiment
        pre0 = make_preprocessor(False); pipe0 = Pipeline([('pre', pre0), ('model', mdl)])
        pipe0.fit(Xtr_wo, ytr)
        yhat0 = pipe0.predict(Xte_wo)
        m0 = evaluate(yte, yhat0); m0.update({'Model': name, 'Sentiment Included': 'No'})
        rows.append(m0); pipes[(name,'No')] = pipe0

        # with sentiment
        pre1 = make_preprocessor(True); pipe1 = Pipeline([('pre', pre1), ('model', mdl)])
        pipe1.fit(Xtr_w, ytr)
        yhat1 = pipe1.predict(Xte_w)
        m1 = evaluate(yte, yhat1); m1.update({'Model': name, 'Sentiment Included': 'Yes'})
        rows.append(m1); pipes[(name,'Yes')] = pipe1

    res = pd.DataFrame(rows).round(4).sort_values(['Model','Sentiment Included'])
    res_path = os.path.join(outdir, 'rating_prediction_results.csv'); res.to_csv(res_path, index=False)

    # Best by RMSE
    best = res.loc[res['RMSE'].idxmin()]
    best_pipe = pipes[(best['Model'], best['Sentiment Included'])]
    Xte_best = Xte_w if best['Sentiment Included']=='Yes' else Xte_wo
    yhat_best = best_pipe.predict(Xte_best)

    if not no_fig:
        # Figure: Actual vs Predicted
        plt.figure()
        plt.scatter(yte, yhat_best, alpha=0.6)
        mn, mx = min(yte.min(), yhat_best.min()), max(yte.max(), yhat_best.max())
        plt.plot([mn,mx],[mn,mx])
        plt.xlabel('Actual Rating'); plt.ylabel('Predicted Rating')
        plt.title(f'Actual vs Predicted — {best["Model"]} (Sentiment: {best["Sentiment Included"]})')
        plt.savefig(os.path.join(outdir,'actual_vs_predicted.png'), bbox_inches='tight'); plt.close()

        # Feature importance or LR coefficients
        pre = best_pipe.named_steps['pre']
        feats = get_feature_names(pre, best['Sentiment Included']=='Yes')
        model = best_pipe.named_steps['model']

        if hasattr(model,'feature_importances_'):
            top = (pd.DataFrame({'feature': feats, 'importance': model.feature_importances_})
                     .sort_values('importance', ascending=False).head(15))
            plt.figure(); plt.barh(top['feature'], top['importance']); plt.gca().invert_yaxis()
            plt.xlabel('Importance'); plt.title(f'Top Features — {best["Model"]}')
            plt.savefig(os.path.join(outdir,'feature_importance.png'), bbox_inches='tight'); plt.close()
        elif hasattr(model,'coef_'):
            coefs = np.ravel(model.coef_)
            top = (pd.DataFrame({'feature': feats, 'abs_coef': np.abs(coefs)})
                     .sort_values('abs_coef', ascending=False).head(15))
            plt.figure(); plt.barh(top['feature'], top['abs_coef']); plt.gca().invert_yaxis()
            plt.xlabel('|Coefficient|'); plt.title('Linear Regression — Top Coefficients (|β|)')
            plt.savefig(os.path.join(outdir,'feature_importance.png'), bbox_inches='tight'); plt.close()

        # Methodology diagram
        text = ("Phase 1: Data Collection\n- Amazon reviews dataset\n\n"
                "Phase 2: Data Preprocessing\n- Clean, type casting, split\n\n"
                "Phase 3: Feature Engineering\n- Helpful votes, recommend flag, length\n"
                "- Sentiment score + one-hot category\n\n"
                "Phase 4: Model Training\n- Linear Regression / Random Forest / XGBoost\n\n"
                "Phase 5: Evaluation\n- RMSE, MAE, R²; ablation (+/- sentiment)")
        plt.figure(); plt.axis('off')
        plt.text(0.05, 0.95, 'Methodology Workflow', fontsize=14, va='top')
        plt.text(0.05, 0.88, text, fontsize=10, va='top')
        plt.savefig(os.path.join(outdir,'methodology_diagram.png'), bbox_inches='tight'); plt.close()

    print(f'Saved results to: {res_path}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/reviews.csv')
    ap.add_argument('--outdir', default='outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--subsample', type=int, default=None)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--no-fig', action='store_true')
    args = ap.parse_args()
    if not (args.fast or args.full):
        args.fast = True
    run(args.csv, args.outdir, args.seed, args.fast, args.full, args.subsample, args.no_fig)
