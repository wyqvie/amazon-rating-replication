from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

BASE_NUM = ['reviews.numHelpful','reviews.doRecommend','review_length']
SENT_NUM = BASE_NUM + ['sentiment_score']
SENT_CAT = ['sentiment_category']

def make_preprocessor(include_sentiment: bool) -> ColumnTransformer:
    if include_sentiment:
        return ColumnTransformer([
            ('num','passthrough', SENT_NUM),
            ('cat', OneHotEncoder(handle_unknown='ignore'), SENT_CAT)
        ])
    else:
        return ColumnTransformer([('num','passthrough', BASE_NUM)])

def get_feature_names(pre: ColumnTransformer, include_sentiment: bool):
    num = pre.transformers_[0][2]
    cats = []
    if include_sentiment and len(pre.transformers_) > 1:
        enc = pre.named_transformers_['cat']
        if hasattr(enc,'get_feature_names_out'):
            cats = list(enc.get_feature_names_out(SENT_CAT))
    return list(num) + cats
