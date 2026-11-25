"""Training-only sklearn preprocessing; reconstruct a fresh pipeline for each fold."""
import numpy as np
from .financial import FEATURES, CAPITAL


def predictor_matrix(frame, *, zero_as_missing=False):
    # Explicit allowlist prevents labels, event dates and IDs entering the model.
    x=frame.loc[:,list(FEATURES)].astype(float).copy()
    if zero_as_missing:
        for c in CAPITAL: x[c]=x[c].mask(x[c].eq(0))
    return x


def make_preprocessor():
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import Pipeline
    return Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scaler',RobustScaler(quantile_range=(25,75)))])


def fit_training(frame, *, zero_as_missing=False):
    if frame.empty or set(frame.SPLIT)!= {'train'} or frame.Y_FAIL_4Q.isna().any():
        raise ValueError('Expected labelled training rows only')
    from .policy import FIRST, TRAIN_END
    import pandas as pd
    if not pd.to_datetime(frame.DECISION_DATE).between(pd.Timestamp(FIRST),pd.Timestamp(TRAIN_END)).all(): raise ValueError('Training dates outside fixed split')
    x=predictor_matrix(frame,zero_as_missing=zero_as_missing)
    pipe=make_preprocessor(); pipe.fit(x)
    # Check the trained transformation on training rows only, in bounded batches.
    for i in range(0,len(x),50000):
        if not np.isfinite(pipe.transform(x.iloc[i:i+50000])).all(): raise ValueError('Nonfinite transformed training matrix')
    im=pipe.named_steps['imputer']; sc=pipe.named_steps['scaler']
    diagnostic={'training_rows':len(x),'training_min':str(pd.to_datetime(frame.DECISION_DATE).min().date()),'training_max':str(pd.to_datetime(frame.DECISION_DATE).max().date()),'zero_as_missing':zero_as_missing,'feature_names':list(FEATURES),'transformed_names':pipe.get_feature_names_out().tolist(),'median':im.statistics_.tolist(),'center':sc.center_.tolist(),'scale':sc.scale_.tolist(),'missing_indicators':im.indicator_.features_.tolist(),'all_null_features':x.columns[x.isna().all()].tolist(),'constant_or_empty_features':x.columns[x.nunique(dropna=True)<=1].tolist()}
    return pipe,diagnostic
