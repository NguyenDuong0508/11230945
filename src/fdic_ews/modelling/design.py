"""Chronological model development and explicitly restricted predictor sets."""
from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd
from fdic_ews.features.financial import FEATURES, CAPITAL
from fdic_ews.features.policy import FIRST, TRAIN_END, DEV_END, TEST_START, TEST_END, SEED
from fdic_ews.quality.policy import shift_quarter

@dataclass(frozen=True)
class Fold:
    name: str
    train_end: date
    start: date
    end: date

FOLDS = (
    Fold('F1', date(1987,12,31), date(1989,3,31), date(1992,12,31)),
    Fold('F2', date(1992,12,31), date(1994,3,31), date(1997,12,31)),
    Fold('F3', date(1997,12,31), date(1999,3,31), date(2000,12,31)),
    Fold('F4', TRAIN_END, date(2002,3,31), DEV_END),
)
CORE_EXCLUDE = {'TANGIBLE_EQUITY_ASSETS_PCT','LOANS_DEPOSITS_PCT','ALLOWANCE_LOANS_PCT','TEXAS_LIKE_PCT','RBC1AAJ_ZERO','RBCRWAJ_ZERO','REPORT_AGE_QUARTERS','ROA_PEER_DEV_PPT'}
QUALITY_EXCLUDE = {'REPORTS_TO_DATE','PRIOR_GAP_QUARTERS','REPORT_AGE_QUARTERS','DQ_EXTREME_ROE','MISSING_FINANCIAL_COUNT','RBC1AAJ_ZERO','RBCRWAJ_ZERO'}
SPECS = {
    'prior': ('prior', FEATURES, False),
    'logistic': ('logistic', FEATURES, False),
    'hist_gradient': ('hist_gradient', FEATURES, False),
    'lightgbm_full': ('lightgbm', FEATURES, False),
    'lightgbm_core': ('lightgbm', tuple(k for k in FEATURES if k not in CORE_EXCLUDE), False),
    'lightgbm_no_peer': ('lightgbm', tuple(k for k in FEATURES if k != 'ROA_PEER_DEV_PPT'), False),
    'lightgbm_no_quality': ('lightgbm', tuple(k for k in FEATURES if k not in QUALITY_EXCLUDE), False),
    'lightgbm_zero_missing': ('lightgbm', FEATURES, True),
}
META = ['CERT','DECISION_DATE','HORIZON_END','FAILURE_DATE','SPLIT','Y_FAIL_4Q']
DEV_SPLITS = {'train','validation'}

def validate_design():
    assert shift_quarter(DEV_END,4)<TEST_START
    for i,f in enumerate(FOLDS):
        assert FIRST<=f.train_end<f.start<=f.end<=DEV_END
        assert shift_quarter(f.train_end,4)<f.start
        if i: assert shift_quarter(FOLDS[i-1].end,4)<f.start
    assert len(SPECS['lightgbm_core'][1])==21


def validate_rows(frame, role):
    from fdic_ews.features.policy import split_for
    if role not in ('development','test'): raise ValueError('Unknown population role')
    allowed=DEV_SPLITS if role=='development' else {'test'}
    if frame.empty or not set(frame.SPLIT)<=allowed or frame.Y_FAIL_4Q.isna().any(): raise ValueError('Invalid labelled population')
    if not set(frame.Y_FAIL_4Q.unique())<={0,1}: raise ValueError('Invalid binary labels')
    q=pd.to_datetime(frame.DECISION_DATE); h=pd.to_datetime(frame.HORIZON_END); d=pd.to_datetime(frame.FAILURE_DATE)
    if not (q.map({x:split_for(x.date()) for x in q.unique()})==frame.SPLIT).all(): raise ValueError('Split/date mismatch')
    if not (q.map({x:pd.Timestamp(shift_quarter(x.date(),4)) for x in q.unique()})==h).all(): raise ValueError('Invalid outcome horizon')
    pos=frame.Y_FAIL_4Q.eq(1)
    if not d.notna().eq(pos).all() or not ((d[pos]>q[pos])&(d[pos]<=h[pos])).all(): raise ValueError('Invalid failure dates')
    if frame.duplicated(['CERT','DECISION_DATE']).any(): raise ValueError('Duplicate decision keys')
    if set(frame.Y_FAIL_4Q.unique())!={0,1}: raise ValueError('Both classes required')


def split_fold(dev, fold):
    q=pd.to_datetime(dev.DECISION_DATE)
    train=dev.loc[q.le(pd.Timestamp(fold.train_end))]
    valid=dev.loc[q.between(pd.Timestamp(fold.start),pd.Timestamp(fold.end))]
    if train.empty or valid.empty or set(train.SPLIT)!={'train'} or not set(valid.SPLIT)<=DEV_SPLITS: raise ValueError('Invalid fold membership')
    if pd.to_datetime(train.HORIZON_END).max()>=pd.to_datetime(valid.DECISION_DATE).min(): raise ValueError('Overlapping training outcome horizon')
    for f in (train,valid):
        if set(f.Y_FAIL_4Q.unique())!={0,1}: raise ValueError(f'{fold.name}: both classes required; do not silently drop a fold')
    return train,valid


def matrix(frame, name):
    _,features,mask=SPECS[name]
    x=frame.loc[:,list(features)].astype(float).copy()
    if mask:
        for k in CAPITAL:
            if k in x: x[k]=x[k].mask(x[k].eq(0))
    if np.isinf(x.to_numpy()).any(): raise ValueError('Nonfinite raw predictor')
    return x


def clip_logistic(x):
    return np.clip(x,-25.,25.)


def estimator(name):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from fdic_ews.features.preprocessing import make_preprocessor
    kind,_,_=SPECS[name]
    if kind=='prior': return DummyClassifier(strategy='prior')
    prep=make_preprocessor().set_output(transform="pandas")
    steps=list(prep.steps)
    if kind=='logistic':
        steps.append(('clip',FunctionTransformer(clip_logistic,feature_names_out='one-to-one')))
        model=LogisticRegression(C=.5,max_iter=500,solver='lbfgs',random_state=SEED)
    elif kind=='hist_gradient':
        model=HistGradientBoostingClassifier(learning_rate=.07,max_iter=80,max_leaf_nodes=15,min_samples_leaf=120,early_stopping=False,random_state=SEED)
    elif kind=='lightgbm':
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=120,learning_rate=.05,num_leaves=15,max_depth=5,min_child_samples=120,reg_lambda=1.,verbosity=-1,n_jobs=2,random_state=SEED,deterministic=True,force_col_wise=True)
    else: raise ValueError('Unknown estimator')
    return Pipeline([*steps,('classifier',model)])


def fit_model(train,name):
    from threadpoolctl import threadpool_limits
    from time import perf_counter
    if train.empty or not set(train.SPLIT)<=DEV_SPLITS or set(train.Y_FAIL_4Q.unique())!={0,1}: raise ValueError('Only labelled development data may be fitted')
    if pd.to_datetime(train.DECISION_DATE).max()>pd.Timestamp(DEV_END) or pd.to_datetime(train.HORIZON_END).max()>=pd.Timestamp(TEST_START): raise ValueError('Training crosses holdout boundary')
    model=estimator(name)
    x=matrix(train,name); y=train.Y_FAIL_4Q.astype(int)
    t=perf_counter()
    with threadpool_limits(limits=2): model.fit(x,y)
    return model,perf_counter()-t


def score_model(model,frame,name):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2): p=np.asarray(model.predict_proba(matrix(frame,name))[:,list(model.classes_).index(1)],dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError('Invalid probabilities')
    out=frame.loc[:,META].copy().reset_index(drop=True)
    out['score']=p
    return out
