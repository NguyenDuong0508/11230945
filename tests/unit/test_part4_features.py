"""Regression tests for financial semantics, event time and train isolation."""
from datetime import date
import numpy as np
import pandas as pd
import pytest
from fdic_ews.features.policy import *
from fdic_ews.features.financial import *
from fdic_ews.features.preprocessing import fit_training,predictor_matrix
from fdic_ews.features.pipeline import build_quarter,validate_partition,accounting

Q=date(2006,12,31); H=date(2007,12,31); R=date(2006,9,30)

def raw(d,certs=(1,2),asset=100):
    x=pd.DataFrame({k:[1.0]*len(certs) for k in RAW_NUMERIC}); x['CERT']=certs
    x['REPDTE']=d; x['STNAME']='NY'; x['ASSET']=asset; x['EQ']=10.; x['INTAN']=2.; x['NCLNLS']=3.; x['ORE']=1.; x['LNATRES']=2.; x['DEP']=80.; x['ROA']=list(range(1,len(certs)+1)); x['RBC1AAJ']=0.; x['REPORTS_TO_DATE']=5; x['PRIOR_GAP_QUARTERS']=0; x['DQ_ROE_ABS_GT_100']=False; x['ASSUMED_AVAILABLE_AT']=available_at(d)
    return x

def feat(d,certs=(1,2)):
    return report_features(raw(d,certs),raw(shift_quarter(d,-1),certs,80),raw(shift_quarter(d,-4),certs,50))

@pytest.mark.parametrize('event_type',['failure','assistance','institution_end','unknown_resolution'])
def test_future_event_invariance(event_type):
    assert outcome(Q,[],[H])==outcome(Q,[(date(2008,11,23),event_type)],[H,date(2009,3,31)])==(0,'observed_at_horizon',None)
    assert risk_status(Q,R,[(date(2008,11,23),event_type)])=='at_risk'

def test_failure_boundary_and_early_failure():
    for d in (date(2007,1,1),H): assert outcome(Q,[(d,'failure')],[])[0]==1
    assert risk_status(Q,R,[(Q,'failure')])=='prior_exit'
    assert outcome(Q,[(Q,'failure')],[H])[0]==0 # risk gate is separate

def test_assistance_not_failure():
    assert outcome(Q,[(date(2007,6,1),'assistance')],[H])[0]==0
    assert outcome(Q,[(date(2007,6,1),'assistance')],[])[0] is None

def test_competing_exit():
    e=[(date(2007,6,1),'institution_end')]
    assert outcome(Q,e,[])[0] is None
    assert outcome(Q,e,[],candidate_exit_zero=True)[0]==0
    assert outcome(Q,e,[H],candidate_exit_zero=True)[0] is None
    assert outcome(Q,[(date(2007,6,1),'verified_nonfailure_exit')],[])[0]==0

def test_conflicting_failure_is_horizon_local():
    e=[(date(2007,1,1),'failure')]
    assert outcome(Q,e,[H])[0] is None
    assert outcome(Q,e,[date(2008,3,31)])[0]==1
    assert outcome(Q,e+[(date(2008,3,31),'failure')],[])[0]==1
    assert outcome(Q,e+[(date(2007,6,1),'failure')],[])[0] is None

def test_deduplicate_and_sentinel():
    row=dict(CERT=1,linked=True,date_status='valid',event_date=H,event_type='failure')
    assert event_index([row,row,{**row,'event_date':date(9999,12,31),'event_type':'institution_end'}])=={1:((H,'failure'),)}

def test_split_and_maturity():
    assert_split(); assert split_for(TRAIN_END)=='train'; assert split_for(date(2001,3,31))=='purge_train_validation'; assert split_for(VALID_START)=='validation'; assert split_for(date(2007,12,31))=='purge_development_test'; assert split_for(TEST_START)=='test'; assert split_for(LAST)=='unused_post_test'
    with pytest.raises(ValueError): outcome(date(2025,9,30),[],[])

def test_financial_formulas():
    f=feat(R); assert f.LOG_ASSET.iloc[0]==pytest.approx(np.log1p(100)); assert f.TEXAS_LIKE_PCT.iloc[0]==40; assert f.TANGIBLE_EQUITY_ASSETS_PCT.iloc[0]==8; assert f.ASSET_GROWTH_QOQ_PCT.iloc[0]==25; assert f.ASSET_GROWTH_YOY_PCT.iloc[0]==100
    assert f.RBC1AAJ_ZERO.iloc[0]==1 and f[CAPITAL[0]].iloc[0]==0

def test_missing_predecessor_and_denominators():
    x=raw(R); x.loc[0,'ASSET']=0; x.loc[0,'EQ']=-5; x.loc[1,'INTAN']=np.nan
    prev=raw(shift_quarter(R,-1),(2,),80); f=report_features(x,prev,prev.iloc[:0]); assert np.isnan(f.LOG_ASSET.iloc[0]); assert np.isnan(f.ASSET_GROWTH_QOQ_PCT.iloc[0]); assert np.isnan(f.TEXAS_LIKE_PCT.iloc[0]); assert np.isnan(f.TEXAS_LIKE_PCT.iloc[1]); assert f.ASSET_GROWTH_QOQ_PCT.iloc[1]==25

def test_exact_history_rejects_wrong_quarter():
    with pytest.raises(ValueError,match='Non-exact'):
        report_features(raw(R),raw(R),raw(shift_quarter(R,-4)))

def test_availability_and_staleness():
    old=shift_quarter(R,-1); latest=feat(R,(1,)); earlier=feat(old,(1,2)); future=feat(Q,(3,)); stale=feat(shift_quarter(R,-2),(4,))
    z=latest_reports(Q,[(Q,future),(R,latest),(old,earlier),(shift_quarter(R,-2),stale)])
    assert set(z.CERT)=={1,2}; assert z.loc[z.CERT==1,'REPORT_DATE'].iloc[0]==pd.Timestamp(R); assert z.loc[z.CERT==2,'REPORT_DATE'].iloc[0]==pd.Timestamp(old)
    assert set(latest_reports(Q,[(R,latest),(old,earlier)],120).CERT)=={1,2}

def test_peer_leave_one_out_and_outcome_independence():
    f=feat(R); z=decision_features(f,Q); assert z.ROA_PEER_DEV_PPT.tolist()==[-1.,1.]
    a,_,_=build_quarter(Q,[(R,f)],{}, {H:{1,2}})
    b,_,_=build_quarter(Q,[(R,f)],{1:((date(2007,3,1),'failure'),)}, {})
    pd.testing.assert_frame_equal(a[list(FEATURES)],b[list(FEATURES)])
    assert b.Y_FAIL_4Q.iloc[0]==1
    assert np.isnan(decision_features(f.iloc[:1],Q).ROA_PEER_DEV_PPT.iloc[0])

def test_prior_exit_only_affects_risk_and_future_does_not():
    f=feat(R); a,_,_=build_quarter(Q,[(R,f)],{}, {H:{1,2}})
    b,_,_=build_quarter(Q,[(R,f)],{1:((date(2008,11,23),'assistance'),)}, {H:{1,2},date(2009,3,31):{1}})
    pd.testing.assert_frame_equal(a,b); validate_partition(Q,a)
    z,audit,_=build_quarter(Q,[(R,f)],{1:((date(2006,12,1),'failure'),)}, {H:{1,2}})
    assert z.CERT.tolist()==[2] and audit['prior_exit']==1

def test_empty_candidate():
    z,a,r=build_quarter(FIRST,[],{},{}); validate_partition(FIRST,z); assert z.empty and a['candidate_rows']==0

def test_train_only_preprocessor_and_allowlist():
    q=date(2000,12,31); f=feat(shift_quarter(q,-1)); z,_,_=build_quarter(q,[(shift_quarter(q,-1),f)],{}, {shift_quarter(q,4):{1,2}})
    pipe,info=fit_training(z); assert info['training_rows']==2; assert len(info['feature_names'])==len(FEATURES); assert np.isfinite(pipe.transform(predictor_matrix(z))).all()
    assert set(predictor_matrix(z))==set(FEATURES); assert predictor_matrix(z,zero_as_missing=True)[CAPITAL[0]].isna().all()
    other=z.assign(SPLIT='test');
    with pytest.raises(ValueError): fit_training(pd.concat([z,other]))
    with pytest.raises(ValueError): fit_training(z.assign(DECISION_DATE=TEST_START))
    med=pipe.named_steps['imputer'].statistics_.copy(); changed=z.assign(LOG_ASSET=1e9); pipe.transform(predictor_matrix(changed)); np.testing.assert_array_equal(med,pipe.named_steps['imputer'].statistics_)

def test_holdout_gate():
    audits=[]
    for s in ('train','validation','test'):
        audits.append(dict(split=s,candidate_rows=2,risk_rows=2,prior_exit=0,prior_exit_conflict=0,report_unavailable_or_stale=0,labelled=2,positive=1,negative=1,censored=0))
    with pytest.raises(RuntimeError,match='minimum 50'): accounting(audits,[],{'test':{(1,date(2009,1,1))}})
