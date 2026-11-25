from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from fdic_ews.quality.policy import (parse_date, shift_quarter, available_at, latest_available_report,
    classify_restype, event_records, terminal_audit, index_events, horizon_outcome)
from fdic_ews.quality.financial import clean_quarter, NUMERIC, availability_counts
from fdic_ews.quality.panel import event_diagnostics, summarize_fields, save_csv, make_figures

Q=date(2006,12,31); H=date(2007,12,31)

def e(kind, day):
    return dict(event_type=kind,event_date=day,linked=True,date_status="valid")

@pytest.mark.parametrize("value,day,status", [("20081123",date(2008,11,23),"valid"),("2008-11-23",date(2008,11,23),"valid"),
    ("11/23/2008",date(2008,11,23),"valid"),(None,None,"missing"),("",None,"missing"),("20080230",None,"invalid"),("0",None,"invalid")])
def test_dates(value,day,status):
    assert parse_date(value)==(day,status)

@pytest.mark.parametrize("value,kind",[("FAILURE","failure"),(" assistance ","assistance"),("PA","unknown_resolution"),(None,"unknown_resolution")])
def test_event_type(value,kind):
    assert classify_restype(value)==kind

def test_citibank_regression():
    events=event_records([dict(CERT=7213,FIN=1,RESTYPE="ASSISTANCE",FAILDATE="20081123")],[],{7213})
    assert events[0]["event_type"]=="assistance"
    assert horizon_outcome(Q,[],[Q,H])==horizon_outcome(Q,events,[Q,H,date(2009,3,31)])==(0,"observed_at_horizon")
    assert terminal_audit({7213:(Q,date(2009,3,31),3)},events)[0]["audit_class"]=="unresolved_discontinuation"

@pytest.mark.parametrize("extra",[e("assistance",date(2008,11,23)),e("failure",date(2008,11,23)),e("institution_end",date(2008,11,23)),e("unknown_resolution",date(2008,11,23))])
def test_horizon_invariance(extra):
    assert horizon_outcome(Q,[extra],[Q,H,date(2009,3,31)])==horizon_outcome(Q,[],[Q,H])

def test_failure_boundary():
    assert horizon_outcome(Q,[e("failure",H)],[Q,H])==(1,"failure_within_horizon")
    assert horizon_outcome(Q,[e("failure",Q)],[Q])[0] is None
    assert horizon_outcome(Q,[e("failure",date(2008,1,1))],[Q,H])[0]==0

def test_competing_and_conflicts():
    exit=e("institution_end",date(2007,6,1))
    assert horizon_outcome(Q,[exit],[Q],other_exit="zero")[0]==0
    assert horizon_outcome(Q,[exit],[Q],other_exit="censor")[0] is None
    assert horizon_outcome(Q,[exit],[Q,H])[0] is None
    assert horizon_outcome(Q,[e("failure",date(2007,6,1))],[Q,H])[0] is None
    assert horizon_outcome(Q,[e("unknown_resolution",date(2007,6,1))],[Q,H])[0] is None
    assert horizon_outcome(Q,[],[Q,date(2008,3,31)])[0] is None
    assert horizon_outcome(date(2025,9,30),[],[])[0] is None

def test_failure_conflict_after_horizon_does_not_change_label():
    ev=e("failure",date(2007,9,1))
    assert horizon_outcome(Q,[ev],[Q])[0]==1
    assert horizon_outcome(Q,[ev,e("failure",date(2010,1,1))],[Q,date(2008,3,31)])[0]==1

def test_availability_and_quarters():
    assert shift_quarter(Q,4)==H
    assert shift_quarter(Q,1)==date(2007,3,31)
    assert available_at(Q,90)==date(2007,3,31)
    assert latest_available_report(date(2007,3,31),[Q,date(2007,3,31)])==Q
    assert latest_available_report(date(2007,3,31),[Q,date(2007,3,31)],120) is None


def frame(q,certs=(1,2),**kwargs):
    data={f:[1.]*len(certs) for f in NUMERIC}
    data.update(CERT=list(certs),REPDTE=[q]*len(certs),STNAME=["NY"]*len(certs))
    data.update(kwargs)
    return pd.DataFrame(data)


def test_cleaning_flags_and_zero_preservation():
    df=frame(Q,ASSET=[0.,100.],DEP=[5.,120.],EQ=[-2.,1.],ROE=[-200.,2.],RBCRWAJ=[0.,np.nan],NETINC=[np.inf,2.])
    clean,audit,flags,flows,caps=clean_quarter(df,Q,{})
    assert len(clean)==2 and np.isnan(clean.ASSET[0]) and clean.DQ_ASSET_NONPOSITIVE[0]
    assert clean.EQ[0]==-2 and clean.ROE[0]==-200 and clean.RBCRWAJ[0]==0
    assert clean.DQ_DEP_GT_ASSET[1] and clean.DQ_NONFINITE_VALUE[0]
    assert np.isnan(clean.NETINC[0]) and np.isinf(df.NETINC[0])
    assert next(x for x in caps if x['field']=='RBCRWAJ')['sensitivity_nonnull']==0
    assert next(x for x in audit if x['field']=='ASSET')['zeros']==1


def test_ytd_and_gap():
    h={}; q1=date(2012,3,31); q2=date(2012,6,30); q4=date(2012,12,31); qn=date(2013,3,31)
    a=clean_quarter(frame(q1,NETINC=[10.,-2.]),q1,h)[0]
    b=clean_quarter(frame(q2,NETINC=[25.,-5.]),q2,h)[0]
    c=clean_quarter(frame(q4,NETINC=[40.,1.]),q4,h)[0]
    d=clean_quarter(frame(qn,NETINC=[4.,-1.]),qn,h)[0]
    assert a.NETINC_Q_IF_YTD.tolist()==[10.,-2.]
    assert b.NETINC_Q_IF_YTD.tolist()==[15.,-3.]
    assert c.NETINC_Q_IF_YTD.isna().all() and c.DQ_REPORTING_GAP.all()
    assert c.PRIOR_GAP_QUARTERS.tolist()==[1,1] and d.NETINC_Q_IF_YTD.tolist()==[4.,-1.]
    assert d.REPORTS_TO_DATE.tolist()==[4,4]


def test_prefix_invariance_and_history_order():
    h={}; x=frame(Q)
    old=clean_quarter(x,Q,h)[0]
    future=date(2007,3,31); clean_quarter(frame(future,ASSET=[-1.,1.]),future,h)
    pd.testing.assert_frame_equal(old,clean_quarter(x,Q,{})[0])
    with pytest.raises(ValueError,match="History"):
        clean_quarter(x,Q,h)
    assert all(x['available_rows']==0 for x in availability_counts(Q,old) if x['decision_quarter']==str(Q))


def test_duplicates_rejected():
    with pytest.raises(ValueError,match="keys"):
        clean_quarter(frame(Q,certs=(1,1)),Q,{})


def test_reference_reconciliation():
    bounds={1:(Q,H,5),2:(Q,H,5)}
    events=event_records([dict(CERT=1,FIN=1,RESTYPE='FAILURE',FAILDATE='20071231'),
                          dict(CERT=2,FIN=2,RESTYPE='ASSISTANCE',FAILDATE='20081123'),
                          dict(CERT=None,FIN=3,RESTYPE='FAILURE',FAILDATE='bad')],
                         [dict(CERT=2,ENDEFYMD='20080101',ACTIVE='0')],set(bounds))
    _,info=event_diagnostics(events,bounds)
    assert info['failure_type_records']==2 and info['assistance_records']==1 and info['unlinked_failure_type_records']==1
    assert len(index_events(events)[2])==2
    assert sum(r['reports'] for r in terminal_audit(bounds,events))==10


def test_figures(tmp_path):
    q=date(2020,3,31); _,audit,_,_,caps=clean_quarter(frame(q),q,{})
    save_csv(tmp_path/'quarter_coverage.csv',[{'quarter':str(q),'rows':2}])
    save_csv(tmp_path/'field_summary.csv',summarize_fields(audit))
    save_csv(tmp_path/'capital_audit.csv',caps)
    make_figures(tmp_path)
    assert all((tmp_path/f).stat().st_size>1000 for f in ('coverage.png','missingness.png','capital_reporting.png'))


def test_verified_exit_and_prior_candidate():
    end=e("verified_nonfailure_exit",date(2007,6,1))
    assert horizon_outcome(Q,[end],[Q])==(0,"verified_nonfailure_exit")
    assert horizon_outcome(Q,[end],[Q],verified_exit="censor")[0] is None
    assert horizon_outcome(Q,[e("institution_end",date(2005,1,1))],[Q,H])[0]==0
