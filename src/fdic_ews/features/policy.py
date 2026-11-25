"""Fixed chronological protocol and horizon-local event adjudication."""
from datetime import date
from collections import defaultdict
from fdic_ews.quality.policy import shift_quarter, available_at, quarter_id

FIRST = date(1984,3,31)
LAST = date(2025,6,30)
STUDY_END = date(2026,6,30)
TRAIN_END = date(2000,12,31)
VALID_START = date(2002,3,31)
DEV_END = date(2006,12,31)
TEST_START = date(2008,3,31)
TEST_END = date(2014,12,31)
LAG = 90
MAX_AGE = 2
MIN_TEST_FAILURES = 50
SEED = 11230945


def split_for(q):
    quarter_id(q)
    if not FIRST <= q <= LAST: raise ValueError('Decision outside mature study scope')
    if q <= TRAIN_END: return 'train'
    if q < VALID_START: return 'purge_train_validation'
    if q <= DEV_END: return 'validation'
    if q < TEST_START: return 'purge_development_test'
    if q <= TEST_END: return 'test'
    return 'unused_post_test'


def event_index(records):
    """Keep dated linked evidence; exclude open-ended sentinel dates."""
    result=defaultdict(set)
    for r in records:
        d=r['event_date']; typ=r['event_type']
        if r['linked'] and r['CERT'] is not None and d is not None and r['date_status']=='valid' and d.year!=9999:
            result[int(r['CERT'])].add((d,typ))
    return {k:tuple(sorted(v)) for k,v in result.items()}


def risk_status(q, report, events):
    """As-of risk-set proxy: no dated failure or candidate exit on/before q."""
    if available_at(report,LAG)>q or not 1<=quarter_id(q)-quarter_id(report)<=MAX_AGE:
        return 'report_unavailable_or_stale'
    prior=[(d,t) for d,t in events if d<=q and t in ('failure','institution_end','verified_nonfailure_exit')]
    if prior:
        return 'prior_exit_conflict' if any(d<report for d,t in prior) else 'prior_exit'
    return 'at_risk'


def outcome(q, events, future_reports, *, candidate_exit_zero=False):
    """Only (q,h] events and reports are admissible for outcome adjudication.

    Candidate institutional ends are censored, not asserted to be mergers.
    Later reports/events do not modify historical adjudication.
    """
    h=shift_quarter(q,4)
    if q>LAST or h>STUDY_END: raise ValueError('Immature label')
    ev={(d,t) for d,t in events if q<d<=h}
    reports={d for d in future_reports if q<d<=h}
    f=sorted({d for d,t in ev if t=='failure'})
    ends=sorted({d for d,t in ev if t in ('institution_end','verified_nonfailure_exit')})
    verified={d for d,t in ev if t=='verified_nonfailure_exit'}
    if any(t=='unknown_resolution' for d,t in ev): return None,'unknown_in_horizon',None
    if len(f)>1 or len(ends)>1: return None,'multiple_in_horizon_events',None
    if f:
        d=f[0]
        if any(r>d for r in reports) or any(e<d for e in ends) or verified:
            return None,'conflicting_failure_evidence',None
        return 1,'failure_within_horizon',d
    if ends:
        d=ends[0]
        if any(r>d for r in reports): return None,'conflicting_exit_evidence',None
        if d in verified: return 0,'verified_nonfailure_exit',None
        return (0,'assumed_nonfailure_exit',None) if candidate_exit_zero else (None,'unconfirmed_competing_exit',None)
    if h in reports: return 0,'observed_at_horizon',None
    return None,'incomplete_followup',None


def assert_split():
    assert shift_quarter(TRAIN_END,4)<VALID_START
    assert shift_quarter(DEV_END,4)<TEST_START
    assert shift_quarter(LAST,4)==STUDY_END
