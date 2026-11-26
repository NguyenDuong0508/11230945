"""Development-only economic proxies, with conservative vintage gating."""
from datetime import date
import numpy as np
import pandas as pd
from fdic_ews.features.policy import DEV_END
from fdic_ews.quality.policy import shift_quarter

CUTOFF=shift_quarter(DEV_END,4)


def parse_asof(value):
    # Reject ambiguous numeric/text formats rather than infer a recent vintage.
    from datetime import datetime
    for fmt in ('%Y-%m-%d','%Y%m%d','%m/%d/%Y'):
        try: return datetime.strptime(str(value).strip()[:10],fmt).date()
        except (ValueError,TypeError): pass
    return None


def estimate(dev,records):
    if not set(dev.SPLIT)<={'train','validation'} or pd.to_datetime(dev.HORIZON_END).max()>pd.Timestamp(CUTOFF): raise ValueError('Cost population crosses development cutoff')
    p=dev.loc[dev.Y_FAIL_4Q.eq(1)].copy()
    p['FAILURE_DATE']=pd.to_datetime(p.FAILURE_DATE).dt.date
    keys=set(zip(p.CERT.astype(int),p.FAILURE_DATE))
    costs={}; current=set(); bad_vintage=0
    for r in records:
        d=r['event_date']; key=(r['CERT'],d)
        if r['event_type']!='failure' or not r['linked'] or r['date_status']!='valid' or key not in keys: continue
        c=r.get('COST')
        if c is None or not np.isfinite(c) or c<=0: continue
        current.add(key); a=parse_asof(r.get('COSTMOSTRECENTASOF'))
        if a is None or not d<=a<=CUTOFF:
            bad_vintage+=1; continue
        costs.setdefault(key,set()).add(float(c))
    eligible=[next(iter(v)) for v in costs.values() if len(v)==1]
    # Asset fallback is a predeclared scenario, not a measured resolution expense.
    assets=np.expm1(dev.LOG_ASSET.dropna().to_numpy(dtype=float)); assets=assets[np.isfinite(assets)&(assets>0)]
    p=p.sort_values('DECISION_DATE').drop_duplicates(['CERT','FAILURE_DATE'],keep='last')
    fail_assets=np.expm1(p.LOG_ASSET.dropna().to_numpy(dtype=float)); fail_assets=fail_assets[np.isfinite(fail_assets)&(fail_assets>0)]
    if not len(assets) or not len(fail_assets): raise ValueError('Missing asset evidence for cost scenario')
    median_assets=float(np.median(assets)); median_failed=float(np.median(fail_assets))
    source='dated_fdic_cost' if len(eligible)>=20 else 'assumed_10pct_of_median_failed_bank_assets'
    fn=float(np.median(eligible)) if len(eligible)>=20 else .10*median_failed
    return dict(cutoff=str(CUTOFF),unit='USD thousands',represented_development_events=len(keys),positive_current_cost_events=len(current),
        eligible_dated_cost_events=len(eligible),conflicting_dated_cost_events=sum(len(v)>1 for v in costs.values()),excluded_cost_records_by_vintage=bad_vintage,
        median_eligible_fdic_cost=float(np.median(eligible)) if eligible else None,median_development_assets=median_assets,median_failed_event_assets=median_failed,
        missed_cost_source=source,c_fp=.001*median_assets,c_fn=fn,minimum_dated_cost_events=20,review_asset_fraction=.001,fallback_loss_fraction=.10)
