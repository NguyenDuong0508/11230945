"""Row ranking, event detection, and exact empirical threshold optimization."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss


def event_scores(pred):
    p=pred.loc[pred.Y_FAIL_4Q.eq(1)]
    if p.FAILURE_DATE.isna().any(): raise ValueError('Positive row without event date')
    return p.groupby(['CERT','FAILURE_DATE'],sort=False).score.max()


def top_quarter_recall(pred, fraction=.05):
    if not 0<fraction<=1: raise ValueError('Invalid capacity')
    ordered=pred.sort_values(['DECISION_DATE','score','CERT'],ascending=[True,False,True],kind='stable')
    rank=ordered.groupby('DECISION_DATE').cumcount()
    cap=np.ceil(ordered.groupby('DECISION_DATE').CERT.transform('size')*fraction).astype(int)
    return float(ordered.loc[rank<cap,'Y_FAIL_4Q'].sum()/pred.Y_FAIL_4Q.sum()) if pred.Y_FAIL_4Q.sum() else None


def ranking(pred):
    y=pred.Y_FAIL_4Q.astype(int).to_numpy(); p=pred.score.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not set(y)<={0,1}: raise ValueError('Invalid predictions')
    both=len(np.unique(y))==2
    return dict(rows=len(y),positives=int(y.sum()),prevalence=float(y.mean()),
        ap=float(average_precision_score(y,p)) if both else None,
        roc_auc=float(roc_auc_score(y,p)) if both else None,
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),
        recall_top5pct_by_quarter=top_quarter_recall(pred))


def threshold(pred, c_fp, c_fn):
    """Enumerate unique scores, including no alarm; one missed cost per event."""
    if not np.isfinite([c_fp,c_fn]).all() or min(c_fp,c_fn)<=0: raise ValueError('Costs must be positive and finite')
    ev=event_scores(pred).to_numpy(dtype=float)
    neg=pred.loc[pred.Y_FAIL_4Q.eq(0),'score'].to_numpy(dtype=float)
    scores=pred.score.to_numpy(dtype=float)
    if not len(ev) or not len(neg) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any(): raise ValueError('Invalid threshold population')
    grid=np.r_[2.,np.unique(scores)[::-1]]
    fp=len(neg)-np.searchsorted(np.sort(neg),grid,side='left')
    missed=np.searchsorted(np.sort(ev),grid,side='left')
    loss=fp*c_fp+missed*c_fn
    i=int(np.argmin(loss))  # descending grid: exact ties favor the higher cutoff
    return dict(threshold=float(grid[i]),loss=float(loss[i]),false_alarms=int(fp[i]),missed_events=int(missed[i]),candidate_thresholds=len(grid))


def operating(pred, cutoff, c_fp, c_fn):
    y=pred.Y_FAIL_4Q.eq(1); a=pred.score.ge(cutoff)
    tp=int((a&y).sum()); fp=int((a&~y).sum()); fn=int((~a&y).sum()); tn=int((~a&~y).sum())
    ev=event_scores(pred); missed=int(ev.lt(cutoff).sum()); detected=len(ev)-missed
    alert=pred.loc[a&y].groupby(['CERT','FAILURE_DATE']).DECISION_DATE.min().reset_index()
    lead=(pd.to_datetime(alert.FAILURE_DATE)-pd.to_datetime(alert.DECISION_DATE)).dt.days
    return dict(threshold=float(cutoff),tp=tp,fp=fp,fn=fn,tn=tn,alerts=tp+fp,alert_rate=(tp+fp)/len(pred),
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        fpr=fp/(fp+tn) if fp+tn else None,events=len(ev),detected_events=detected,missed_events=missed,
        event_recall=detected/len(ev) if len(ev) else None,median_lead_days=float(lead.median()) if len(lead) else None,
        lead_at_least_90_days=int(lead.ge(90).sum()),loss_usd_thousands=float(fp*c_fp+missed*c_fn))


def event_detail(pred,cutoff):
    p=pred.loc[pred.Y_FAIL_4Q.eq(1)].copy()
    p['alert_date']=pd.to_datetime(p.DECISION_DATE).where(p.score.ge(cutoff))
    out=p.groupby(['CERT','FAILURE_DATE'],as_index=False).agg(positive_rows=('score','size'),max_score=('score','max'),first_alert=('alert_date','min'))
    out['detected']=out.first_alert.notna()
    out['lead_days']=(pd.to_datetime(out.FAILURE_DATE)-out.first_alert).dt.days
    return out


def calibration(pred):
    f=pred.copy(); f['bin']=pd.cut(f.score,[0,.001,.005,.01,.02,.05,.1,.2,.5,1],include_lowest=True)
    return f.groupby('bin',observed=True).agg(rows=('score','size'),mean_score=('score','mean'),observed_rate=('Y_FAIL_4Q','mean')).reset_index().assign(bin=lambda x:x.bin.astype(str))
