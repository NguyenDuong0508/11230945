"""Sealed development selection, one test attempt, and read-only evidence verification."""
from pathlib import Path
import importlib.metadata as metadata
import platform
import json
import numpy as np
import pandas as pd
from fdic_ews.contracts import IntegrityError, read_json, write_json, sha256, utc_now
from fdic_ews.acquisition import code_revision
from fdic_ews.features.policy import TEST_START, TEST_END, DEV_END, TRAIN_END, MIN_TEST_FAILURES
from fdic_ews.features.financial import FEATURES
from .design import (FOLDS,SPECS,META,DEV_SPLITS,validate_design,validate_rows,split_fold,fit_model,score_model)
from .metrics import ranking,threshold,operating,event_detail,calibration
from .costs import estimate


def inventory(root,exclude=()):
    return {p.relative_to(root).as_posix():sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.relative_to(root).as_posix() not in exclude}


def checked_pointer(workspace, name, directory):
    p=read_json(workspace/name); root=(workspace/p['path']).resolve()
    if root.parent!=workspace/directory or sha256(root/'manifest.json')!=p['manifest_sha256']: raise IntegrityError(f'Invalid {directory} pointer')
    m=read_json(root/'manifest.json')
    if m['state']!='VERIFIED': raise IntegrityError(f'Unverified {directory}')
    return root,m,p['manifest_sha256']


def upstream(workspace):
    """Verify sealed identities and bytes without decoding holdout rows."""
    if (workspace/'pending_source.json').exists(): raise IntegrityError('Pending acquisition')
    bronze,bm,b=checked_pointer(workspace,'active_source.json','bronze')
    silver,sm,s=checked_pointer(workspace,'active_panel.json','silver')
    gold,gm,g=checked_pointer(workspace,'active_features.json','gold')
    if sm['source_manifest_sha256']!=b or gm['source_manifest_sha256']!=b or gm['silver_manifest_sha256']!=s: raise IntegrityError('Upstream lineage mismatch')
    if inventory(gold,('manifest.json',))!=gm['files']: raise IntegrityError('Gold artifact integrity mismatch')
    if inventory(silver,('manifest.json',))!=sm['files']: raise IntegrityError('Silver artifact integrity mismatch')
    from fdic_ews.contracts import quarter_ends
    from fdic_ews.features.policy import FIRST,LAST,LAG,MAX_AGE,VALID_START
    if set(gm['panel'])!=set(map(str,quarter_ends(FIRST,LAST))): raise IntegrityError('Incomplete Gold calendar')
    expected={'lag_days':LAG,'max_report_age_quarters':MAX_AGE,'train_end':str(TRAIN_END),'validation_start':str(VALID_START),'development_end':str(DEV_END),'test_start':str(TEST_START),'test_end':str(TEST_END),'last_decision':str(LAST)}
    if any(gm['policy'].get(k)!=v for k,v in expected.items()) or gm['summary']['feature_count']!=len(FEATURES) or gm['summary']['test_distinct_failure_certs_in_calendar_period']<MIN_TEST_FAILURES: raise IntegrityError('Gold policy mismatch')
    return gold,gm,silver,{'gold':g,'silver':s,'bronze':b}


def read_population(gold,gm,role):
    import pyarrow.parquet as pq
    from fdic_ews.features.policy import split_for
    allowed=DEV_SPLITS if role=='development' else {'test'} if role=='test' else set()
    if not allowed: raise ValueError('Unknown population')
    parts=[]
    for d,e in sorted(gm['panel'].items()):
        if split_for(pd.Timestamp(d).date()) not in allowed: continue
        path=gold/e['path']
        if sha256(path)!=e['sha256']: raise IntegrityError('Gold partition changed')
        f=pq.ParquetFile(path).read(columns=[*META,*FEATURES]).to_pandas()
        if len(f)!=e['rows'] or not pd.to_datetime(f.DECISION_DATE).eq(pd.Timestamp(d)).all(): raise IntegrityError('Gold partition mismatch')
        parts.append(f.loc[f.Y_FAIL_4Q.notna()])
    if not parts: raise IntegrityError('Empty population')
    frame=pd.concat(parts,ignore_index=True)
    for k in ('DECISION_DATE','HORIZON_END','FAILURE_DATE'): frame[k]=pd.to_datetime(frame[k])
    frame['Y_FAIL_4Q']=frame.Y_FAIL_4Q.astype('int8')
    frame=frame.sort_values(['DECISION_DATE','CERT']).reset_index(drop=True)
    validate_rows(frame,role)
    return frame


def check_population_counts(gold,frame,role):
    allowed=DEV_SPLITS if role=='development' else {'test'}
    expected=pd.read_csv(gold/'split_summary.csv').set_index('split')
    for name in allowed:
        f=frame.loc[frame.SPLIT.eq(name)]
        if len(f)!=int(expected.loc[name,'labelled']) or int(f.Y_FAIL_4Q.sum())!=int(expected.loc[name,'positive']): raise IntegrityError('Gold split population mismatch')


def read_cost_records(silver):
    import pyarrow.parquet as pq
    from .costs import CUTOFF
    records=pq.ParquetFile(silver/'event_ledger.parquet').read(columns=['CERT','event_date','event_type','linked','date_status','COST','COSTMOSTRECENTASOF']).to_pylist()
    # Only development-date records leave this reader.
    return [r for r in records if r['event_date'] is not None and r['event_date']<=CUTOFF]


def evidence_path(workspace,identity):
    # A stable Gold identity prevents creating a second test run by changing run IDs.
    return workspace/'modelling'/identity['gold']


def protocol_digest():
    # A sealed fit is bound to the code used for evaluation, not to notebook-only commits.
    root=Path(__file__).parent
    from hashlib import sha256 as h
    from fdic_ews.features import financial,policy,preprocessing
    paths=[*sorted(root.glob('*.py')),Path(financial.__file__),Path(policy.__file__),Path(preprocessing.__file__)]
    return h(b''.join(p.read_bytes() for p in paths)).hexdigest()


def csv(path,rows):
    pd.DataFrame(rows).to_csv(path,index=False)


def parquet(path,frame):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),path,compression='snappy')


def model_summary(fold_metrics):
    f=pd.DataFrame(fold_metrics)
    if f.groupby('model').fold.nunique().to_dict()!={n:len(FOLDS) for n in SPECS} or f.ap.isna().any(): raise IntegrityError('Incomplete model comparison')
    out=f.groupby('model',sort=False).agg(mean_ap=('ap','mean'),mean_roc_auc=('roc_auc','mean'),mean_brier=('brier','mean'),median_fit_seconds=('fit_seconds','median')).reset_index()
    out['early_folds_mean_ap']=out.model.map(f.loc[f.fold.ne('F4')].groupby('model').ap.mean())
    out['late_fold_ap']=out.model.map(f.loc[f.fold.eq('F4')].set_index('model').ap)
    # Deterministic tie-break by declared order, not test performance.
    out['order']=out.model.map({n:i for i,n in enumerate(SPECS)})
    out=out.sort_values(['mean_ap','order'],ascending=[False,True]).drop(columns='order').reset_index(drop=True)
    selected=out.loc[out.model.ne('prior'),'model'].iloc[0]
    out['selected']=out.model.eq(selected)
    return out,selected


def develop_frames(dev,records,root,*,spark=True):
    """Fit all specifications on disjoint chronological validation folds."""
    import joblib
    validate_design(); validate_rows(dev,'development')
    root.mkdir(parents=True,exist_ok=False)
    cost=estimate(dev,records); write_json(root/'cost_policy.json',cost)
    fp,fn=cost['c_fp'],cost['c_fn']
    fold_metrics=[]; fold_counts=[]; predictions={n:[] for n in SPECS}
    for fold in FOLDS:
        train,valid=split_fold(dev,fold)
        fold_counts.append(dict(fold=fold.name,train_end=str(fold.train_end),validation_start=str(fold.start),validation_end=str(fold.end),train_rows=len(train),train_positive=int(train.Y_FAIL_4Q.sum()),validation_rows=len(valid),validation_positive=int(valid.Y_FAIL_4Q.sum()),validation_events=int(valid.loc[valid.Y_FAIL_4Q.eq(1),['CERT','FAILURE_DATE']].drop_duplicates().shape[0])))
        for name in SPECS:
            model,elapsed=fit_model(train,name); pred=score_model(model,valid,name); pred['fold']=fold.name
            predictions[name].append(pred)
            fold_metrics.append(dict(fold=fold.name,model=name,fit_seconds=elapsed,**ranking(pred)))
            print(f'{fold.name} {name}: AP={fold_metrics[-1]["ap"]:.6f}',flush=True)
            del model
    csv(root/'fold_counts.csv',fold_counts); csv(root/'fold_metrics.csv',fold_metrics)
    summary,chosen=model_summary(fold_metrics); summary.to_csv(root/'model_comparison.csv',index=False)
    pooled={n:pd.concat(frames,ignore_index=True) for n,frames in predictions.items()}
    for n,p in pooled.items():
        if p.duplicated(['CERT','DECISION_DATE']).any(): raise IntegrityError('Duplicate out-of-fold observations')
        parquet(root/f'oof_{n}.parquet',p)
    csv(root/'pooled_metrics.csv',[dict(model=n,**ranking(p)) for n,p in pooled.items()])
    locked={}; sensitivity=[]; dev_operating=[]
    for n in (chosen,'prior'):
        p=pooled[n]
        for scenario,fp_mult,fn_mult in [('review_half',.5,1),('primary',1,1),('review_double',2,1),('miss_half',1,.5),('miss_double',1,2)]:
            result=threshold(p,fp*fp_mult,fn*fn_mult)
            op=operating(p,result['threshold'],fp*fp_mult,fn*fn_mult)
            if not np.isclose(result['loss'],op['loss_usd_thousands']): raise IntegrityError('Threshold loss mismatch')
            sensitivity.append(dict(model=n,scenario=scenario,c_fp=fp*fp_mult,c_fn=fn*fn_mult,candidates=result['candidate_thresholds'],**op))
            if scenario=='primary': locked[n]=result['threshold']; dev_operating.append(dict(model=n,**op))
    csv(root/'threshold_sensitivity.csv',sensitivity); csv(root/'development_operating.csv',dev_operating)
    # The native Spark demonstration is not eligible for model selection.
    if spark:
        from .spark_demo import run_demo
        train,valid=split_fold(dev,FOLDS[0]); write_json(root/'spark_demo.json',run_demo(train,valid))
    for n in (chosen,'prior'):
        model,elapsed=fit_model(dev,n)
        joblib.dump(model,root/f'final_{n}.joblib',compress=3)
        if n==chosen:
            info={'fit_rows':len(dev),'fit_seconds':elapsed,'first_decision':str(dev.DECISION_DATE.min().date()),'last_decision':str(dev.DECISION_DATE.max().date()),'features':list(SPECS[n][1]),'capital_zero_as_missing':SPECS[n][2]}
            write_json(root/'final_fit.json',info)
            clf=model.named_steps['classifier']; names=model[:-1].get_feature_names_out().tolist()
            if hasattr(clf,'booster_'):
                importance=clf.booster_.feature_importance(importance_type='gain'); kind='training_gain'
            elif hasattr(clf,'coef_'):
                importance=np.abs(clf.coef_[0]); kind='absolute_standardized_coefficient'
            else: importance=None; kind='not_available'
            if importance is not None: csv(root/'final_model_importance.csv',pd.DataFrame({'feature':names,'importance':importance,'kind':kind}).sort_values('importance',ascending=False))
    return {'selected_model':chosen,'thresholds':locked,'cost_policy':cost,'development_rows':len(dev),'selection_metric':'equal-weight mean fold average precision','spark_demo_completed':spark,'fold_count':len(FOLDS),'candidate_count':len(SPECS)}


def environment():
    return {'python':platform.python_version(),**{k:metadata.version(k) for k in ('numpy','pandas','scikit-learn','lightgbm','pyarrow','joblib','pyspark')}}


def develop(workspace):
    workspace=Path(workspace).resolve(); gold,gm,silver,identity=upstream(workspace); root=evidence_path(workspace,identity)
    if (root.parent/'test_attempt.json').exists(): raise IntegrityError('Historical test already attempted in this workspace; development is closed')
    if root.exists(): raise IntegrityError('Development directory already exists; inspect its state. Never remove a test attempt to rerun.')
    dev=read_population(gold,gm,'development'); check_population_counts(gold,dev,'development'); summary=develop_frames(dev,read_cost_records(silver),root)
    if upstream(workspace)[3]!=identity: raise IntegrityError('Upstream changed during development')
    manifest={'state':'SEALED','created_utc':utc_now(),'identity':identity,'protocol_sha256':protocol_digest(),'code_revision':code_revision(),'environment':environment(),'summary':summary,'files':inventory(root)}
    write_json(root/'development_manifest.json',manifest)
    return root,manifest


def verify_development(root,identity,*,require_code=True):
    m=read_json(root/'development_manifest.json')
    if m['state']!='SEALED' or m['identity']!=identity or (require_code and m['protocol_sha256']!=protocol_digest()): raise IntegrityError('Development seal mismatch')
    actual=inventory(root,('development_manifest.json','test_attempt.json'))
    actual={k:v for k,v in actual.items() if not k.startswith('test/')}
    if actual!=m['files']: raise IntegrityError('Development evidence changed')
    if not m['summary']['spark_demo_completed']: raise IntegrityError('Required Spark demonstration missing')
    return m


def verify_oof(root,m):
    """Reconstruct fold AP and locked thresholds from saved development scores only."""
    import pyarrow.parquet as pq
    reference=None; rows=[]; sealed=m['summary']; saved=pd.read_csv(root/'fold_metrics.csv',float_precision='round_trip')
    for name in SPECS:
        p=pq.ParquetFile(root/f'oof_{name}.parquet').read().to_pandas(); validate_rows(p,'development')
        if set(p.fold)!={f.name for f in FOLDS}: raise IntegrityError('Unexpected out-of-fold labels')
        keys=p[[*META,'fold']]
        if reference is None: reference=keys
        elif not keys.equals(reference): raise IntegrityError('Out-of-fold populations differ')
        for fold in FOLDS:
            v=p.loc[p.fold.eq(fold.name)]
            if v.empty or not pd.to_datetime(v.DECISION_DATE).between(pd.Timestamp(fold.start),pd.Timestamp(fold.end)).all(): raise IntegrityError('Out-of-fold date mismatch')
            stored=saved.loc[saved.model.eq(name)&saved.fold.eq(fold.name)]
            if len(stored)!=1: raise IntegrityError('Fold evidence missing')
            r=ranking(v)
            for k,value in r.items():
                if value is None:
                    if not pd.isna(stored.iloc[0][k]): raise IntegrityError('Fold metric mismatch')
                elif not np.isclose(value,stored.iloc[0][k],rtol=1e-12,atol=1e-14): raise IntegrityError(f'Fold metric mismatch: {name}/{fold.name}/{k}')
            rows.append(dict(model=name,fold=fold.name,fit_seconds=float(stored.iloc[0].fit_seconds),**r))
        if name in sealed['thresholds']:
            cost=sealed['cost_policy']; t=threshold(p,cost['c_fp'],cost['c_fn'])['threshold']
            if t!=sealed['thresholds'][name]: raise IntegrityError('Locked threshold mismatch')
    _,chosen=model_summary(rows)
    if chosen!=sealed['selected_model']: raise IntegrityError('Model selection mismatch')


def mark_attempt(root,seal_hash,identity=None):
    # Exclusive creation before any holdout Parquet decoding or predict_proba call.
    try:
        with (root/'test_attempt.json').open('x',encoding='utf-8') as f:
            json.dump({'state':'ATTEMPTED','started_utc':utc_now(),'development_manifest_sha256':seal_hash,'identity':identity},f); f.flush()
            import os
            os.fsync(f.fileno())
    except FileExistsError as e: raise IntegrityError('Historical test already attempted; use read-only verify, do not rescore') from e


def test_tables(predictions,summary,out,*,write=True):
    fp,fn=summary['cost_policy']['c_fp'],summary['cost_policy']['c_fn']; chosen=summary['selected_model']
    metrics=[]; annual=[]; calendar=[]
    for n,p in predictions.items():
        cutoff=summary['thresholds'][n]
        metrics.append(dict(model=n,**ranking(p),**operating(p,cutoff,fp,fn)))
        detail=event_detail(p,cutoff)
        cal=detail.loc[pd.to_datetime(detail.FAILURE_DATE).between(pd.Timestamp(TEST_START),pd.Timestamp(TEST_END))]
        calendar.append(dict(model=n,represented_events=len(cal),distinct_certs=int(cal.CERT.nunique()),detected_events=int(cal.detected.sum()),event_recall=float(cal.detected.mean()) if len(cal) else None,median_lead_days=float(cal.loc[cal.detected,'lead_days'].median()) if cal.detected.any() else None))
        for year,part in p.groupby(pd.to_datetime(p.DECISION_DATE).dt.year): annual.append(dict(model=n,year=int(year),**ranking(part),**operating(part,cutoff,fp,fn)))
        if write:
            detail.to_csv(out/f'events_{n}.csv',index=False); calibration(p).to_csv(out/f'calibration_{n}.csv',index=False)
    # No-alarm reference and all-alarm reference do not fit or score another model.
    p=predictions[chosen]
    refs=[dict(policy=label,**operating(p,t,fp,fn)) for label,t in [('no_alarm',2.),('all_alarm',0.)]]
    tables={'test_metrics.csv':metrics,'annual_metrics.csv':annual,'test_calendar_events.csv':calendar,'reference_policies.csv':refs}
    if write:
        for name,rows in tables.items(): csv(out/name,rows)
    else:
        for name,rows in tables.items():
            if (out/name).read_text()!=pd.DataFrame(rows).to_csv(index=False): raise IntegrityError(f'Test metric recomputation mismatch: {name}')
        for n,p in predictions.items():
            if (out/f'events_{n}.csv').read_text()!=event_detail(p,summary['thresholds'][n]).to_csv(index=False) or (out/f'calibration_{n}.csv').read_text()!=calibration(p).to_csv(index=False): raise IntegrityError('Test event/calibration recomputation mismatch')
    return metrics,calendar


def evaluate(workspace):
    import joblib
    workspace=Path(workspace).resolve(); gold,gm,silver,identity=upstream(workspace); root=evidence_path(workspace,identity)
    if (root.parent/'test_attempt.json').exists(): raise IntegrityError('Historical test already attempted; use read-only verify, do not rescore')
    m=verify_development(root,identity); seal=sha256(root/'development_manifest.json')
    if m['environment']!=environment(): raise IntegrityError('Runtime differs from the sealed fitting environment')
    verify_oof(root,m)
    mark_attempt(root.parent,seal,identity)
    # No holdout read before the marker. An exception leaves the attempt in place.
    test=read_population(gold,gm,'test'); check_population_counts(gold,test,'test'); out=root/'test'; out.mkdir(exist_ok=False)
    pred={}
    for n in (m['summary']['selected_model'],'prior'):
        model=joblib.load(root/f'final_{n}.joblib'); p=score_model(model,test,n); pred[n]=p
        parquet(out/f'predictions_{n}.parquet',p)
    metrics,cal=test_tables(pred,m['summary'],out)
    if any(row['distinct_certs']!=gm['summary']['test_distinct_failure_certs_in_calendar_period'] for row in cal) or any(row['events']!=gm['summary']['test_distinct_positive_events_through_final_horizon'] for row in metrics): raise IntegrityError('Test failure-count mismatch')
    if upstream(workspace)[3]!=identity: raise IntegrityError('Upstream changed during test')
    tm={'state':'VERIFIED','created_utc':utc_now(),'development_manifest_sha256':seal,'attempt_sha256':sha256(root.parent/'test_attempt.json'),'identity':identity,'files':inventory(out),'rows':len(test)}
    write_json(out/'manifest.json',tm)
    return root,tm


def verify(workspace):
    """Recompute metrics only from sealed predictions; never load fitted estimators."""
    import pyarrow.parquet as pq
    workspace=Path(workspace).resolve(); gold,gm,silver,identity=upstream(workspace); root=evidence_path(workspace,identity)
    m=verify_development(root,identity,require_code=False); out=root/'test'; tm=read_json(out/'manifest.json')
    if tm['state']!='VERIFIED' or tm['identity']!=identity or tm['development_manifest_sha256']!=sha256(root/'development_manifest.json') or tm['attempt_sha256']!=sha256(root.parent/'test_attempt.json') or tm['files']!=inventory(out,('manifest.json',)): raise IntegrityError('Invalid test evidence seal')
    attempt=read_json(root.parent/'test_attempt.json')
    if attempt['development_manifest_sha256']!=tm['development_manifest_sha256'] or attempt['identity']!=identity: raise IntegrityError('Attempt/selection mismatch')
    pred={}
    for n in (m['summary']['selected_model'],'prior'):
        p=pq.ParquetFile(out/f'predictions_{n}.parquet').read().to_pandas(); validate_rows(p,'test'); pred[n]=p
        if len(p)!=tm['rows']: raise IntegrityError('Prediction row mismatch')
    if not pred['prior'][META].equals(pred[m['summary']['selected_model']][META]): raise IntegrityError('Baseline/model population mismatch')
    verify_oof(root,m)
    test_tables(pred,m['summary'],out,write=False)
    return root,tm
