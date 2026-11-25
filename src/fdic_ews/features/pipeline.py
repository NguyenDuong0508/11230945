"""Quarter-bounded Gold construction, audit, and manifest-backed verification."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import platform
import importlib.metadata as metadata
import pandas as pd
from fdic_ews.contracts import (IntegrityError, quarter_ends, sha256, read_json, write_json, utc_now)
from fdic_ews.acquisition import parquet_commit, run_id, code_revision
from .policy import (FIRST,LAST,LAG,MAX_AGE,TRAIN_END,VALID_START,DEV_END,TEST_START,TEST_END,MIN_TEST_FAILURES,SEED,split_for,event_index,risk_status,outcome,assert_split)
from fdic_ews.quality.policy import shift_quarter, available_at
from .financial import FEATURES,DETAILS,INPUT_COLUMNS,CAPITAL,report_features,latest_reports,decision_features
from .preprocessing import fit_training

KEYS=('CERT','DECISION_DATE','REPORT_DATE','AVAILABLE_AT','HORIZON_END','SPLIT')
OUTCOMES=('Y_FAIL_4Q','LABEL_REASON','FAILURE_DATE')
COLUMNS=(*KEYS,*FEATURES,*OUTCOMES)


def gold_schema():
    import pyarrow as pa
    fields=[pa.field('CERT',pa.int64(),False)]
    fields.extend(pa.field(k,pa.date32(),False) for k in ('DECISION_DATE','REPORT_DATE','AVAILABLE_AT','HORIZON_END'))
    fields.append(pa.field('SPLIT',pa.string(),False))
    fields.extend(pa.field(k,pa.float64()) for k in FEATURES)
    fields.extend([pa.field('Y_FAIL_4Q',pa.int8()),pa.field('LABEL_REASON',pa.string(),False),pa.field('FAILURE_DATE',pa.date32())])
    return pa.schema(fields)


def to_table(frame):
    import pyarrow as pa
    return pa.Table.from_pandas(frame.loc[:,list(COLUMNS)],schema=gold_schema(),preserve_index=False,safe=True).replace_schema_metadata(None)


def relative(q): return f'panel/year={q.year}/quarter={q.month//3}/part.parquet'


def build_quarter(q, frames, events, reports):
    """One decision date; peers use only the risk set before outcomes are attached."""
    split=split_for(q); h=shift_quarter(q,4)
    candidates=latest_reports(q,frames)
    audit={'decision_date':str(q),'split':split,'candidate_rows':0,'risk_rows':0,'prior_exit':0,'prior_exit_conflict':0,'report_unavailable_or_stale':0,'labelled':0,'positive':0,'negative':0,'censored':0,'lag120_candidate_rows':None,'candidate_exit_zero_extra_negatives':None}
    dev=split in ('train','validation')
    if dev:
        alt=latest_reports(q,frames,lag=120)
        audit['lag120_candidate_rows']=0 if alt is None else len(alt)
        audit['candidate_exit_zero_extra_negatives']=0
    if candidates is None:
        return pd.DataFrame(columns=COLUMNS),audit,Counter()
    audit['candidate_rows']=len(candidates)
    # Explicitly use only information dated no later than the decision.
    status=[risk_status(q,r.date(),events.get(int(c),())) for c,r in zip(candidates.CERT,candidates.REPORT_DATE)]
    for k,v in Counter(status).items():
        if k!='at_risk': audit[k]=v
    z=decision_features(candidates.loc[[s=='at_risk' for s in status]].reset_index(drop=True),q)
    z['DECISION_DATE']=q; z['HORIZON_END']=h; z['SPLIT']=split
    quarters=[shift_quarter(q,i) for i in range(1,5)]
    sets=[reports.get(d,set()) for d in quarters]
    labels=[]; reasons=[]; dates=[]
    for c in z.CERT:
        c=int(c); ev=events.get(c,())
        # Reports after the horizon are never consulted.
        if not ev:
            result=(0,'observed_at_horizon',None) if c in sets[-1] else (None,'incomplete_followup',None)
        else:
            result=outcome(q,ev,[d for d,s in zip(quarters,sets) if c in s])
        y,why,day=result; labels.append(y); reasons.append(why); dates.append(day)
    z['Y_FAIL_4Q']=pd.array(labels,dtype='Int8'); z['LABEL_REASON']=reasons; z['FAILURE_DATE']=dates
    z=z.loc[:,list(COLUMNS)].sort_values('CERT').reset_index(drop=True)
    audit.update(risk_rows=len(z),labelled=int(z.Y_FAIL_4Q.notna().sum()),positive=int(z.Y_FAIL_4Q.eq(1).sum()),negative=int(z.Y_FAIL_4Q.eq(0).sum()),censored=int(z.Y_FAIL_4Q.isna().sum()))
    counts=Counter(reasons)
    if dev: audit['candidate_exit_zero_extra_negatives']=counts['unconfirmed_competing_exit']
    if audit['candidate_rows']!=audit['risk_rows']+sum(audit[k] for k in ('prior_exit','prior_exit_conflict','report_unavailable_or_stale')) or audit['risk_rows']!=audit['labelled']+audit['censored'] or audit['labelled']!=audit['positive']+audit['negative']:
        raise IntegrityError('Risk and outcome accounting mismatch')
    return z,audit,counts


def iter_gold(silver,manifest):
    import pyarrow.parquet as pq
    assert_split()
    entries=manifest['panel']; dates=quarter_ends()
    if set(entries)!=set(map(str,dates)): raise IntegrityError('Incomplete Silver calendar')
    reports={}
    for d in dates:
        e=entries[str(d)]; path=silver/e['path']
        if sha256(path)!=e['sha256']: raise IntegrityError('Silver partition changed')
        a=pq.ParquetFile(path).read(columns=['CERT'])['CERT'].to_pylist()
        if len(a)!=e['rows'] or len(set(a))!=len(a): raise IntegrityError('Invalid source keys')
        reports[d]=set(a)
    ev=event_index(pq.ParquetFile(silver/'event_ledger.parquet').read().to_pylist())
    empty=pd.DataFrame(columns=INPUT_COLUMNS)
    for k in INPUT_COLUMNS:
        if k not in ('REPDTE','STNAME','ASSUMED_AVAILABLE_AT'): empty[k]=pd.Series(dtype='float64')
    @lru_cache(maxsize=8)
    def load(d):
        if d<FIRST: return empty
        e=entries[str(d)]; f=pq.ParquetFile(silver/e['path']).read(columns=list(INPUT_COLUMNS)).to_pandas()
        if not pd.to_datetime(f.REPDTE).eq(pd.Timestamp(d)).all() or not pd.to_datetime(f.ASSUMED_AVAILABLE_AT).eq(pd.Timestamp(available_at(d,LAG))).all(): raise IntegrityError('Invalid source dates')
        return f
    @lru_cache(maxsize=3)
    def features(d):
        return report_features(load(d),load(shift_quarter(d,-1)),load(shift_quarter(d,-4)))
    for q in quarter_ends(FIRST,LAST):
        frames=[(r,features(r)) for i in (1,2) if (r:=shift_quarter(q,-i))>=FIRST]
        yield q,*build_quarter(q,frames,ev,reports)


def validate_partition(q, frame):
    if tuple(frame.columns)!=COLUMNS or frame.CERT.duplicated().any(): raise IntegrityError('Invalid Gold schema/keys')
    if frame.empty: return
    if not pd.to_datetime(frame.DECISION_DATE).eq(pd.Timestamp(q)).all() or set(frame.SPLIT)!={split_for(q)}: raise IntegrityError('Decision-date split mismatch')
    r=pd.to_datetime(frame.REPORT_DATE); a=pd.to_datetime(frame.AVAILABLE_AT); h=pd.to_datetime(frame.HORIZON_END)
    if not (a.eq(r+pd.Timedelta(days=LAG)) & a.le(pd.Timestamp(q)) & r.lt(pd.Timestamp(q)) & h.eq(pd.Timestamp(shift_quarter(q,4)))).all(): raise IntegrityError('Invalid information timing')
    if not frame.REPORT_AGE_QUARTERS.between(1,MAX_AGE).all(): raise IntegrityError('Stale report')
    y=frame.Y_FAIL_4Q; f=pd.to_datetime(frame.FAILURE_DATE)
    pos=y.eq(1).fillna(False)
    if not set(y.dropna().astype(int))<={0,1} or not f.notna().eq(pos).all() or not ((f[pos]>pd.Timestamp(q)) & (f[pos]<=h[pos])).all(): raise IntegrityError('Invalid label interval')
    import numpy as np
    if np.isinf(frame[list(FEATURES)].to_numpy(dtype=float)).any(): raise IntegrityError('Nonfinite predictor')


def accounting(audits, reason_rows, positive_events):
    df=pd.DataFrame(audits); cols=['candidate_rows','risk_rows','prior_exit','prior_exit_conflict','report_unavailable_or_stale','labelled','positive','negative','censored']
    split=df.groupby('split')[cols].sum().reset_index()
    events={s:set(v) for s,v in positive_events.items()}
    split['distinct_positive_events']=[len(events.get(s,set())) for s in split.split]
    split['distinct_positive_certs']=[len({c for c,d in events.get(s,set())}) for s in split.split]
    eligible_test={(c,d) for c,d in events.get('test',set()) if TEST_START<=d<=TEST_END}
    actual_test_certs=len({c for c,d in eligible_test})
    if actual_test_certs<MIN_TEST_FAILURES: raise IntegrityError(f'Fixed holdout has only {actual_test_certs} distinct represented failure CERTs within its calendar period; minimum {MIN_TEST_FAILURES}. Do not automatically move the split.')
    for s in ('train','validation','test'):
        r=split.loc[split.split.eq(s)]
        if r.empty or not int(r.positive.iloc[0]) or not int(r.negative.iloc[0]): raise IntegrityError(f'Both classes required in {s}')
    return split,{'test_distinct_failure_certs_in_calendar_period':actual_test_certs,'test_distinct_failure_events_in_calendar_period':len(eligible_test),'test_distinct_positive_events_through_final_horizon':len(events.get('test',set())), 'risk_rows':int(df.risk_rows.sum()),'labelled_rows':int(df.labelled.sum()),'decision_quarters':len(df),'feature_count':len(FEATURES)}


def training_frame(root, entries):
    import pyarrow.parquet as pq
    frames=[]
    for d,e in entries.items():
        if split_for(pd.Timestamp(d).date())!='train': continue
        f=pq.ParquetFile(root/e['path']).read(columns=['DECISION_DATE','SPLIT','Y_FAIL_4Q',*FEATURES]).to_pandas()
        frames.append(f.loc[f.Y_FAIL_4Q.notna()])
    if not frames: raise IntegrityError('No training partitions')
    return pd.concat(frames,ignore_index=True)


def diagnostics(root, entries, *, save=False):
    import joblib
    train=training_frame(root,entries)
    result={}
    for name,alt in [('primary',False),('capital_zero_as_missing',True)]:
        pipe,info=fit_training(train,zero_as_missing=alt)
        result[name]=info
        if save: joblib.dump(pipe,root/(name+'_preprocessor.joblib'),compress=3)
    # Development-only diagnostics: train, not holdout.
    x=train[list(FEATURES)]; missing=pd.DataFrame({'feature':FEATURES,'train_rows':len(x),'missing':x.isna().sum().to_numpy(),'missing_pct':x.isna().mean().to_numpy()*100,'distinct_nonnull':x.nunique().to_numpy()})
    missing['missing_if_capital_zeros_masked']=missing['missing']
    for k in CAPITAL: missing.loc[missing.feature.eq(k),'missing_if_capital_zeros_masked']=int(x[k].isna().sum()+x[k].eq(0).sum())
    return result,missing


def csv(path,frame): frame.to_csv(path,index=False)


def process(silver,manifest,root,*,write):
    import pyarrow.parquet as pq
    audits=[]; reason_rows=[]; events=defaultdict(set); entries={}
    for q,frame,audit,reasons in iter_gold(silver,manifest):
        validate_partition(q,frame); target=root/relative(q); table=to_table(frame)
        if write: parquet_commit(table,target)
        else:
            actual=pq.ParquetFile(target).read()
            if actual.schema!=gold_schema() or not actual.equals(table): raise IntegrityError(f'Gold recomputation differs: {q}')
        entries[str(q)]={'path':relative(q),'rows':len(frame),'sha256':sha256(target)}
        audits.append(audit)
        reason_rows.extend({'decision_date':str(q),'split':audit['split'],'reason':k,'rows':v} for k,v in sorted(reasons.items()))
        if not frame.empty:
            positive=frame.loc[frame.Y_FAIL_4Q.eq(1).fillna(False),['CERT','FAILURE_DATE']]
            events[audit['split']].update((int(c),d) for c,d in positive.itertuples(index=False,name=None))
        if write and (len(audits)%20==0 or q==LAST): print(f'{len(audits)} decision quarters verified: {q}',flush=True)
    split,summary=accounting(audits,reason_rows,events)
    # Count event coverage without evaluating any holdout model or feature distribution.
    ledger=pq.ParquetFile(silver/'event_ledger.parquet').read().to_pylist()
    linked={(int(r['CERT']),r['event_date']) for r in ledger if r['event_type']=='failure' and r['linked'] and r['CERT'] is not None and r['date_status']=='valid' and r['event_date'] is not None and TEST_START<=r['event_date']<=TEST_END}
    represented={(c,d) for c,d in events.get('test',set()) if TEST_START<=d<=TEST_END}
    if not represented<=linked: raise IntegrityError('Positive event absent from linked ledger')
    summary.update(test_linked_failure_events_in_calendar_period=len(linked),test_unrepresented_linked_failure_events_in_calendar_period=len(linked-represented))
    results,missing=diagnostics(root,entries,save=write)
    dictionary=pd.DataFrame([{'feature':k,'group':v[0],'formula':v[1],'rationale':v[2]} for k,v in DETAILS.items()])
    tables={'quarter_audit.csv':pd.DataFrame(audits),'label_reasons.csv':pd.DataFrame(reason_rows),'split_summary.csv':split,'training_feature_quality.csv':missing,'feature_dictionary.csv':dictionary}
    if write:
        for name,f in tables.items(): csv(root/name,f)
        write_json(root/'preprocessing.json',results); write_json(root/'summary.json',summary)
    else:
        for name,f in tables.items():
            if (root/name).read_text()!=f.to_csv(index=False): raise IntegrityError(f'Audit mismatch: {name}')
        if read_json(root/'preprocessing.json')!=results or read_json(root/'summary.json')!=summary: raise IntegrityError('Training diagnostics mismatch')
    return entries,summary


def build(silver,manifest,root):
    root.mkdir(parents=True,exist_ok=False)
    entries,summary=process(silver,manifest,root,write=True)
    m={'state':'VALIDATING','created_utc':utc_now(),'code_revision':code_revision(),'source_manifest_sha256':manifest['source_manifest_sha256'],'silver_manifest_sha256':sha256(silver/'manifest.json'),'policy':{'lag_days':LAG,'max_report_age_quarters':MAX_AGE,'train_end':str(TRAIN_END),'validation_start':str(VALID_START),'development_end':str(DEV_END),'test_start':str(TEST_START),'test_end':str(TEST_END),'last_decision':str(LAST),'minimum_test_failure_certs':MIN_TEST_FAILURES,'seed':SEED,'historical_vintages_verified':False},'environment':{'python':platform.python_version(),**{k:metadata.version(k) for k in ('pandas','numpy','pyarrow','scikit-learn','joblib')}},'panel':entries,'summary':summary,'files':{p.relative_to(root).as_posix():sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    write_json(root/'manifest.json',m)
    verify_run(silver,manifest,root,allow_validating=True)
    m['state']='VERIFIED'
    write_json(root/'manifest.json',m)
    return m


def verify_run(silver,manifest,root,*,allow_validating=False):
    m=read_json(root/'manifest.json')
    if m['state'] not in ({'VERIFIED','VALIDATING'} if allow_validating else {'VERIFIED'}) or m['silver_manifest_sha256']!=sha256(silver/'manifest.json') or m['source_manifest_sha256']!=manifest['source_manifest_sha256']: raise IntegrityError('Gold lineage mismatch')
    files={p.relative_to(root).as_posix():sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'manifest.json'}
    if files!=m['files']: raise IntegrityError('Gold artifact integrity failed')
    entries,summary=process(silver,manifest,root,write=False)
    if entries!=m['panel'] or summary!=m['summary']: raise IntegrityError('Gold reconciliation failed')
    return m


def run(workspace):
    from fdic_ews.quality.panel import resolve_panel
    workspace=Path(workspace).resolve()
    if (workspace/'pending_source.json').exists(): raise IntegrityError('Pending source acquisition')
    silver,m=resolve_panel(workspace); digest=sha256(silver/'manifest.json')
    root=workspace/'gold'/run_id(); result=build(silver,m,root)
    if read_json(workspace/'active_panel.json')['manifest_sha256']!=digest: raise IntegrityError('Active Silver changed')
    write_json(workspace/'active_features.json',{'path':root.relative_to(workspace).as_posix(),'manifest_sha256':sha256(root/'manifest.json')})
    return root,result


def resolve_features(workspace):
    from fdic_ews.quality.panel import resolve_panel
    workspace=Path(workspace).resolve(); p=read_json(workspace/'active_features.json'); root=(workspace/p['path']).resolve()
    if root.parent!=workspace/'gold' or sha256(root/'manifest.json')!=p['manifest_sha256']: raise IntegrityError('Invalid Gold pointer')
    silver,m=resolve_panel(workspace)
    return root,verify_run(silver,m,root)
