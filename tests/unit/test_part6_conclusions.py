"""Synthetic artifact contracts; no downloads, Parquet decoding or estimator calls."""
import csv
from pathlib import Path

import pytest

from fdic_ews.conclusions import collect, operating_checks, storage_findings, processing_findings
from fdic_ews.conclusions import FORMATS, STORAGE_WORKLOADS, PROCESS_WORKLOADS
from fdic_ews.contracts import IntegrityError, quarter_ends, sha256, write_json, read_json


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def inventory(root):
    return {p.relative_to(root).as_posix():sha256(p) for p in root.rglob('*') if p.is_file()}


def op(tp, fp, fn, tn, events, detected, threshold=.03):
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn, alerts=tp+fp,
                alert_rate=(tp+fp)/(tp+fp+fn+tn), precision=tp/(tp+fp) if tp+fp else '',
                recall=tp/(tp+fn), fpr=fp/(fp+tn), events=events, detected_events=detected,
                missed_events=events-detected, event_recall=detected/events,
                median_lead_days=100 if detected else '', lead_at_least_90_days=detected,
                loss_usd_thousands=fp*2+(events-detected)*10)


def storage_fixture():
    sizes = [dict(format=f,bytes=1000 if f=='csv' else 500,reduction_vs_csv_pct=0 if f=='csv' else 50) for f in FORMATS]
    summary = [dict(format=f,workload=w,median_seconds=2 if f=='csv' else 1,rows=1700,
                    candidate_fragments={'year_2020':4,'quarter_2020q1':1}.get(w,170) if f.startswith('partitioned') else '',
                    total_fragments=170 if f.startswith('partitioned') else '',median_peak_rss_mib=100)
               for f in FORMATS for w in STORAGE_WORKLOADS]
    summary += [dict(format=e,workload='csv_projection',median_seconds=1,rows=1700,candidate_fragments='',total_fragments='',median_peak_rss_mib=100) for e in ('pandas','polars')]
    return sizes,summary


def processing_fixture():
    jobs = [(s,e,w,2) for s in (1,5,10) for e in ('pandas','polars') for w in PROCESS_WORKLOADS]
    jobs += [(1,'pyspark',w,2) for w in PROCESS_WORKLOADS]+[(1,'polars',w,1) for w in PROCESS_WORKLOADS]
    return [dict(scale=str(s),engine=e,workload=w,threads=str(t),input_rows=s*(100 if w=='quarter_summary' else 1700),median_seconds=s*(2 if e=='pandas' else 1),median_peak_rss_mib=100*s) for s,e,w,t in jobs]


@pytest.fixture
def workspace(tmp_path):
    ws=tmp_path
    def directory(folder):
        p=ws/folder/'run'; p.mkdir(parents=True); return p
    def seal(root, name, pointer_name, m, key='manifest_sha256'):
        write_json(root/name,m); digest=sha256(root/name)
        p=dict(path=root.relative_to(ws).as_posix(), **{key:digest})
        if pointer_name=='active_storage.json': p['source_manifest_sha256']=m['source_manifest_sha256']
        write_json(ws/pointer_name,p); return digest
    bronze=directory('bronze')
    def artifact(name, rows):
        target=bronze/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(name)
        return dict(path=name,sha256=sha256(target),quality=dict(rows=rows),api=dict(api_total=rows))
    entries={str(q):artifact(f'financials/{q}.parquet',10) for q in quarter_ends()}
    references={k:artifact(k+'.parquet',n) for k,n in [('failures',80),('institutions',100)]}
    b=seal(bronze,'manifest.json','active_source.json',dict(state='VERIFIED',created_utc='2026-01-01',sealed_utc='2026-01-02',financials=entries,references=references))
    write_json(bronze/'verification.json',dict(status='PASS',source_state='VERIFIED',manifest_sha256=b,row_counts=dict(financials=1700,failures=80,institutions=100),quarter_count=170,distinct_financial_certs=100))
    storage=directory('storage_evidence'); sizes,summary=storage_fixture()
    save_csv(storage/'sizes.csv',sizes); save_csv(storage/'summary.csv',summary)
    seal(storage,'evidence.json','active_storage.json',dict(status='PASS',source_manifest_sha256=b,files=inventory(storage),measured_trials=110),'evidence_sha256')
    silver=directory('silver'); ss=dict(source_rows=1700,panel_rows=1700,rows_removed=0,bronze_failure_rows=80,bronze_institution_rows=100,asset_to_null=0,gap_rows=2,event_ledger_rows=180,distinct_certs=100,event_diagnostics={"assistance_records":5},horizon_case={"invariant":True})
    write_json(silver/'summary.json',ss)
    for name,rows in [('quality_flags.csv',[dict(rule='REPORTING_GAP',flagged=2)]),('field_summary.csv',[dict(field='ASSET',missing_pct=0,zero_pct=0)]),('terminal_summary.csv',[dict(audit_class='unknown',institutions=3)]),('availability_sensitivity.csv',[dict(report_quarter='2000-03-31',lag_days=90,decision_quarter='2000-06-30',available_rows=10,report_rows=10)])]: save_csv(silver/name,rows)
    s=seal(silver,'manifest.json','active_panel.json',dict(state='VERIFIED',source_manifest_sha256=b,summary=ss,files=inventory(silver)))
    processing=directory('processing_evidence'); save_csv(processing/'summary.csv',processing_fixture()); save_csv(processing/'parallelism.csv',[dict(workload='peer_join',speedup=1.5)]); write_json(processing/'environment.json',dict(ram_mib=10000))
    seal(processing,'manifest.json','active_processing.json',dict(state='VERIFIED',source_manifest_sha256=b,silver_manifest_sha256=s,files=inventory(processing),measured_trials=72))
    gold=directory('gold'); gs=dict(risk_rows=490,labelled_rows=460,test_distinct_failure_certs_in_calendar_period=50,test_distinct_failure_events_in_calendar_period=50,test_distinct_positive_events_through_final_horizon=56)
    write_json(gold/'summary.json',gs)
    save_csv(gold/'split_summary.csv',[dict(split=k,labelled=n,positive=pos,negative=n-pos,risk_rows=n+10,censored=10) for k,n,pos in [('train',150,20),('validation',100,10),('test',210,70)]])
    save_csv(gold/'label_reasons.csv',[dict(reason='incomplete_followup',rows=30)])
    g=seal(gold,'manifest.json','active_features.json',dict(state='VERIFIED',source_manifest_sha256=b,silver_manifest_sha256=s,summary=gs,files=inventory(gold),policy=dict(lag_days=90)))
    identity=dict(bronze=b,silver=s,gold=g); dev=ws/'modelling'/g; dev.mkdir(parents=True); test=dev/'test'
    cost=dict(c_fp=2,c_fn=10,unit='USD thousands',represented_development_events=20,missed_cost_source='fixture',eligible_dated_cost_events=0)
    write_json(dev/'cost_policy.json',cost); save_csv(dev/'model_comparison.csv',[dict(model='chosen',mean_ap=.3),dict(model='prior',mean_ap=.1)])
    save_csv(dev/'fold_counts.csv',[dict(fold='F1',validation_rows=100,validation_positive=10)])
    save_csv(dev/'development_operating.csv',[dict(model='chosen',**op(8,5,2,85,8,7)),dict(model='prior',**op(0,0,10,90,8,0,2))])
    write_json(dev/'spark_demo.json',dict(validation_sampled=False,rows=100,positives=10,ap=.3))
    dm=dict(state='SEALED',identity=identity,summary=dict(selected_model='chosen',cost_policy=cost,thresholds=dict(chosen=.03,prior=2),candidate_count=2,fold_count=1,development_rows=250,spark_demo_completed=True),files=inventory(dev))
    write_json(dev/'development_manifest.json',dm); dh=sha256(dev/'development_manifest.json')
    attempt=dev.parent/'test_attempt.json'; write_json(attempt,dict(state='ATTEMPTED',identity=identity,development_manifest_sha256=dh))
    test.mkdir(); metrics=[]; annual=[]; calendar=[]
    for model,per_year,overall,detected in [('chosen',op(8,2,2,18,8,7),op(56,14,14,126,56,49),45),('prior',op(0,0,10,20,8,0,2),op(0,0,70,140,56,0,2),0)]:
        metrics.append(dict(model=model,rows=210,positives=70,prevalence=1/3,ap=.5 if model=='chosen' else 1/3,roc_auc=.7,brier=.2,log_loss=.4,recall_top5pct_by_quarter=.2,**overall))
        annual.extend(dict(model=model,year=y,rows=30,positives=10,prevalence=1/3,ap=.5,roc_auc=.7,**per_year) for y in range(2008,2015))
        calendar.append(dict(model=model,represented_events=50,distinct_certs=50,detected_events=detected,event_recall=detected/50,median_lead_days=100 if detected else ''))
    save_csv(test/'test_metrics.csv',metrics); save_csv(test/'annual_metrics.csv',annual); save_csv(test/'test_calendar_events.csv',calendar)
    save_csv(test/'reference_policies.csv',[dict(policy='no_alarm',**op(0,0,70,140,56,0,2)),dict(policy='all_alarm',**op(70,140,0,0,56,56,0))])
    write_json(test/'manifest.json',dict(state='VERIFIED',identity=identity,development_manifest_sha256=dh,attempt_sha256=sha256(attempt),rows=210,files=inventory(test)))
    return ws


def test_end_to_end_read_only(workspace):
    before=inventory(workspace); result=collect(workspace)
    assert result['status']=='PASS'
    assert result['source']['quarters']==170
    assert result['modelling']['derived']['calendar_event_recall_pct']==90
    assert result['modelling']['derived']['loss_reduction_vs_no_alarm_pct']==pytest.approx(82.5)
    assert inventory(workspace)==before


def test_corrupt_artifact_stops(workspace):
    (workspace/'silver/run/quality_flags.csv').write_text('corrupt')
    with pytest.raises(IntegrityError,match='Checksum'): collect(workspace)


def test_wrong_lineage_even_if_resealed(workspace):
    path=workspace/'processing_evidence/run/manifest.json'; m=read_json(path); m['silver_manifest_sha256']='wrong'; write_json(path,m)
    pointer=workspace/'active_processing.json'; p=read_json(pointer); p['manifest_sha256']=sha256(path); write_json(pointer,p)
    with pytest.raises(IntegrityError,match='lineage'): collect(workspace)


def test_incomplete_test_stops(workspace):
    next((workspace/'modelling').glob('*/test/manifest.json')).unlink()
    with pytest.raises(FileNotFoundError): collect(workspace)


def test_tampered_attempt_stops(workspace):
    path=workspace/'modelling/test_attempt.json'; p=read_json(path); p['other']='changed'; write_json(path,p)
    with pytest.raises(IntegrityError,match='seal/attempt'): collect(workspace)


def test_pending_acquisition_stops(workspace):
    write_json(workspace/'pending_source.json',{})
    with pytest.raises(IntegrityError,match='Pending'): collect(workspace)


def test_storage_rejects_missing_grid():
    sizes,summary=storage_fixture()
    with pytest.raises(IntegrityError,match='grid'): storage_findings(sizes,summary[:-1])


def test_storage_rejects_bad_pruning():
    sizes,summary=storage_fixture(); next(r for r in summary if r['format']=='partitioned_zstd' and r['workload']=='year_2020')['candidate_fragments']=170
    with pytest.raises(IntegrityError,match='pruning'): storage_findings(sizes,summary)


def test_processing_rejects_wrong_scale():
    rows=processing_fixture(); next(r for r in rows if r['scale']=='10')['input_rows']=1
    with pytest.raises(IntegrityError,match='input count'): processing_findings(rows,1700)


def test_operating_rejects_row_based_loss():
    row=op(8,2,2,18,8,7); row['loss_usd_thousands']=24
    with pytest.raises(IntegrityError,match='Economic loss'): operating_checks(row,2,10)
