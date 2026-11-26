"""Read-only synthesis of verified acquisition, panel, benchmark and model evidence."""
from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from .contracts import IntegrityError, quarter_ends, read_json, sha256

FORMATS = ('csv', 'flat_snappy', 'flat_zstd', 'partitioned_snappy', 'partitioned_zstd')
STORAGE_WORKLOADS = ('full', 'projection', 'year_2020', 'quarter_2020q1')
PROCESS_WORKLOADS = ('quarter_summary', 'adjacent_growth', 'peer_join')


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def number(value):
    result = float(value)
    require(math.isfinite(result), 'Nonfinite evidence value')
    return result


def count(value):
    n = number(value)
    require(n >= 0 and n.is_integer(), 'Invalid count')
    return int(n)


def close(a, b, message):
    require(math.isclose(number(a), number(b), rel_tol=1e-9, abs_tol=1e-9), message)


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    require(bool(rows), f'Empty evidence: {path.name}')
    return rows


def index(rows, *keys):
    result = {tuple(r[k] for k in keys): r for r in rows}
    require(len(result) == len(rows), f'Duplicate evidence keys: {keys}')
    return result


def check_files(root, files, *, exclude=(), exact=True):
    require(bool(files), f'Empty inventory: {root.name}')
    if exact:
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} - set(exclude)
        require(actual == set(files), f'Inventory mismatch: {root.name}')
    for name, digest in files.items():
        p = root / name
        require(p.resolve().is_relative_to(root.resolve()) and p.is_file() and not p.is_symlink(),
                f'Unsafe or absent artifact: {name}')
        require(sha256(p) == digest, f'Checksum mismatch: {name}')


def pointer(workspace, name, folder, manifest='manifest.json', digest_key='manifest_sha256'):
    p = read_json(workspace / name)
    root = (workspace / p['path']).resolve()
    require(root.parent == workspace / folder, f'Invalid {name} path')
    digest = sha256(root / manifest)
    require(digest == p[digest_key], f'Invalid {name} digest')
    return root, read_json(root / manifest), digest


def storage_findings(sizes, summary):
    size = index(sizes, 'format')
    require(set(size) == {(f,) for f in FORMATS}, 'Incomplete storage formats')
    base = count(size[('csv',)]['bytes'])
    require(base > 0, 'Empty CSV')
    out = []
    for f in FORMATS:
        n = count(size[(f,)]['bytes'])
        require(n > 0, 'Empty representation')
        reduction = (1 - n / base) * 100
        close(size[(f,)]['reduction_vs_csv_pct'], reduction, 'Storage size reduction mismatch')
        out.append(dict(format=f, mib=n / 2**20, reduction_pct=reduction))
    data = index(summary, 'format', 'workload')
    expected = {(f,w) for f in FORMATS for w in STORAGE_WORKLOADS} | {(e,'csv_projection') for e in ('pandas','polars')}
    require(set(data) == expected, 'Incomplete storage workload grid')
    queries = []
    for w in STORAGE_WORKLOADS:
        reference = number(data['csv',w]['median_seconds'])
        require(reference > 0, 'Nonpositive CSV latency')
        for f in FORMATS:
            r = data[f,w]; t = number(r['median_seconds'])
            require(t > 0 and count(r['rows']) == count(data['csv',w]['rows']), 'Storage workload mismatch')
            fragments = None if not r['candidate_fragments'] else count(r['candidate_fragments'])
            if f.startswith('partitioned'):
                require(fragments == {'year_2020':4,'quarter_2020q1':1}.get(w,170)
                        and count(r['total_fragments']) == 170, 'Partition pruning mismatch')
            queries.append(dict(format=f, workload=w, seconds=t, speedup_vs_csv=reference/t,
                                peak_rss_mib=number(r['median_peak_rss_mib']), candidate_fragments=fragments))
    return out, queries


def processing_findings(summary, panel_rows):
    data = index(summary, 'scale','engine','workload','threads')
    expected = {(str(s),e,w,'2') for s in (1,5,10) for e in ('pandas','polars') for w in PROCESS_WORKLOADS}
    expected |= {('1','pyspark',w,'2') for w in PROCESS_WORKLOADS} | {('1','polars',w,'1') for w in PROCESS_WORKLOADS}
    require(set(data) == expected, 'Incomplete processing workload grid')
    result = []
    for w in PROCESS_WORKLOADS:
        base_rows = count(data['1','pandas',w,'2']['input_rows'])
        require(0 < base_rows <= panel_rows and (w == 'quarter_summary' or base_rows == panel_rows), 'Invalid logical input count')
        for s in (1,5,10):
            engines = ('pandas','polars','pyspark') if s == 1 else ('pandas','polars')
            reference = number(data[str(s),'pandas',w,'2']['median_seconds'])
            require(reference > 0, 'Nonpositive pandas latency')
            for e in engines:
                r = data[str(s),e,w,'2']; t = number(r['median_seconds'])
                require(t > 0 and count(r['input_rows']) == s*base_rows, 'Processing input count mismatch')
                one = number(data['1',e,w,'2']['median_seconds'])
                require(one > 0, 'Nonpositive base latency')
                result.append(dict(workload=w, scale=s, engine=e, input_rows=count(r['input_rows']),
                                   seconds=t, speedup_vs_pandas=reference/t, time_ratio_vs_1x=t/one,
                                   peak_rss_mib=number(r['median_peak_rss_mib'])))
    return result


def operating_checks(r, c_fp, c_fn, expected_rows=None):
    tp, fp, fn, tn, events, hit, missed = (count(r[k]) for k in ('tp','fp','fn','tn','events','detected_events','missed_events'))
    rows = tp+fp+fn+tn
    require(rows > 0 and events > 0 and events == hit+missed, 'Operating count reconciliation failed')
    if expected_rows is not None:
        require(rows == expected_rows, 'Operating population mismatch')
    require(count(r['alerts']) == tp+fp and count(r['lead_at_least_90_days']) <= hit, 'Alarm/lead count mismatch')
    close(r['alert_rate'], (tp+fp)/rows, 'Alert rate mismatch')
    close(r['event_recall'], hit/events, 'Event recall mismatch')
    close(r['loss_usd_thousands'], fp*c_fp+missed*c_fn, 'Economic loss mismatch')
    if tp+fp:
        close(r['precision'], tp/(tp+fp), 'Precision mismatch')
    else:
        require(r['precision'] == '', 'No-alarm precision must be undefined')
    if tp+fn:
        close(r['recall'], tp/(tp+fn), 'Row recall mismatch')
    if fp+tn:
        close(r['fpr'], fp/(fp+tn), 'False-positive rate mismatch')
    return rows


def model_findings(dev, test, gm, dm, tm, splits):
    s = dm['summary']; chosen = s['selected_model']; cost = s['cost_policy']
    c_fp, c_fn = number(cost['c_fp']), number(cost['c_fn'])
    require(c_fp > 0 and c_fn > 0 and cost['unit'] == 'USD thousands', 'Invalid cost policy')
    require(read_json(dev/'cost_policy.json') == cost, 'Cost seal mismatch')
    require(set(s['thresholds']) == {chosen,'prior'} and chosen != 'prior', 'Invalid frozen policies')
    cmp = csv_rows(dev/'model_comparison.csv'); by_model = index(cmp,'model')
    require(len(cmp) == count(s['candidate_count']) and chosen == max((r for r in cmp if r['model']!='prior'), key=lambda r:number(r['mean_ap']))['model'], 'Selection mismatch')
    folds = csv_rows(dev/'fold_counts.csv')
    require(len(index(folds,'fold')) == count(s['fold_count']), 'Fold count mismatch')
    oof_rows = sum(count(r['validation_rows']) for r in folds)
    dev_ops = index(csv_rows(dev/'development_operating.csv'),'model')
    metrics = index(csv_rows(test/'test_metrics.csv'),'model')
    cal = index(csv_rows(test/'test_calendar_events.csv'),'model')
    require(set(metrics) == set(cal) == set(dev_ops) == {(chosen,),('prior',)}, 'Model coverage mismatch')
    test_rows = count(splits['test',]['labelled']); positives = count(splits['test',]['positive'])
    require(count(tm['rows']) == test_rows and count(s['development_rows']) == sum(count(splits[k,]['labelled']) for k in ('train','validation')), 'Model/Gold split mismatch')
    annual = csv_rows(test/'annual_metrics.csv'); annual_i = index(annual,'model','year')
    require(set(annual_i) == {(n,str(y)) for n in (chosen,'prior') for y in range(2008,2015)}, 'Annual coverage mismatch')
    for n in (chosen,'prior'):
        r=metrics[n,]; d=dev_ops[n,]; a=cal[n,]
        operating_checks(d,c_fp,c_fn,oof_rows); operating_checks(r,c_fp,c_fn,test_rows)
        require(count(d['tp'])+count(d['fn']) == sum(count(f['validation_positive']) for f in folds), 'OOF positives mismatch')
        require(count(r['rows']) == test_rows and count(r['positives']) == positives == count(r['tp'])+count(r['fn']), 'Test positives mismatch')
        close(r['prevalence'], positives/test_rows, 'Test prevalence mismatch')
        for row in (r,d): close(row['threshold'],s['thresholds'][n],'Frozen threshold mismatch')
        require(count(r['events']) == count(gm['summary']['test_distinct_positive_events_through_final_horizon']), 'Test horizon events mismatch')
        require(count(a['represented_events']) == count(gm['summary']['test_distinct_failure_events_in_calendar_period']) and count(a['distinct_certs']) == count(gm['summary']['test_distinct_failure_certs_in_calendar_period']), 'Calendar events mismatch')
        require(count(a['detected_events']) <= count(r['detected_events']) and count(a['detected_events']) <= count(a['represented_events']), 'Calendar detections mismatch')
        close(a['event_recall'], count(a['detected_events'])/count(a['represented_events']), 'Calendar recall mismatch')
        parts = [x for x in annual if x['model']==n]
        for k in ('rows','positives','tp','fp','fn','tn','alerts'):
            require(sum(count(x[k]) for x in parts)==count(r[k]), f'Annual row reconciliation failed: {k}')
    refs = index(csv_rows(test/'reference_policies.csv'),'policy')
    require(set(refs)=={('no_alarm',),('all_alarm',)}, 'Reference policies absent')
    for p in refs.values(): operating_checks(p,c_fp,c_fn,test_rows)
    r=metrics[chosen,]; b=metrics['prior',]; a=cal[chosen,]; no=refs['no_alarm',]; all_=refs['all_alarm',]
    require(count(no['alerts'])==0 and count(no['missed_events'])==count(r['events']) and count(all_['alerts'])==test_rows and count(all_['missed_events'])==0, 'Reference policy mismatch')
    spark=read_json(dev/'spark_demo.json'); require(spark['validation_sampled'] is False and s['spark_demo_completed'], 'Spark evidence missing')
    require(count(spark['rows'])==count(folds[0]['validation_rows']) and count(spark['positives'])==count(folds[0]['validation_positive']), 'Spark validation population mismatch')
    return dict(selected_model=chosen, development_rows=count(s['development_rows']), development_events=count(cost['represented_development_events']),
                folds=folds, comparison=cmp, selected_mean_ap=number(by_model[chosen,]['mean_ap']), cost=cost,
                thresholds=s['thresholds'], test_metrics=list(metrics.values()), calendar=list(cal.values()), annual=annual,
                reference_policies=list(refs.values()), spark=spark,
                derived=dict(ap_lift_vs_prevalence=number(r['ap'])/(positives/test_rows),
                             event_recall_pct=100*number(r['event_recall']),calendar_event_recall_pct=100*number(a['event_recall']),
                             early_90d_all_events_pct=100*count(r['lead_at_least_90_days'])/count(r['events']),
                             loss_reduction_vs_prior_pct=100*(1-number(r['loss_usd_thousands'])/number(b['loss_usd_thousands'])) if number(b['loss_usd_thousands'])>0 else None,
                             loss_reduction_vs_no_alarm_pct=100*(1-number(r['loss_usd_thousands'])/number(no['loss_usd_thousands']))))


def collect(workspace):
    """Verify artifact identities and reconcile summary evidence; never fit or score models."""
    ws = Path(workspace).resolve()
    require(not (ws/'pending_source.json').exists(), 'Pending acquisition')
    bronze,bm,b = pointer(ws,'active_source.json','bronze')
    storage,st,st_hash = pointer(ws,'active_storage.json','storage_evidence','evidence.json','evidence_sha256')
    silver,sm,s = pointer(ws,'active_panel.json','silver')
    processing,pm,p = pointer(ws,'active_processing.json','processing_evidence')
    gold,gm,g = pointer(ws,'active_features.json','gold')
    require(all(m['state']=='VERIFIED' for m in (bm,sm,pm,gm)) and st['status']=='PASS', 'Upstream stage not verified')
    require(all(m['source_manifest_sha256']==b for m in (st,sm,pm,gm)) and all(m['silver_manifest_sha256']==s for m in (pm,gm)), 'Upstream lineage mismatch')
    require(read_json(ws/'active_storage.json')['source_manifest_sha256']==b, 'Storage pointer source mismatch')
    require(set(bm['financials'])==set(map(str,quarter_ends())) and set(bm['references'])=={'failures','institutions'}, 'Bronze calendar/reference mismatch')
    entries = [*bm['financials'].values(),*bm['references'].values()]
    files = {e['path']:e['sha256'] for e in entries}
    require(len(files)==len(entries) and set(files)=={x.relative_to(bronze).as_posix() for x in bronze.rglob('*.parquet')}, 'Bronze inventory mismatch')
    check_files(bronze,files,exact=False)
    for e in entries: require(count(e['quality']['rows'])==count(e['api']['api_total']), 'Bronze API count mismatch')
    counts = dict(financials=sum(count(e['quality']['rows']) for e in bm['financials'].values()), **{k:count(e['quality']['rows']) for k,e in bm['references'].items()})
    bv=read_json(bronze/'verification.json')
    require(bv['status']=='PASS' and bv['source_state']=='VERIFIED' and bv['manifest_sha256']==b and bv['row_counts']==counts and bv['quarter_count']==len(bm['financials']), 'Bronze verification mismatch')
    check_files(storage,st['files'],exclude=('evidence.json',))
    for root,m in ((silver,sm),(processing,pm),(gold,gm)): check_files(root,m['files'],exclude=('manifest.json',))
    ss=sm['summary']; gs=gm['summary']
    require(count(ss['source_rows'])==count(ss['panel_rows'])==counts['financials'] and count(ss['rows_removed'])==0, 'Silver row reconciliation failed')
    require(count(ss['bronze_failure_rows'])==counts['failures'] and count(ss['bronze_institution_rows'])==counts['institutions'], 'Reference counts mismatch')
    require(read_json(silver/'summary.json')==ss and read_json(gold/'summary.json')==gs, 'Stage summary mismatch')
    splits=index(csv_rows(gold/'split_summary.csv'),'split')
    for r in splits.values():
        require(count(r['labelled'])==count(r['positive'])+count(r['negative']) and count(r['risk_rows'])==count(r['labelled'])+count(r['censored']), 'Gold split accounting mismatch')
    for k in ('risk_rows','labelled'): require(sum(count(r[k]) for r in splits.values())==count(gs['risk_rows' if k=='risk_rows' else 'labelled_rows']), 'Gold total mismatch')
    require(count(gs['test_distinct_failure_certs_in_calendar_period'])>=50, 'Insufficient represented test failures')
    identity={'bronze':b,'silver':s,'gold':g}; dev=ws/'modelling'/g; test=dev/'test'; attempt=dev.parent/'test_attempt.json'
    dm=read_json(dev/'development_manifest.json'); tm=read_json(test/'manifest.json'); dh=sha256(dev/'development_manifest.json'); th=sha256(test/'manifest.json')
    am=read_json(attempt)
    require(dm['state']=='SEALED' and tm['state']=='VERIFIED' and dm['identity']==tm['identity']==am['identity']==identity, 'Model lineage mismatch')
    require(tm['development_manifest_sha256']==am['development_manifest_sha256']==dh and tm['attempt_sha256']==sha256(attempt), 'Model seal/attempt mismatch')
    actual_dev={x.relative_to(dev).as_posix() for x in dev.rglob('*') if x.is_file() and not x.relative_to(dev).as_posix().startswith('test/')}-{ 'development_manifest.json' }
    require(actual_dev==set(dm['files']), 'Development inventory mismatch')
    check_files(dev,dm['files'],exact=False); check_files(test,tm['files'],exclude=('manifest.json',))
    sizes,queries=storage_findings(csv_rows(storage/'sizes.csv'),csv_rows(storage/'summary.csv'))
    pro=processing_findings(csv_rows(processing/'summary.csv'),counts['financials'])
    flags=Counter()
    for r in csv_rows(silver/'quality_flags.csv'): flags[r['rule']]+=count(r['flagged'])
    fields=csv_rows(silver/'field_summary.csv')
    model=model_findings(dev,test,gm,dm,tm,splits)
    return dict(status='PASS', verification_scope='Artifact hashes, lineage and aggregate arithmetic; no model fitting or scoring',
                lineage=dict(**identity,storage=st_hash,processing=p,development=dh,test=th),
                source=dict(row_counts=counts,quarters=len(bm['financials']),distinct_certs=bv['distinct_financial_certs'],created_utc=bm['created_utc'],sealed_utc=bm['sealed_utc']),
                storage=dict(sizes=sizes,queries=queries,measured_trials=st['measured_trials']),
                quality=dict(summary=ss,flags=[dict(rule=k,flagged_rows=v) for k,v in sorted(flags.items())],fields=fields,terminal=csv_rows(silver/'terminal_summary.csv'),availability=csv_rows(silver/'availability_sensitivity.csv')),
                processing=dict(results=pro,parallelism=csv_rows(processing/'parallelism.csv'),environment=read_json(processing/'environment.json'),measured_trials=pm['measured_trials']),
                features=dict(summary=gs,policy=gm['policy'],splits=list(splits.values()),label_reasons=csv_rows(gold/'label_reasons.csv')),
                modelling=model)
