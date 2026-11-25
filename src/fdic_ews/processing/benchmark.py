"""Verified staging, controlled scenarios and manifest-backed evidence."""
from pathlib import Path
import csv, os, random, statistics, subprocess, sys, tempfile, platform, math
from .engines import COLUMNS, WORKLOADS, START, compare
from fdic_ews.contracts import sha256, read_json, write_json, IntegrityError, utc_now
from fdic_ews.acquisition import run_id, code_revision

SCALES=(1,5,10)
SEED=11230945


def scenarios():
    jobs=[(s,e,w,2) for s in SCALES for e in ('pandas','polars') for w in WORKLOADS]
    jobs += [(1,'pyspark',w,2) for w in WORKLOADS]+[(1,'polars',w,1) for w in WORKLOADS]
    random.Random(SEED).shuffle(jobs); return jobs


def stage(silver,manifest,root):
    """One projected Parquet file per quarter at every scale; disjoint synthetic CERTs."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    max_cert=0
    for q,e in sorted(manifest['panel'].items()):
        p=silver/e['path']
        if sha256(p)!=e['sha256']: raise IntegrityError('Silver partition changed')
        t=pq.ParquetFile(p).read(columns=['CERT']); max_cert=max(max_cert,pc.max(t['CERT']).as_py())
    if max_cert<=0 or max_cert*max(SCALES)>=2**63: raise IntegrityError('Unsafe synthetic CERT range')
    meta={s:{'rows':0,'filtered_rows':0,'bytes':0,'files':{}} for s in SCALES}
    for s in SCALES: (root/str(s)).mkdir()
    for q,e in sorted(manifest['panel'].items()):
        p=silver/e['path']; t=pq.ParquetFile(p).read(columns=list(COLUMNS)).select(list(COLUMNS))
        if sha256(p)!=e['sha256'] or t.num_rows!=e['rows']: raise IntegrityError('Staging source mismatch')
        ix=t.schema.get_field_index('CERT'); pieces=[t]
        for i in range(1,max(SCALES)):
            pieces.append(t.set_column(ix,t.schema.field(ix),pc.add(t['CERT'],pa.scalar(i*max_cert,type=pa.int64()))))
        for s in SCALES:
            target=root/str(s)/(q+'.parquet'); expected=pa.concat_tables(pieces[:s]); pq.write_table(expected,target,compression='snappy',row_group_size=65536)
            actual=pq.ParquetFile(target).read()
            if not actual.equals(expected): raise IntegrityError('Staged values differ')
            m=meta[s]; m['rows']+=actual.num_rows; m['filtered_rows']+=actual.num_rows if q>=START.isoformat() else 0; m['bytes']+=target.stat().st_size; m['files'][target.name]=sha256(target)
    for s,m in meta.items():
        if m['rows']!=s*manifest['summary']['panel_rows']: raise IntegrityError('Scale count mismatch')
    return meta


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def summarize(trials):
    groups={}
    for r in trials:
        if r['phase']=='measured': groups.setdefault((r['scale'],r['engine'],r['workload'],r['threads']),[]).append(r)
    out=[]
    for (s,e,w,t),rs in sorted(groups.items()):
        times=[r['seconds'] for r in rs]; q=statistics.quantiles(times,n=4,method='inclusive')
        out.append(dict(scale=s,engine=e,workload=w,threads=t,n=len(rs),input_rows=rs[0]['input_rows'],median_seconds=statistics.median(times),iqr_seconds=q[2]-q[0],median_peak_rss_mib=statistics.median(r['peak_rss_mib'] for r in rs),median_incremental_rss_mib=statistics.median(r['incremental_rss_mib'] for r in rs)))
    return out


def comparisons(summary):
    d={(r['scale'],r['engine'],r['workload'],r['threads']):r for r in summary}; scaling=[]; threads=[]
    for e in ('pandas','polars'):
        for w in WORKLOADS:
            base=d[1,e,w,2]['median_seconds']
            for s in SCALES:
                row=d[s,e,w,2]; scaling.append(dict(engine=e,workload=w,scale=s,input_rows=row['input_rows'],time_ratio=row['median_seconds']/base,rows_per_second=row['input_rows']/row['median_seconds']))
    for w in WORKLOADS:
        one=d[1,'polars',w,1]['median_seconds']; two=d[1,'polars',w,2]['median_seconds']; speed=one/two
        threads.append(dict(workload=w,one_thread_s=one,two_threads_s=two,speedup=speed,efficiency=speed/2,apparent_serial_fraction=2/speed-1))
    return scaling,threads


def validate_trials(results,meta,reps=3):
    expected=set(scenarios()); seen=set(); refs={}; trials=[]
    for job,data in results:
        s,e,w,t=job
        if job in seen or job not in expected: raise IntegrityError('Unexpected scenario')
        seen.add(job)
        if len(data['trials'])!=reps+1: raise IntegrityError('Incomplete trials')
        if e!='pandas' and not data.get('plan'): raise IntegrityError('Missing plan')
        count=meta[s]['filtered_rows' if w=='quarter_summary' else 'rows']
        for i,r in enumerate(data['trials']):
            if r['repeat']!=i or r['phase']!=('warmup' if i==0 else 'measured') or not all(math.isfinite(r[k]) for k in ('seconds','baseline_rss_mib','peak_rss_mib','incremental_rss_mib')) or not r['seconds']>0 or r['baseline_rss_mib']<0 or r['peak_rss_mib']<r['baseline_rss_mib'] or not math.isclose(r['incremental_rss_mib'],r['peak_rss_mib']-r['baseline_rss_mib'],abs_tol=1e-8) or r['samples']<2: raise IntegrityError('Invalid trial')
            rows=r['result']
            if sum(x['ROWS'] for x in rows)!=count: raise IntegrityError('Workload row accounting failed')
            if (s,w) in refs: compare(refs[s,w],rows,w)
            else: refs[s,w]=rows
            trials.append(dict(scale=s,engine=e,workload=w,threads=t,input_rows=count,**{k:v for k,v in r.items() if k!='result'}))
    if seen!=expected: raise IntegrityError('Missing scenario')
    for s in SCALES:
        for w in WORKLOADS: compare(refs[1,w],refs[s,w],w,scale=s)
    return trials


def figures(out,summary,scaling):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(9,3.1))
    for ax,w in zip(axes,WORKLOADS):
        r=[x for x in summary if x['scale']==1 and x['threads']==2 and x['workload']==w]; ax.bar([x['engine'] for x in r],[x['median_seconds'] for x in r]); ax.set_title(w.replace('_',' '),fontsize=10); ax.set_ylabel('Median seconds'); ax.tick_params(axis='x',rotation=25)
    fig.tight_layout(); fig.savefig(out/'runtime.png',dpi=150); plt.close(fig)
    for metric,filename,label in [('median_seconds','scaling.png','Median seconds'),('median_peak_rss_mib','memory.png','Sampled peak tree RSS (MiB)')]:
        fig,axes=plt.subplots(1,3,figsize=(9,3.1))
        for ax,w in zip(axes,WORKLOADS):
            for e in ('pandas','polars'):
                r=[x for x in summary if x['threads']==2 and x['workload']==w and x['engine']==e]; ax.plot([x['scale'] for x in r],[x[metric] for x in r],marker='o',label=e)
            ax.set_title(w.replace('_',' '),fontsize=10); ax.set_xticks(SCALES); ax.set_xlabel('Input multiplier'); ax.set_ylabel(label); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out/filename,dpi=150); plt.close(fig)


def run(workspace,scratch):
    from fdic_ews.quality.panel import resolve_panel
    import importlib.metadata as im
    import psutil
    workspace=Path(workspace).resolve(); scratch=Path(scratch).resolve()
    if scratch==workspace or scratch in workspace.parents or workspace in scratch.parents: raise IntegrityError('Use separate local scratch')
    if (workspace/'pending_source.json').exists(): raise IntegrityError('Pending acquisition')
    silver,m=resolve_panel(workspace); digest=sha256(silver/'manifest.json')
    out=workspace/'processing_evidence'/run_id(); out.mkdir(parents=True,exist_ok=False); (out/'workers').mkdir(); (out/'plans').mkdir()
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='part3-',dir=scratch) as tmp:
        root=Path(tmp)/'stage'; meta=stage(silver,m,root); results=[]
        for i,job in enumerate(scenarios(),1):
            s,e,w,t=job; name=f'{s}x_{e}_{w}_{t}t'; dest=out/'workers'/(name+'.json')
            env=dict(os.environ,POLARS_MAX_THREADS=str(t),OMP_NUM_THREADS=str(t),OPENBLAS_NUM_THREADS=str(t),MKL_NUM_THREADS=str(t),SPARK_LOCAL_IP='127.0.0.1',PYSPARK_PYTHON=sys.executable,PYTHONPATH=str(Path(__file__).resolve().parents[2])+os.pathsep+os.environ.get('PYTHONPATH',''))
            cmd=[sys.executable,'-m','fdic_ews.processing.worker','--path',str(root/str(s)),'--engine',e,'--workload',w,'--threads',str(t),'--output',str(dest)]
            proc=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=3600)
            if proc.returncode: raise IntegrityError(f'Worker {name} failed:\n{proc.stdout}\n{proc.stderr}')
            data=read_json(dest); results.append((job,data))
            if data['plan']: (out/'plans'/(name+'.txt')).write_text(data['plan'],encoding='utf-8')
            print(f'{i}/{len(scenarios())} {name}: complete',flush=True)
        for s,entry in meta.items():
            for f,h in entry['files'].items():
                if sha256(root/str(s)/f)!=h: raise IntegrityError('Benchmark input modified')
        trials=validate_trials(results,meta); summary=summarize(trials); scaling,threads=comparisons(summary)
        for f,r in [('trials.csv',trials),('summary.csv',summary),('scaling.csv',scaling),('parallelism.csv',threads)]: csv_write(out/f,r)
        figures(out,summary,scaling); write_json(out/'staging.json',meta)
    env_info=dict(python=platform.python_version(),platform=platform.platform(),cpu_count=os.cpu_count(),cpu_affinity=len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(),'cpu_affinity') else None,ram_mib=psutil.virtual_memory().total/2**20,packages={p:im.version(p) for p in ('pandas','polars','pyarrow','pyspark','psutil')},java=subprocess.run(['java','-version'],capture_output=True,text=True,check=True).stderr)
    write_json(out/'environment.json',env_info)
    if read_json(workspace/'active_panel.json')['manifest_sha256']!=digest or sha256(silver/'manifest.json')!=digest: raise IntegrityError('Active Silver changed during run')
    files={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    evidence=dict(state='VERIFIED',created_utc=utc_now(),code_revision=code_revision(),silver_manifest_sha256=digest,source_manifest_sha256=m['source_manifest_sha256'],seed=SEED,scenarios=len(scenarios()),warmups=len(scenarios()),measured_trials=3*len(scenarios()),files=files)
    write_json(out/'manifest.json',evidence); verify_evidence(workspace,out)
    write_json(workspace/'active_processing.json',dict(path=out.relative_to(workspace).as_posix(),manifest_sha256=sha256(out/'manifest.json')))
    return out


def verify_evidence(workspace,out=None):
    workspace=Path(workspace).resolve(); panel=read_json(workspace/'active_panel.json')
    if out is None:
        p=read_json(workspace/'active_processing.json'); out=(workspace/p['path']).resolve()
        if sha256(out/'manifest.json')!=p['manifest_sha256']: raise IntegrityError('Processing pointer mismatch')
    out=Path(out).resolve(); m=read_json(out/'manifest.json')
    if out.parent!=workspace/'processing_evidence' or m['state']!='VERIFIED' or m['silver_manifest_sha256']!=panel['manifest_sha256']: raise IntegrityError('Processing lineage mismatch')
    if sha256(workspace/panel['path']/'manifest.json')!=panel['manifest_sha256']: raise IntegrityError('Silver manifest mismatch')
    from fdic_ews.acquisition import resolve_pointer
    source=resolve_pointer(workspace)
    if m['source_manifest_sha256']!=sha256(source/'manifest.json'): raise IntegrityError('Bronze lineage mismatch')
    actual={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}
    if actual!=m['files']: raise IntegrityError('Processing artifact mismatch')
    meta={int(k):v for k,v in read_json(out/'staging.json').items()}; results=[]
    for job in scenarios():
        s,e,w,t=job; results.append((job,read_json(out/'workers'/f'{s}x_{e}_{w}_{t}t.json')))
    trials=validate_trials(results,meta)
    if m['scenarios']!=len(scenarios()) or m['warmups']!=len(scenarios()) or m['measured_trials']!=3*len(scenarios()) or len(trials)!=m['warmups']+m['measured_trials']: raise IntegrityError('Trial count mismatch')
    summary=summarize(trials); scaling,parallel=comparisons(summary)
    for name,rows in [('trials.csv',trials),('summary.csv',summary),('scaling.csv',scaling),('parallelism.csv',parallel)]:
        with (out/name).open(newline='',encoding='utf-8') as f:
            actual_rows=list(csv.DictReader(f))
        expected_rows=[{k:str(v) for k,v in r.items()} for r in rows]
        if actual_rows!=expected_rows: raise IntegrityError(f'Summary reconciliation failed: {name}')
    for job,data in results:
        s,e,w,t=job
        if data['plan'] and (out/'plans'/f'{s}x_{e}_{w}_{t}t.txt').read_text(encoding='utf-8')!=data['plan']: raise IntegrityError('Plan evidence mismatch')
    return out,m
