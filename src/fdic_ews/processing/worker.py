"""Isolated scenario worker; sample current process-tree RSS during execution."""
import argparse
import json
import time
import threading
from pathlib import Path
from .engines import collect, compare, spark_session


def memory_mib():
    import psutil
    p=psutil.Process(); total=0
    for child in [p,*p.children(recursive=True)]:
        try: total+=child.memory_info().rss
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    return total/2**20


def measured(fn):
    baseline=memory_mib(); samples=[baseline]; stop=threading.Event()
    def poll():
        while not stop.wait(.02): samples.append(memory_mib())
    thread=threading.Thread(target=poll,daemon=True); thread.start()
    start=time.perf_counter()
    try:
        result=fn(); seconds=time.perf_counter()-start; samples.append(memory_mib())
    finally:
        stop.set(); thread.join()
    return result,dict(seconds=seconds,baseline_rss_mib=baseline,peak_rss_mib=max(samples),incremental_rss_mib=max(samples)-baseline,samples=len(samples))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--path',type=Path,required=True); p.add_argument('--engine',required=True); p.add_argument('--workload',required=True); p.add_argument('--threads',type=int,required=True); p.add_argument('--reps',type=int,default=3); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    spark=None
    if a.engine=='pandas':
        import pandas, pyarrow as pa
        pa.set_cpu_count(a.threads); pa.set_io_thread_count(a.threads)
    elif a.engine=='polars':
        import polars as pl
        if pl.thread_pool_size()!=a.threads: raise RuntimeError('Unexpected Polars thread pool')
    elif a.engine=='pyspark': spark=spark_session(a.threads)
    else: raise ValueError(a.engine)
    trials=[]; reference=None
    try:
        for i in range(a.reps+1):
            (rows,_),metrics=measured(lambda:collect(a.engine,a.path,a.workload,spark))
            if reference is None: reference=rows
            else: compare(reference,rows,a.workload)
            trials.append(dict(phase='warmup' if i==0 else 'measured',repeat=i,**metrics,result=rows))
        plan=None
        if a.engine!='pandas':
            # Plan retrieval is outside measured executions; Spark plan is post-action.
            rows,plan=collect(a.engine,a.path,a.workload,spark,plan=True); compare(reference,rows,a.workload)
        a.output.write_text(json.dumps(dict(trials=trials,plan=plan),allow_nan=False),encoding='utf-8')
    finally:
        if spark is not None: spark.stop()

if __name__=='__main__': main()
