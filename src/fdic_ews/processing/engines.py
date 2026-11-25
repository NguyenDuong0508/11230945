"""Equivalent native analytical queries; no outcomes or retrospective status."""
from datetime import date
import math
import io
from contextlib import redirect_stdout

START = date(2000, 3, 31)
COLUMNS = ('CERT','REPDTE','ASSET','DEP','ROA','STNAME','DQ_ROE_ABS_GT_100','EQ')
WORKLOADS = ('quarter_summary','adjacent_growth','peer_join')
FIELDS = {
    'quarter_summary': ('REPDTE','ROWS','ASSETS','DEPOSITS','MEAN_ROA','EXTREME_ROE'),
    'adjacent_growth': ('REPDTE','ROWS','ADJACENT','VALID_GROWTH','MEAN_GROWTH'),
    'peer_join': ('REPDTE','ROWS','MATCHED','MEAN_ABS_DEVIATION'),
}
COUNTS = {'ROWS','EXTREME_ROE','ADJACENT','VALID_GROWTH','MATCHED'}
SUMS = {'ASSETS','DEPOSITS'}


def normalize(rows, workload):
    out=[]
    for row in rows:
        if set(row)!=set(FIELDS[workload]): raise ValueError('Result columns differ')
        r={}
        for k in FIELDS[workload]:
            v=row[k]
            if k=='REPDTE':
                if hasattr(v,'date'): v=v.date()
                r[k]=date.fromisoformat(str(v)).isoformat()
            elif k in COUNTS:
                if v is None or isinstance(v,bool) or int(v)!=v or v<0: raise ValueError(f'Invalid count: {k}')
                r[k]=int(v)
            elif v is None or math.isnan(float(v)): r[k]=None
            elif math.isfinite(float(v)): r[k]=float(v)
            else: raise ValueError(f'Nonfinite result: {k}')
        out.append(r)
    out.sort(key=lambda x:x['REPDTE'])
    if not out or len({r['REPDTE'] for r in out})!=len(out): raise ValueError('Empty or duplicate result keys')
    return out


def compare(a,b,workload,scale=1):
    """All output keys/nulls/counts exact; floats tolerant to reduction order."""
    a,b=normalize(a,workload),normalize(b,workload)
    if len(a)!=len(b): raise ValueError('Result lengths differ')
    for x,y in zip(a,b):
        for k,v in x.items():
            w=y[k]
            if v is not None and k in COUNTS|SUMS: v*=scale
            if k=='REPDTE' or k in COUNTS or v is None or w is None:
                ok=v==w
            else: ok=math.isclose(v,w,rel_tol=1e-9,abs_tol=1e-7)
            if not ok: raise ValueError(f'Parity mismatch {x["REPDTE"]}/{k}: {v} vs {w}')


def pandas_transform(df,workload):
    import pandas as pd
    df=df.copy(deep=False); df['REPDTE']=pd.to_datetime(df.REPDTE)
    if workload=='quarter_summary':
        df=df.loc[df.REPDTE>=pd.Timestamp(START)]
        g=df.groupby('REPDTE',sort=True)
        out=g.agg(ROWS=('CERT','size'), MEAN_ROA=('ROA','mean'), EXTREME_ROE=('DQ_ROE_ABS_GT_100','sum'))
        out['ASSETS']=g.ASSET.sum(min_count=1); out['DEPOSITS']=g.DEP.sum(min_count=1)
    elif workload=='adjacent_growth':
        df=df.sort_values(['CERT','REPDTE'],kind='stable')
        df['idx']=df.REPDTE.dt.year*4+df.REPDTE.dt.quarter
        g=df.groupby('CERT',sort=False); prior=g.ASSET.shift(); adjacent=df.idx.sub(g.idx.shift()).eq(1)
        valid=adjacent & prior.gt(0) & df.ASSET.notna()
        df['adjacent']=adjacent; df['growth']=100*(df.ASSET/prior.where(valid)-1)
        out=df.groupby('REPDTE').agg(ROWS=('CERT','size'),ADJACENT=('adjacent','sum'),VALID_GROWTH=('growth','count'),MEAN_GROWTH=('growth','mean'))
    elif workload=='peer_join':
        eligible=df.STNAME.notna() & df.STNAME.ne('') & df.ROA.notna()
        peers=df.loc[eligible].groupby(['REPDTE','STNAME'],as_index=False).ROA.mean().rename(columns={'ROA':'peer'})
        df=df.merge(peers,on=['REPDTE','STNAME'],how='left',validate='many_to_one')
        df['dev']=(df.ROA-df.peer).abs()
        out=df.groupby('REPDTE').agg(ROWS=('CERT','size'),MATCHED=('dev','count'),MEAN_ABS_DEVIATION=('dev','mean'))
    else: raise ValueError(workload)
    return normalize(out.reset_index().to_dict('records'),workload)


def pandas_result(path,workload):
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pandas as pd
    cols={'quarter_summary':['CERT','REPDTE','ASSET','DEP','ROA','DQ_ROE_ABS_GT_100'], 'adjacent_growth':['CERT','REPDTE','ASSET'], 'peer_join':['CERT','REPDTE','STNAME','ROA']}[workload]
    pred=ds.field('REPDTE')>=pa.scalar(START) if workload=='quarter_summary' else None
    return pandas_transform(ds.dataset(str(path),format='parquet').to_table(columns=cols,filter=pred).to_pandas(),workload)


def polars_query(path,workload):
    import polars as pl
    lf=pl.scan_parquet(str(path/'*.parquet'),hive_partitioning=False)
    c=pl.col
    if workload=='quarter_summary':
        def total(f): return pl.when(c(f).count()>0).then(c(f).sum()).otherwise(None)
        return lf.filter(c('REPDTE')>=pl.lit(START)).group_by('REPDTE').agg(pl.len().alias('ROWS'),total('ASSET').alias('ASSETS'),total('DEP').alias('DEPOSITS'),c('ROA').mean().alias('MEAN_ROA'),c('DQ_ROE_ABS_GT_100').cast(pl.Int64).sum().alias('EXTREME_ROE')).sort('REPDTE')
    if workload=='adjacent_growth':
        lf=lf.select('CERT','REPDTE','ASSET').sort('CERT','REPDTE').with_columns((c('REPDTE').dt.year()*4+c('REPDTE').dt.quarter().cast(pl.Int32)).alias('idx'))
        lf=lf.with_columns(c('idx').shift(1).over('CERT').alias('prev_idx'),c('ASSET').shift(1).over('CERT').alias('prev_asset')).with_columns((c('idx')-c('prev_idx')==1).fill_null(False).alias('adjacent'))
        lf=lf.with_columns(pl.when(c('adjacent') & (c('prev_asset')>0) & c('ASSET').is_not_null()).then(100*(c('ASSET')/c('prev_asset')-1)).otherwise(None).alias('growth'))
        return lf.group_by('REPDTE').agg(pl.len().alias('ROWS'),c('adjacent').cast(pl.Int64).sum().alias('ADJACENT'),c('growth').count().alias('VALID_GROWTH'),c('growth').mean().alias('MEAN_GROWTH')).sort('REPDTE')
    if workload=='peer_join':
        lf=lf.select('REPDTE','STNAME','ROA')
        peer=lf.filter(c('STNAME').is_not_null() & (c('STNAME')!='') & c('ROA').is_not_null()).group_by('REPDTE','STNAME').agg(c('ROA').mean().alias('peer'))
        return lf.join(peer,on=['REPDTE','STNAME'],how='left').with_columns((c('ROA')-c('peer')).abs().alias('dev')).group_by('REPDTE').agg(pl.len().alias('ROWS'),c('dev').count().alias('MATCHED'),c('dev').mean().alias('MEAN_ABS_DEVIATION')).sort('REPDTE')
    raise ValueError(workload)


def spark_session(threads=2):
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master(f'local[{threads}]').appName('BFT65B-Processing').config('spark.ui.enabled','false').config('spark.driver.memory','2g').config('spark.sql.shuffle.partitions','8').config('spark.sql.adaptive.enabled','true').config('spark.sql.session.timeZone','UTC').config('spark.driver.bindAddress','127.0.0.1').config('spark.driver.host','127.0.0.1').getOrCreate())


def spark_query(spark,path,workload):
    from pyspark.sql import functions as F, Window
    df=spark.read.parquet(str(path)); c=F.col
    if workload=='quarter_summary':
        return df.filter(c('REPDTE')>=F.lit(START)).groupBy('REPDTE').agg(F.count('*').alias('ROWS'),F.sum('ASSET').alias('ASSETS'),F.sum('DEP').alias('DEPOSITS'),F.avg('ROA').alias('MEAN_ROA'),F.sum(c('DQ_ROE_ABS_GT_100').cast('long')).alias('EXTREME_ROE')).orderBy('REPDTE')
    if workload=='adjacent_growth':
        w=Window.partitionBy('CERT').orderBy('REPDTE')
        df=df.select('CERT','REPDTE','ASSET').withColumn('idx',F.year('REPDTE')*4+F.quarter('REPDTE')).withColumn('prev_idx',F.lag('idx').over(w)).withColumn('prev_asset',F.lag('ASSET').over(w))
        df=df.withColumn('adjacent',F.when(c('idx')-c('prev_idx')==1,1).otherwise(0)).withColumn('growth',F.when((c('adjacent')==1)&(c('prev_asset')>0)&c('ASSET').isNotNull(),100*(c('ASSET')/c('prev_asset')-1)))
        return df.groupBy('REPDTE').agg(F.count('*').alias('ROWS'),F.sum('adjacent').alias('ADJACENT'),F.count('growth').alias('VALID_GROWTH'),F.avg('growth').alias('MEAN_GROWTH')).orderBy('REPDTE')
    if workload=='peer_join':
        df=df.select('REPDTE','STNAME','ROA'); peer=df.filter(c('STNAME').isNotNull() & (c('STNAME')!='') & c('ROA').isNotNull()).groupBy('REPDTE','STNAME').agg(F.avg('ROA').alias('peer'))
        return df.join(peer,['REPDTE','STNAME'],'left').withColumn('dev',F.abs(c('ROA')-c('peer'))).groupBy('REPDTE').agg(F.count('*').alias('ROWS'),F.count('dev').alias('MATCHED'),F.avg('dev').alias('MEAN_ABS_DEVIATION')).orderBy('REPDTE')
    raise ValueError(workload)


def collect(engine,path,workload,spark=None,plan=False):
    if engine=='pandas': return pandas_result(path,workload),None
    if engine=='polars':
        query=polars_query(path,workload); rows=normalize(query.collect().to_dicts(),workload)
        return rows,query.explain(optimized=True) if plan else None
    if engine=='pyspark':
        query=spark_query(spark,path,workload); rows=normalize([r.asDict() for r in query.collect()],workload)
        if plan:
            buf=io.StringIO()
            with redirect_stdout(buf): query.explain(mode='formatted')
            return rows,buf.getvalue()
        return rows,None
    raise ValueError(engine)
