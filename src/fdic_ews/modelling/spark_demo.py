"""Development-only Spark MLlib demonstration with unsampled validation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import numpy as np
import pandas as pd
from fdic_ews.features.policy import SEED
from .metrics import ranking
from .design import META

SPARK_FEATURES=['LOG_ASSET','EQUITY_ASSETS_PCT','DEPOSITS_ASSETS_PCT','LOANS_ASSETS_PCT','ROA_PCT']


def run_demo(train,valid,*,negative_limit=20000):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyspark.sql import SparkSession, functions as F
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Imputer, VectorAssembler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.functions import vector_to_array
    from time import perf_counter
    if set(train.SPLIT)!={'train'} or not set(valid.SPLIT)<={'train','validation'} or pd.to_datetime(train.HORIZON_END).max()>=pd.to_datetime(valid.DECISION_DATE).min(): raise ValueError('Invalid Spark demonstration split')
    neg=train.loc[train.Y_FAIL_4Q.eq(0)]; pos=train.loc[train.Y_FAIL_4Q.eq(1)]
    sampled=pd.concat([pos,neg.sample(n=min(len(neg),negative_limit),random_state=SEED)],ignore_index=True)
    if set(sampled.Y_FAIL_4Q.unique())!={0,1} or sampled[SPARK_FEATURES].isna().all().any(): raise ValueError('Spark training lacks required observations')
    os.environ.setdefault('SPARK_LOCAL_IP','127.0.0.1')
    import sys
    os.environ['PYSPARK_PYTHON']=sys.executable
    with TemporaryDirectory(prefix='fdic_mllib_') as td:
        d=Path(td)
        for name,f in [('train',sampled),('validation',valid)]:
            x=f[SPARK_FEATURES].astype(float).copy(); x['label']=f.Y_FAIL_4Q.astype(float).to_numpy(); x['row_id']=np.arange(len(f),dtype=np.int64)
            pq.write_table(pa.Table.from_pandas(x,preserve_index=False),d/(name+'.parquet'))
        spark=(SparkSession.builder.master('local[2]').appName('FDICDevelopmentMLlib')
               .config('spark.ui.enabled','false').config('spark.sql.shuffle.partitions','4')
               .config('spark.driver.host','127.0.0.1').config('spark.driver.bindAddress','127.0.0.1').getOrCreate())
        try:
            spark.sparkContext.setLogLevel('ERROR')
            cols=[c+'_imputed' for c in SPARK_FEATURES]
            pipeline=Pipeline(stages=[Imputer(inputCols=SPARK_FEATURES,outputCols=cols,strategy='median'),VectorAssembler(inputCols=cols,outputCol='features'),LogisticRegression(featuresCol='features',labelCol='label',maxIter=50,regParam=.1,elasticNetParam=0.)])
            t=perf_counter(); model=pipeline.fit(spark.read.parquet(str(d/'train.parquet'))); fit_seconds=perf_counter()-t
            t=perf_counter(); scored=model.transform(spark.read.parquet(str(d/'validation.parquet'))).select('row_id','label',vector_to_array('probability')[1].alias('score'))
            rows=scored.orderBy('row_id').collect(); score_seconds=perf_counter()-t
            if [r.row_id for r in rows]!=list(range(len(valid))) or [int(r.label) for r in rows]!=valid.Y_FAIL_4Q.astype(int).tolist(): raise ValueError('Spark validation row mismatch')
            pred=valid[META].copy().reset_index(drop=True); pred['score']=[r.score for r in rows]
            return {**ranking(pred),'spark_version':spark.version,'mode':'local[2]','features':SPARK_FEATURES,
                    'training_population_rows':len(train),'sampled_training_rows':len(sampled),'sampled_positive_rows':len(pos),
                    'sampled_negative_rows':len(sampled)-len(pos),'validation_sampled':False,'fit_seconds':fit_seconds,'score_seconds':score_seconds}
        finally: spark.stop()
