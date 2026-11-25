"""Vectorized report-local and exact-calendar historical financial features."""
import numpy as np
import pandas as pd
from .policy import LAG, MAX_AGE
from fdic_ews.quality.policy import shift_quarter, available_at, quarter_id

# name: (economic group, formula/definition, rationale)
DETAILS={
 'LOG_ASSET':('size','log1p(ASSET), ASSET > 0','Bank scale'),
 'EQUITY_ASSETS_PCT':('capital','100 * EQ / ASSET','Loss absorption'),
 'TANGIBLE_EQUITY_ASSETS_PCT':('capital','100 * (EQ - INTAN) / ASSET','Tangible loss absorption'),
 'DEPOSITS_ASSETS_PCT':('funding','100 * DEP / ASSET','Deposit funding dependence'),
 'CASH_ASSETS_PCT':('liquidity','100 * CHBAL / ASSET','Cash buffer'),
 'SECURITIES_ASSETS_PCT':('liquidity','100 * SC / ASSET','Securities allocation, not necessarily liquid'),
 'LOANS_ASSETS_PCT':('asset_quality','100 * LNLSGR / ASSET','Credit concentration'),
 'BROKERED_DEPOSITS_PCT':('funding','100 * BRO / DEP','Wholesale funding dependence'),
 'LOANS_DEPOSITS_PCT':('liquidity','100 * LNLSGR / DEP','Loan funding pressure'),
 'NONCURRENT_LOANS_PCT':('asset_quality','100 * NCLNLS / LNLSGR','Noncurrent credit exposure'),
 'ALLOWANCE_LOANS_PCT':('asset_quality','100 * LNATRES / LNLSGR','Reserve relative to loans'),
 'TEXAS_LIKE_PCT':('asset_quality','100 * (NCLNLS + ORE) / (EQ - INTAN + LNATRES)','Distressed assets relative to tangible equity and reserves; proxy'),
 'ROA_PCT':('earnings','reported ROA','Profitability'),
 'ROE_PCT':('earnings','reported ROE','Profitability relative to equity'),
 'NIM_PCT':('earnings','reported NIMY','Net interest margin'),
 'RBC1AAJ_REPORTED_PCT':('capital','reported RBC1AAJ','Reported capital measure, definition varies historically'),
 'RBCRWAJ_REPORTED_PCT':('capital','reported RBCRWAJ','Reported risk-based capital measure'),
 'RBC1AAJ_ZERO':('data_quality','1 if reported RBC1AAJ = 0; null if absent','Ambiguous capital reporting'),
 'RBCRWAJ_ZERO':('data_quality','1 if reported RBCRWAJ = 0; null if absent','Ambiguous capital reporting'),
 'ASSET_GROWTH_QOQ_PCT':('growth','100 * (ASSET / ASSET[q-1] - 1)','Recent balance-sheet growth'),
 'DEPOSIT_GROWTH_QOQ_PCT':('growth','100 * (DEP / DEP[q-1] - 1)','Deposit movement'),
 'ASSET_GROWTH_YOY_PCT':('growth','100 * (ASSET / ASSET[q-4] - 1)','Annual expansion'),
 'ROA_CHANGE_YOY_PPT':('earnings','ROA - ROA[q-4]','Profitability deterioration'),
 'REPORTS_TO_DATE':('report_history','Silver backward-looking count','Reporting history'),
 'PRIOR_GAP_QUARTERS':('report_history','Silver preceding report gap','Reporting discontinuity'),
 'REPORT_AGE_QUARTERS':('report_history','decision quarter minus report quarter','Information staleness'),
 'DQ_EXTREME_ROE':('data_quality','Silver DQ_ROE_ABS_GT_100','Extreme ratio flag'),
 'MISSING_FINANCIAL_COUNT':('data_quality','Missing values in fixed raw financial inputs','Information completeness'),
 'ROA_PEER_DEV_PPT':('earnings','ROA minus leave-one-out mean among available same-state same-report-quarter peers','Relative profitability'),
}
FEATURES=tuple(DETAILS)
RAW_NUMERIC=tuple('ASSET EQ INTAN DEP CHBAL SC LNLSGR BRO NCLNLS LNATRES ORE ROA ROE NIMY RBC1AAJ RBCRWAJ'.split())
INPUT_COLUMNS=('CERT','REPDTE','STNAME',*RAW_NUMERIC,'REPORTS_TO_DATE','PRIOR_GAP_QUARTERS','DQ_ROE_ABS_GT_100','ASSUMED_AVAILABLE_AT')
CAPITAL=('RBC1AAJ_REPORTED_PCT','RBCRWAJ_REPORTED_PCT')


def ratio(a,b):
    return (100*a/b.where(b>0)).replace([np.inf,-np.inf],np.nan)


def growth(a,b): return ratio(a,b)-100


def report_features(frame,prev1,prev4):
    """The caller supplies exact previous-quarter and previous-year tables."""
    if frame.empty: raise ValueError('Empty current report partition')
    d=pd.to_datetime(frame.REPDTE).iloc[0].date()
    for f,expected in ((frame,d),(prev1,shift_quarter(d,-1)),(prev4,shift_quarter(d,-4))):
        if not pd.to_datetime(f.REPDTE).eq(pd.Timestamp(expected)).all(): raise ValueError('Non-exact historical report partition')
    x=frame.set_index('CERT',verify_integrity=True)
    p=prev1.set_index('CERT',verify_integrity=True); y=prev4.set_index('CERT',verify_integrity=True)
    z=pd.DataFrame(index=x.index)
    z['REPORT_DATE']=pd.to_datetime(x.REPDTE); z['AVAILABLE_AT']=pd.to_datetime(x.ASSUMED_AVAILABLE_AT); z['STNAME']=x.STNAME
    asset=x.ASSET
    z['LOG_ASSET']=np.log1p(asset.where(asset>0))
    for name,n,d in [('EQUITY_ASSETS_PCT',x.EQ,asset),('TANGIBLE_EQUITY_ASSETS_PCT',x.EQ-x.INTAN,asset),('DEPOSITS_ASSETS_PCT',x.DEP,asset),('CASH_ASSETS_PCT',x.CHBAL,asset),('SECURITIES_ASSETS_PCT',x.SC,asset),('LOANS_ASSETS_PCT',x.LNLSGR,asset),('BROKERED_DEPOSITS_PCT',x.BRO,x.DEP),('LOANS_DEPOSITS_PCT',x.LNLSGR,x.DEP),('NONCURRENT_LOANS_PCT',x.NCLNLS,x.LNLSGR),('ALLOWANCE_LOANS_PCT',x.LNATRES,x.LNLSGR)]:
        z[name]=ratio(n,d)
    # Require nonnegative components; negative/zero combined capital yields null.
    n=(x.NCLNLS+x.ORE).where((x.NCLNLS>=0)&(x.ORE>=0))
    d=(x.EQ-x.INTAN+x.LNATRES).where((x.INTAN>=0)&(x.LNATRES>=0))
    z['TEXAS_LIKE_PCT']=ratio(n,d)
    for k,v in [('ROA_PCT','ROA'),('ROE_PCT','ROE'),('NIM_PCT','NIMY'),(CAPITAL[0],'RBC1AAJ'),(CAPITAL[1],'RBCRWAJ')]: z[k]=x[v]
    for v in ('RBC1AAJ','RBCRWAJ'): z[v+'_ZERO']=x[v].eq(0).astype(float).where(x[v].notna())
    z['ASSET_GROWTH_QOQ_PCT']=growth(asset,p.ASSET.reindex(x.index))
    z['DEPOSIT_GROWTH_QOQ_PCT']=growth(x.DEP,p.DEP.reindex(x.index))
    z['ASSET_GROWTH_YOY_PCT']=growth(asset,y.ASSET.reindex(x.index))
    z['ROA_CHANGE_YOY_PPT']=x.ROA-y.ROA.reindex(x.index)
    for k in ('REPORTS_TO_DATE','PRIOR_GAP_QUARTERS'): z[k]=x[k]
    z['DQ_EXTREME_ROE']=x.DQ_ROE_ABS_GT_100.astype(float)
    z['MISSING_FINANCIAL_COUNT']=x[list(RAW_NUMERIC)].isna().sum(axis=1)
    return z.reset_index()


def latest_reports(q, frames,lag=LAG):
    """Select from eligible report-quarter frames; no future-report search."""
    eligible=[]
    for r,frame in frames:
        if 1<=quarter_id(q)-quarter_id(r)<=MAX_AGE and available_at(r,lag)<=q:
            eligible.append((r,frame))
    if not eligible: return None
    return pd.concat([f for r,f in sorted(eligible,key=lambda p:p[0],reverse=True)],ignore_index=True).drop_duplicates('CERT',keep='first').copy()


def decision_features(frame,q):
    """Peer calculation after as-of risk filtering, before outcome adjudication."""
    z=frame.copy()
    z['REPORT_AGE_QUARTERS']=(q.year*4+q.month//3)-(z.REPORT_DATE.dt.year*4+z.REPORT_DATE.dt.quarter)
    ok=z.STNAME.notna() & z.STNAME.ne('') & z.ROA_PCT.notna()
    p=z.loc[ok,['STNAME','REPORT_DATE','ROA_PCT']]
    g=p.groupby(['STNAME','REPORT_DATE'])['ROA_PCT']
    peer=(g.transform('sum')-p.ROA_PCT)/(g.transform('count')-1).where(g.transform('count')>1)
    z['ROA_PEER_DEV_PPT']=(p.ROA_PCT-peer).reindex(z.index)
    z[list(FEATURES)]=z[list(FEATURES)].astype(float).replace([np.inf,-np.inf],np.nan)
    return z[['CERT','REPORT_DATE','AVAILABLE_AT',*FEATURES]]
