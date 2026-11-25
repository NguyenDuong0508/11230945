"""Quarter-wise financial audit and deterministic backward-looking transformations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .policy import quarter_id, available_at, LAGS

# Income-statement fields for an explicitly conditional YTD sensitivity view.
FLOW_FIELDS = ("NETINC", "INTINC", "EINTEXP", "NONII", "NONIX")
CAPITAL_FIELDS = ("RBC1AAJ", "RBCRWAJ")
NUMERIC = tuple("ASSET DEP DEPDOM EQ BRO CHBAL SC LNLSGR LNLSNET LNRE LNCI LNCON LNATRES NCLNLS P3ASSET P9ASSET NAASSET NTLNLS ORE INTAN NETINC INTINC EINTEXP NONII NONIX ELNATR ROA ROE NIMY RBC1AAJ RBCRWAJ".split())
RULES = {
    "ASSET_NONPOSITIVE": ("invalid denominator", "Set ASSET to null in Silver; retain row and flag; Bronze unchanged"),
    "NEGATIVE_DEP": ("review", "Retain and flag"),
    "NEGATIVE_LOANS": ("review", "Retain and flag"),
    "DEP_GT_ASSET": ("review", "Retain and flag; distressed balance sheets may be genuine"),
    "LOANS_GT_ASSET": ("review", "Retain and flag"),
    "NONCURRENT_GT_LOANS": ("review", "Retain and flag; verify scope and definitions"),
    "NEGATIVE_EQUITY": ("risk signal", "Retain; not an automatic error"),
    "ROA_ABS_GT_100": ("review", "Retain; diagnostic threshold, not a regulatory bound"),
    "ROE_ABS_GT_100": ("review", "Retain; small equity denominator can cause extremes"),
    "CAPITAL_ZERO": ("ambiguous reporting", "Retain original zeros; separate sensitivity treatment"),
    "NONFINITE_VALUE": ("invalid", "Replace nonfinite numeric values with null and flag"),
    "REPORTING_GAP": ("temporal review", "No synthetic quarters or forward filling"),
}


def clean_quarter(frame, q, history):
    """Process one quarter in chronological order; history contains past only.

    history maps CERT to (last quarter index, previous flow values, first report,
    number of observed reports). Conditional YTD differences are explicitly
    named *_Q_IF_YTD and are not default predictors.
    """
    df = frame.copy().sort_values("CERT", kind="stable").reset_index(drop=True)
    if df.empty or df["CERT"].isna().any() or df["CERT"].duplicated().any() or (df["CERT"] <= 0).any():
        raise ValueError("Invalid bank-quarter keys")
    if not pd.to_datetime(df["REPDTE"]).dt.date.eq(q).all():
        raise ValueError("Unexpected reporting date")
    k = quarter_id(q)
    certs = df["CERT"].tolist()
    prev = [history.get(c) for c in certs]
    if any(p is not None and p[0] >= k for p in prev):
        raise ValueError("History must precede the current quarter")
    audit = []
    nonfinite_any = np.zeros(len(df), dtype=bool)
    for f in NUMERIC:
        v = pd.to_numeric(df[f], errors="raise").astype("float64")
        nonfinite = v.notna() & ~np.isfinite(v)
        nonfinite_any |= nonfinite.to_numpy()
        finite = v.mask(nonfinite)
        audit.append(dict(quarter=q.isoformat(), field=f, rows=len(df), missing=int(v.isna().sum()),
                          zeros=int(v.eq(0).sum()), negative=int(v.lt(0).sum()), nonfinite=int(nonfinite.sum()),
                          min=None if finite.isna().all() else float(finite.min()),
                          max=None if finite.isna().all() else float(finite.max())))
        df[f] = finite
    state = df["STNAME"]
    audit.append(dict(quarter=q.isoformat(), field="STNAME", rows=len(df), missing=int(state.isna().sum()),
                      zeros=0, negative=0, nonfinite=0, min=None, max=None))
    flags = {
        "ASSET_NONPOSITIVE": df.ASSET.le(0), "NEGATIVE_DEP": df.DEP.lt(0),
        "NEGATIVE_LOANS": df.LNLSGR.lt(0), "DEP_GT_ASSET": df.DEP.gt(df.ASSET),
        "LOANS_GT_ASSET": df.LNLSGR.gt(df.ASSET), "NONCURRENT_GT_LOANS": df.NCLNLS.gt(df.LNLSGR),
        "NEGATIVE_EQUITY": df.EQ.lt(0), "ROA_ABS_GT_100": df.ROA.abs().gt(100),
        "ROE_ABS_GT_100": df.ROE.abs().gt(100),
        "CAPITAL_ZERO": df[list(CAPITAL_FIELDS)].eq(0).any(axis=1),
        "NONFINITE_VALUE": pd.Series(nonfinite_any),
        "REPORTING_GAP": pd.Series([p is not None and k-p[0]>1 for p in prev]),
    }
    for name, flag in flags.items():
        df["DQ_" + name] = flag.astype(bool)
    df["ASSET"] = df.ASSET.mask(df.DQ_ASSET_NONPOSITIVE)
    df["PRIOR_GAP_QUARTERS"] = pd.array([None if p is None else k-p[0]-1 for p in prev], dtype="Int32")
    df["REPORTS_TO_DATE"] = np.asarray([1 if p is None else p[3]+1 for p in prev], dtype="int32")
    df["ASSUMED_AVAILABLE_AT"] = available_at(q)
    flow_stats = []
    for f in FLOW_FIELDS:
        values = df[f].to_numpy(dtype=float)
        prior = np.asarray([np.nan if p is None or p[0] != k-1 else p[1][f] for p in prev], dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            derived = values.copy() if q.month == 3 else values-prior
        derived[~np.isfinite(derived)] = np.nan
        df[f + "_Q_IF_YTD"] = derived
        flow_stats.append(dict(quarter=q.isoformat(), field=f, reporting_regime="pre_2012_mixed" if q.year<2012 else "2012_onward_call_reports", source_nonnull=int(np.isfinite(values).sum()),
                               candidate_nonnull=int(np.isfinite(derived).sum()),
                               missing_predecessor=int((np.isfinite(values) & ~np.isfinite(prior)).sum()) if q.month!=3 else 0,
                               candidate_negative=int((derived<0).sum())))
    cap_stats = []
    for f in CAPITAL_FIELDS:
        v = df[f]
        # Separate scenario, never treat zero as missing in primary data.
        cap_stats.append(dict(quarter=q.isoformat(), field=f, regime="2020_onward" if q.year>=2020 else "pre_2020",
                              rows=len(df), zeros=int(v.eq(0).sum()), missing=int(v.isna().sum()),
                              sensitivity_nonnull=int(v.mask(v.eq(0)).notna().sum())))
    for i, c in enumerate(certs):
        p = prev[i]
        history[c] = (k, {f: float(df[f].iat[i]) for f in FLOW_FIELDS}, q if p is None else p[2], int(df.REPORTS_TO_DATE.iat[i]))
    flag_counts = [dict(quarter=q.isoformat(), rule=f, rows=len(df), flagged=int(df["DQ_"+f].sum())) for f in RULES]
    return df, audit, flag_counts, flow_stats, cap_stats


def availability_counts(q, frame):
    """Availability at later quarter-end decision dates; no outcome-based filtering."""
    from .policy import shift_quarter
    return [dict(report_quarter=q.isoformat(), lag_days=lag,
                 decision_quarter=shift_quarter(q, steps).isoformat(),
                 available_rows=len(frame) if available_at(q, lag)<=shift_quarter(q, steps) else 0,
                 report_rows=len(frame)) for lag in LAGS for steps in (0,1,2)]
