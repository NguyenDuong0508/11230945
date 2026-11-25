"""Construct and verify a Silver panel from the active Bronze manifest."""
from __future__ import annotations
import csv
import platform
from collections import Counter, defaultdict
from pathlib import Path

from fdic_ews.acquisition import resolve_pointer, verify as verify_bronze, parquet_commit, run_id, code_revision
from fdic_ews.contracts import IntegrityError, FIELDS, schema, quarter_ends, sha256, read_json, write_json, utc_now, partition_path
from .financial import clean_quarter, availability_counts, RULES, FLOW_FIELDS, CAPITAL_FIELDS
from .policy import event_records, terminal_audit, index_events, horizon_outcome, LAGS, STUDY_END, shift_quarter

POLICY_VERSION = "1.0.0"


def panel_schema():
    import pyarrow as pa
    return pa.schema(list(schema("financials")) + [pa.field("DQ_"+r, pa.bool_(), nullable=False) for r in RULES]
                     + [pa.field("PRIOR_GAP_QUARTERS", pa.int32()), pa.field("REPORTS_TO_DATE", pa.int32(), nullable=False),
                        pa.field("ASSUMED_AVAILABLE_AT", pa.date32(), nullable=False)]
                     + [pa.field(f+"_Q_IF_YTD", pa.float64()) for f in FLOW_FIELDS])


def to_panel_table(df):
    import pyarrow as pa
    # Explicit Arrow types preserve nullability and date32 across every quarter.
    return pa.Table.from_pandas(df, schema=panel_schema(), preserve_index=False, safe=True).replace_schema_metadata(None)


def event_schema():
    import pyarrow as pa
    return pa.schema([("event_id", pa.string()), ("CERT", pa.int64()), ("FIN", pa.int64()),
        ("event_date", pa.date32()), ("date_status", pa.string()), ("event_type", pa.string()),
        ("source", pa.string()), ("RESTYPE", pa.string()), ("RESTYPE1", pa.string()),
        ("source_date", pa.string()), ("linked", pa.bool_()), ("COST", pa.float64()),
        ("COSTMOSTRECENTASOF", pa.string()), ("ACTIVE", pa.string()), ("CHANGEC1", pa.string())])


def terminal_schema():
    import pyarrow as pa
    return pa.schema([("CERT",pa.int64()),("first_report",pa.date32()),("last_report",pa.date32()),("reports",pa.int64()),
                      ("audit_class",pa.string()),("failure_dates",pa.int64()),("assistance_records",pa.int64()),
                      ("uncertain_records",pa.int64()),("dated_ends",pa.int64())])


def save_csv(path, rows, fields=None):
    if fields is None:
        if not rows:
            raise IntegrityError(f"No audit rows: {path.name}")
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def summarize_fields(rows):
    totals = defaultdict(Counter)
    for r in rows:
        totals[r["field"]].update({k:r[k] for k in ("rows","missing","zeros","negative","nonfinite")})
    return [dict(field=f, **dict(v), missing_pct=100*v["missing"]/v["rows"], zero_pct=100*v["zeros"]/v["rows"]) for f,v in totals.items()]


def event_diagnostics(events, bounds):
    """Event records are not independent failed-bank counts."""
    counts = Counter((e["source"], e["event_type"], e["date_status"], bool(e["linked"])) for e in events)
    event_summary = [dict(source=s,event_type=t,date_status=d,linked=l,records=n) for (s,t,d,l),n in sorted(counts.items())]
    fin_counts = Counter(e["FIN"] for e in events if e["source"]=="failures" and e["FIN"] is not None)
    failures = [e for e in events if e["event_type"]=="failure"]
    by_date = {(e["CERT"],e["event_date"]) for e in failures if e["linked"] and e["date_status"]=="valid" and e["event_date"]<=STUDY_END}
    info = dict(source_failure_records=sum(e["source"]=="failures" for e in events),
        assistance_records=sum(e["event_type"]=="assistance" for e in events), failure_type_records=len(failures),
        unknown_resolution_records=sum(e["event_type"]=="unknown_resolution" for e in events),
        unlinked_failure_type_records=sum(not e["linked"] for e in failures),
        missing_cert_failure_source_records=sum(e["source"]=="failures" and e["CERT"] is None for e in events),
        nonpositive_cert_failure_source_records=sum(e["source"]=="failures" and e["CERT"] is not None and e["CERT"]<=0 for e in events),
        unknown_resolution_values=sorted({str(e["RESTYPE"]) for e in events if e["event_type"]=="unknown_resolution"}),
        invalid_or_missing_failure_dates=sum(e["date_status"]!="valid" for e in failures),
        duplicated_fin_values=sum(n>1 for n in fin_counts.values()),
        linked_distinct_failure_cert_dates_through_study_end=len(by_date),
        linked_distinct_failure_certs_through_study_end=len({c for c,d in by_date}),
        failure_dates_before_first_report=sum(d<bounds[c][0] for c,d in by_date),
        failure_dates_before_last_report=sum(d<bounds[c][1] for c,d in by_date),
        end_dates_after_study_end=sum(e["event_type"]=="institution_end" and e["event_date"] is not None and e["event_date"]>STUDY_END for e in events))
    return event_summary, info


def make_figures(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    coverage = pd.read_csv(output/"quarter_coverage.csv")
    fields = pd.read_csv(output/"field_summary.csv").sort_values("missing_pct",ascending=True)
    capital = pd.read_csv(output/"capital_audit.csv")
    fig, ax=plt.subplots(figsize=(8,3))
    ax.plot(pd.to_datetime(coverage.quarter), coverage.rows, linewidth=1.6, color="#35667c")
    ax.set(ylabel="Observed banks",xlabel="Reporting quarter"); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(output/"coverage.png",dpi=160); plt.close(fig)
    fig, ax=plt.subplots(figsize=(7,6))
    ax.barh(fields.field,fields.missing_pct,color="#35667c"); ax.set(xlabel="Source missing (%)",xlim=(0,100)); ax.tick_params(axis="y",labelsize=8)
    fig.tight_layout(); fig.savefig(output/"missingness.png",dpi=160); plt.close(fig)
    by_q = pd.read_csv(output/"field_by_quarter.csv") if (output/"field_by_quarter.csv").exists() else None
    if by_q is not None:
        by_q["year"]=pd.to_datetime(by_q.quarter).dt.year
        annual=by_q.groupby(["field","year"])[["missing","rows"]].sum()
        heat=(100*annual.missing/annual.rows).unstack("year")
        fig,ax=plt.subplots(figsize=(10,6))
        im=ax.imshow(heat.to_numpy(),aspect="auto",vmin=0,vmax=100,cmap="Blues")
        ax.set_yticks(range(len(heat.index)),heat.index,fontsize=7)
        step=max(1,len(heat.columns)//10)
        ax.set_xticks(range(0,len(heat.columns),step),heat.columns[::step],rotation=45,fontsize=8)
        ax.set(xlabel="Reporting year"); fig.colorbar(im,ax=ax,label="Missing (%)")
        fig.tight_layout(); fig.savefig(output/"historical_missingness.png",dpi=160); plt.close(fig)
    fig, axes=plt.subplots(2,1,figsize=(8,4.8),sharex=True)
    for ax,field in zip(axes,CAPITAL_FIELDS):
        c=capital[capital.field==field]; x=pd.to_datetime(c.quarter)
        ax.plot(x,100*c.zeros/c.rows,label="Reported zero",color="#35667c")
        ax.plot(x,100*c.missing/c.rows,label="Missing",color="#a76a38")
        ax.axvline(pd.Timestamp("2020-01-01"),color="grey",linestyle="--",linewidth=1)
        ax.set(ylabel="Rows (%)",title=field); ax.legend(fontsize=8,loc="best")
    fig.tight_layout(); fig.savefig(output/"capital_reporting.png",dpi=160); plt.close(fig)


def build(source, output):
    """Read each verified quarter once, preserve every observed bank-quarter."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents or source==output:
        raise IntegrityError("Use a fresh Silver directory separate from Bronze")
    verified=verify_bronze(source)
    original=read_json(source/"manifest.json")
    output.mkdir(parents=True)
    history={}; dates=defaultdict(list); bounds={}; index={}
    audits=[]; flags=[]; flows=[]; caps=[]; availability=[]; coverage=[]
    for q in quarter_ends():
        entry=original["financials"][str(q)]
        raw=pq.ParquetFile(source/entry["path"]).read()
        df,a,f,y,c=clean_quarter(raw.to_pandas(date_as_object=True),q,history)
        table=to_panel_table(df)
        rel="panel/"+partition_path(q).removeprefix("financials/")
        parquet_commit(table,output/rel)
        index[str(q)]={"path":rel,"rows":table.num_rows,"sha256":sha256(output/rel)}
        audits.extend(a); flags.extend(f); flows.extend(y); caps.extend(c); availability.extend(availability_counts(q,df))
        coverage.append(dict(quarter=str(q),rows=len(df),source_rows=raw.num_rows,asset_to_null=int(df.DQ_ASSET_NONPOSITIVE.sum()),
                             any_quality_flag=int(df[["DQ_"+r for r in RULES]].any(axis=1).sum()),
                             gaps=int(df.DQ_REPORTING_GAP.sum()),new_observed=int(df.REPORTS_TO_DATE.eq(1).sum())))
        for cert in df.CERT:
            dates[cert].append(q)
        print(f"{q}: {len(df):,} observed rows",flush=True)
    bounds={cert:(d[0],d[-1],len(d)) for cert,d in dates.items()}
    failures=pq.ParquetFile(source/"failures.parquet").read().to_pylist()
    institutions=pq.ParquetFile(source/"institutions.parquet").read().to_pylist()
    events=event_records(failures,institutions,set(bounds)); terminal=terminal_audit(bounds,events)
    parquet_commit(pa.Table.from_pylist(events,schema=event_schema()),output/"event_ledger.parquet")
    parquet_commit(pa.Table.from_pylist(terminal,schema=terminal_schema()),output/"terminal_audit.parquet")
    event_summary,event_info=event_diagnostics(events,bounds)
    save_csv(output/"field_by_quarter.csv",audits); save_csv(output/"field_summary.csv",summarize_fields(audits))
    save_csv(output/"quality_flags.csv",flags); save_csv(output/"income_semantics_audit.csv",flows)
    save_csv(output/"capital_audit.csv",caps); save_csv(output/"availability_sensitivity.csv",availability)
    save_csv(output/"quarter_coverage.csv",coverage); save_csv(output/"event_summary.csv",event_summary)
    annual=Counter((e["event_date"].year,e["event_type"]) for e in events if e["source"]=="failures" and e["date_status"]=="valid")
    save_csv(output/"event_year_counts.csv",[dict(year=y,event_type=t,records=n) for (y,t),n in sorted(annual.items())], fields=["year","event_type","records"])
    save_csv(output/"terminal_summary.csv",[dict(audit_class=k,institutions=v) for k,v in sorted(Counter(r["audit_class"] for r in terminal).items())])
    save_csv(output/"quality_policy.csv",[dict(rule=k,severity=v[0],treatment=v[1]) for k,v in RULES.items()])
    # Real-data diagnostic; not a label table and not a model-selection criterion.
    from datetime import date
    q=date(2006,12,31); h=shift_quarter(q,4); ev=index_events(events).get(7213,[]); reports=dates.get(7213,[])
    case={"CERT":7213,"report_quarter":str(q),"horizon_end":str(h),"observed_report":q in reports,
          "later_assistance_count":sum(e["event_type"]=="assistance" and e["event_date"] is not None and e["event_date"]>h for e in ev)}
    if q in reports:
        before=horizon_outcome(q,[e for e in ev if e["event_date"] is not None and e["event_date"]<=h], [r for r in reports if r<=h])
        full=horizon_outcome(q,ev,reports)
        case.update(truncated_result=list(before),full_result=list(full),invariant=before==full)
        if before!=full: raise IntegrityError("Horizon invariance violated")
    write_json(output/"event_diagnostics.json",event_info); write_json(output/"horizon_case.json",case)
    make_figures(output)
    summary={"source_rows":verified["row_counts"]["financials"],"panel_rows":sum(e["rows"] for e in index.values()),
             "quarters":len(index),"distinct_certs":len(bounds),"asset_to_null":sum(r["asset_to_null"] for r in coverage),
             "rows_removed":0,"gap_rows":sum(r["gaps"] for r in coverage),"event_ledger_rows":len(events),
             "bronze_failure_rows":verified["row_counts"]["failures"],"bronze_institution_rows":verified["row_counts"]["institutions"],
             "institution_cert_linked":len(set(bounds)&{r["CERT"] for r in institutions}),
             "terminal_audit_rows":len(terminal),"event_diagnostics":event_info,"horizon_case":case,
             "availability_assumption_days":90,"historical_vintages_verified":False}
    if summary["source_rows"]!=summary["panel_rows"] or event_info["source_failure_records"]!=summary["bronze_failure_rows"]:
        raise IntegrityError("Source-to-Silver reconciliation failed")
    write_json(output/"summary.json",summary)
    files={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    manifest={"state":"VALIDATING","policy_version":POLICY_VERSION,"created_utc":utc_now(),"code_revision":code_revision(),
              "python":platform.python_version(),"source_manifest_sha256":verified["manifest_sha256"],
              "panel":index,"files":files,"summary":summary}
    write_json(output/"manifest.json",manifest)
    verify_panel(source,output,allow_validating=True)
    manifest["state"]="VERIFIED"
    write_json(output/"manifest.json",manifest)
    return manifest


def verify_panel(source, output, *, allow_validating=False):
    """Independently reconcile every Silver partition to its Bronze input."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    source,output=Path(source).resolve(),Path(output).resolve()
    m=read_json(output/"manifest.json")
    if m["state"] not in ({"VERIFIED", "VALIDATING"} if allow_validating else {"VERIFIED"}) or m["policy_version"]!=POLICY_VERSION or m["source_manifest_sha256"]!=sha256(source/"manifest.json"):
        raise IntegrityError("Silver lineage or policy mismatch")
    actual={p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}-{ "manifest.json" }
    if actual!=set(m["files"]): raise IntegrityError("Silver file inventory mismatch")
    for rel,digest in m["files"].items():
        if sha256(output/rel)!=digest: raise IntegrityError(f"Modified Silver artifact: {rel}")
    if set(m["panel"])!={str(q) for q in quarter_ends()}: raise IntegrityError("Incomplete Silver calendar")
    raw_index=read_json(source/"manifest.json")["financials"]
    history={}; n=0; certs=set()
    for q in quarter_ends():
        e=m["panel"][str(q)]; raw_e=raw_index[str(q)]
        if sha256(source/raw_e["path"])!=raw_e["sha256"]: raise IntegrityError("Bronze partition changed")
        raw=pq.ParquetFile(source/raw_e["path"]).read()
        table=pq.ParquetFile(output/e["path"]).read()
        expected=to_panel_table(clean_quarter(raw.to_pandas(date_as_object=True),q,history)[0])
        if table.schema!=panel_schema() or not table.equals(expected) or table.num_rows!=e["rows"]:
            raise IntegrityError(f"Silver value/row reconciliation failed: {q}")
        certs.update(table["CERT"].to_pylist()); n+=table.num_rows
    if n!=m["summary"]["source_rows"] or n!=m["summary"]["panel_rows"] or len(certs)!=m["summary"]["distinct_certs"]:
        raise IntegrityError("Silver count reconciliation failed")
    # Verify reference ledgers against the source and observed keys independently.
    reference=read_json(source/"manifest.json")["references"]
    for name in ("failures","institutions"):
        if sha256(source/reference[name]["path"])!=reference[name]["sha256"]: raise IntegrityError("Bronze reference changed")
    events=event_records(pq.ParquetFile(source/"failures.parquet").read().to_pylist(),pq.ParquetFile(source/"institutions.parquet").read().to_pylist(),certs)
    calendar_index={q.year*4+q.month//3-1:q for q in quarter_ends()}
    bounds={c:(v[2],calendar_index[v[0]],v[3]) for c,v in history.items()}
    for name,rows,s in (("event_ledger",events,event_schema()),("terminal_audit",terminal_audit(bounds,events),terminal_schema())):
        if not pq.ParquetFile(output/(name+".parquet")).read().equals(pa.Table.from_pylist(rows,schema=s)):
            raise IntegrityError(f"Reference reconciliation failed: {name}")
    return m


def resolve_panel(workspace):
    workspace=Path(workspace).resolve(); p=read_json(workspace/"active_panel.json")
    root=(workspace/p["path"]).resolve()
    if root.parent!=workspace/"silver" or sha256(root/"manifest.json")!=p["manifest_sha256"]:
        raise IntegrityError("Invalid Silver pointer")
    source=resolve_pointer(workspace)
    m=verify_panel(source,root)
    return root,m


def run(workspace):
    workspace=Path(workspace).resolve()
    if (workspace/"pending_source.json").exists(): raise IntegrityError("Finish acquisition before Part 2")
    source=resolve_pointer(workspace)
    output=workspace/"silver"/run_id(); m=build(source,output)
    if resolve_pointer(workspace)!=source: raise IntegrityError("Active Bronze source changed during run")
    write_json(workspace/"active_panel.json",{"path":output.relative_to(workspace).as_posix(),"manifest_sha256":sha256(output/"manifest.json")})
    return output,m
