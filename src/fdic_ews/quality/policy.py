"""Pure temporal and event policies; independent of file formats and model fitting."""
from __future__ import annotations
import calendar
from datetime import date, datetime, timedelta
from collections import defaultdict, Counter

STUDY_END = date(2026, 6, 30)
LAST_LABEL = date(2025, 6, 30)
LAGS = (60, 90, 120, 180)


def parse_date(value):
    """Return (date, status); never coerce malformed dates to a valid event."""
    if value is None or str(value).strip() == "":
        return None, "missing"
    if isinstance(value, datetime):
        return value.date(), "valid"
    if isinstance(value, date):
        return value, "valid"
    s = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date(), "valid"
        except ValueError:
            pass
    try:
        if len(s) > 10 and s[10] in "T ":
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date(), "valid"
    except ValueError:
        pass
    return None, "invalid"


def quarter_id(d):
    if d.month not in (3, 6, 9, 12) or d.day != calendar.monthrange(d.year, d.month)[1]:
        raise ValueError("Expected calendar quarter end")
    return d.year * 4 + d.month // 3 - 1


def shift_quarter(d, n):
    year, k = divmod(quarter_id(d) + n, 4)
    month = (k + 1) * 3
    return date(year, month, calendar.monthrange(year, month)[1])


def available_at(report, lag=90):
    if lag < 0:
        raise ValueError("Negative reporting delay")
    return report + timedelta(days=lag)


def classify_restype(value):
    token = str(value or "").strip().upper()
    return {"FAILURE": "failure", "ASSISTANCE": "assistance"}.get(token, "unknown_resolution")


def event_records(failures, institutions, panel_certs):
    """Preserve each source row in a reference-only event ledger.

    ENDEFYMD closes a structural event; it is candidate exit evidence, not proof of merger or failure.
    Current ACTIVE and CHANGEC1 are retained only in the institutional audit.
    """
    result = []
    for i, r in enumerate(failures):
        day, status = parse_date(r.get("FAILDATE"))
        cert = r.get("CERT")
        linked = cert is not None and cert > 0 and cert in panel_certs
        result.append(dict(event_id=f"F{i:06d}", CERT=cert, FIN=r.get("FIN"),
                           event_date=day, date_status=status,
                           event_type=classify_restype(r.get("RESTYPE")), source="failures",
                           RESTYPE=r.get("RESTYPE"), RESTYPE1=r.get("RESTYPE1"),
                           source_date=r.get("FAILDATE"), linked=linked,
                           COST=r.get("COST"), COSTMOSTRECENTASOF=r.get("COSTMOSTRECENTASOF"),
                           ACTIVE=None, CHANGEC1=None))
    for i, r in enumerate(institutions):
        raw = r.get("ENDEFYMD")
        day, status = parse_date(raw)
        # Include malformed termination dates, but not ordinary absent end dates.
        if status == "missing":
            continue
        cert = r.get("CERT")
        result.append(dict(event_id=f"I{i:06d}", CERT=cert, FIN=None,
                           event_date=day, date_status=status, event_type="institution_end",
                           source="institutions", RESTYPE=None, RESTYPE1=None, source_date=raw,
                           linked=cert in panel_certs, COST=None, COSTMOSTRECENTASOF=None,
                           ACTIVE=r.get("ACTIVE"), CHANGEC1=r.get("CHANGEC1")))
    return result


def index_events(events):
    indexed = defaultdict(list)
    for e in events:
        if e["linked"]:
            indexed[e["CERT"]].append(e)
    return indexed


def terminal_audit(certs, events, end=STUDY_END):
    """Retrospective audit only. NEVER use its class for sample eligibility."""
    indexed = index_events(events)
    rows = []
    for cert, (first, last, n) in sorted(certs.items()):
        ev = indexed.get(cert, [])
        f = sorted({e["event_date"] for e in ev if e["event_type"] == "failure" and e["event_date"] is not None and e["event_date"] <= end})
        ends = sorted({e["event_date"] for e in ev if e["event_type"] == "institution_end" and e["event_date"] is not None and e["event_date"] <= end})
        assistance = sum(e["event_type"] == "assistance" for e in ev)
        uncertain = sum(e["date_status"] != "valid" or e["event_type"] == "unknown_resolution" for e in ev)
        if len(f) > 1:
            cls = "multiple_failure_dates"
        elif f and f[0] < last:
            cls = "failure_precedes_last_report"
        elif f:
            cls = "failure_evidence"
        elif ends and ends[-1] < last:
            cls = "end_precedes_last_report"
        elif ends:
            cls = "dated_other_end_candidate" if not uncertain else "end_with_uncertain_evidence"
        elif last >= end:
            cls = "observed_at_study_end"
        else:
            cls = "unresolved_discontinuation"
        rows.append(dict(CERT=cert, first_report=first, last_report=last, reports=n,
                         audit_class=cls, failure_dates=len(f), assistance_records=assistance,
                         uncertain_records=uncertain, dated_ends=len(ends)))
    return rows


def horizon_outcome(q, events, report_dates, *, study_end=STUDY_END, other_exit="censor", verified_exit="zero"):
    """Reference outcome policy for Part 4 (no model labels persisted in Part 2).

    Only evidence dated through the horizon can adjudicate an earlier row.
    Reporting observations at/after horizon may establish survival only if a
    report at the horizon exists; later reports alone do not resolve a gap.
    Undated/unlinked records are a dataset-level completeness limitation;
    they cannot be assigned arbitrarily to historical rows. Unknown dated
    transactions within the horizon are censored. An ENDEFYMD candidate without corroborated closure is censored by default;
    zero is an explicit sensitivity assumption, not a verified merger.
    """
    if other_exit not in {"zero", "censor"} or verified_exit not in {"zero", "censor"}:
        raise ValueError("Invalid competing-exit policy")
    h = shift_quarter(q, 4)
    if h > study_end or q > LAST_LABEL:
        return None, "immature"
    reports = {d for d in report_dates if d <= h}
    valid = [e for e in events if e.get("linked", True) and e.get("event_date") is not None and e["event_date"] <= h]
    failures = sorted({e["event_date"] for e in valid if e["event_type"] == "failure"})
    verified_ends = sorted({e["event_date"] for e in valid if e["event_type"] == "verified_nonfailure_exit"})
    ends = sorted({e["event_date"] for e in valid if e["event_type"] == "institution_end" and e["event_date"] > q})
    if any(d <= q for d in failures + verified_ends):
        return None, "not_at_risk_or_conflicting_prior_exit"
    if any(e["event_type"] == "unknown_resolution" and q < e["event_date"] for e in valid):
        return None, "unknown_in_horizon"
    if len(failures) > 1 or len(set(ends + verified_ends)) > 1:
        return None, "multiple_in_horizon_events"
    if failures:
        f = failures[0]
        if any(d > f for d in reports) or verified_ends or (ends and ends[0] < f):
            return None, "conflicting_in_horizon_evidence"
        return 1, "failure_within_horizon"
    if verified_ends:
        d = verified_ends[0]
        if any(r > d for r in reports):
            return None, "conflicting_in_horizon_evidence"
        return (0, "verified_nonfailure_exit") if verified_exit == "zero" else (None, "verified_competing_exit")
    if ends:
        d = ends[0]
        if any(r > d for r in reports):
            return None, "conflicting_in_horizon_evidence"
        return (0, "assumed_other_exit") if other_exit == "zero" else (None, "unconfirmed_competing_exit")
    if h in reports:
        return 0, "observed_at_horizon"
    return None, "incomplete_followup"


def latest_available_report(decision, report_dates, lag=90):
    """Latest source quarter available by the decision date, not future activity."""
    return max((r for r in report_dates if available_at(r, lag) <= decision), default=None)
