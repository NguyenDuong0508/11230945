"""FDIC BankFind API retrieval; requests are independently testable."""
from __future__ import annotations

import hashlib
import json
import time

import requests

from .contracts import FIELDS, IntegrityError, utc_now

BASE = "https://api.fdic.gov/banks"


class FDICClient:
    def __init__(self, session=None, page_size=10_000, attempts=6, sleep=time.sleep):
        if not 1 <= page_size <= 10_000 or attempts < 1:
            raise ValueError("Invalid page size or retry limit")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "NEU-BFT65B-FDIC-Research/1.0", "Accept": "application/json"})
        self.page_size, self.attempts, self.sleep = page_size, attempts, sleep

    def request(self, endpoint, params):
        for attempt in range(self.attempts):
            try:
                response = self.session.get(f"{BASE}/{endpoint}", params=params, timeout=(20, 180))
                if response.status_code == 200:
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise IntegrityError("Invalid API JSON object")
                    self.sleep(0.25)
                    return payload
                if response.status_code not in {408, 425, 429, 500, 502, 503, 504}:
                    raise IntegrityError(f"FDIC HTTP {response.status_code}: {endpoint}, {params}")
                why = f"HTTP {response.status_code}"
                retry_after = response.headers.get("Retry-After", "")
                delay = min(60, int(retry_after)) if retry_after.isdecimal() else min(60, 2**attempt)
            except (requests.RequestException, json.JSONDecodeError) as exc:
                why, delay = str(exc), min(60, 2**attempt)
            if attempt + 1 == self.attempts:
                raise IntegrityError(f"FDIC request exhausted retries: {endpoint}: {why}")
            self.sleep(delay)
        raise AssertionError("unreachable")

    def fetch(self, endpoint, quarter=None):
        if endpoint not in FIELDS or (endpoint == "financials") != (quarter is not None):
            raise ValueError("Financials requires a quarter; reference tables are unfiltered")
        params = {"fields": ",".join(FIELDS[endpoint]), "limit": self.page_size,
                  "sort_by": "FAILDATE" if endpoint == "failures" else "CERT", "sort_order": "ASC", "format": "json"}
        if quarter is not None:
            params["filters"] = f"REPDTE:{quarter:%Y%m%d}"
        rows, pages, total, seen = [], [], None, set()
        started = utc_now()
        while True:
            offset = len(rows)
            payload = self.request(endpoint, {**params, "offset": offset})
            try:
                new_total = payload["meta"]["total"]
                if isinstance(new_total, bool) or not str(new_total).isdigit():
                    raise ValueError("Non-integer total")
                new_total = int(new_total)
                raw = payload["data"]
                if not isinstance(raw, list):
                    raise ValueError("Invalid page array")
                page = [r["data"] for r in raw]
                if not all(isinstance(r, dict) for r in page):
                    raise ValueError("Invalid page record")
            except (KeyError, TypeError, ValueError) as exc:
                raise IntegrityError("Malformed FDIC response envelope") from exc
            if new_total <= 0 or (total is not None and new_total != total):
                raise IntegrityError("Empty response or changing API total")
            total = new_total
            if endpoint == "failures" and total > self.page_size:
                raise IntegrityError("Failures exceeds one page; stable event pagination requires a separate verification")
            if not page or len(page) > self.page_size or offset + len(page) > total:
                raise IntegrityError("Unexpected empty or oversized page")
            digest = hashlib.sha256(json.dumps(page, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if digest in seen:
                raise IntegrityError("Repeated page")
            seen.add(digest)
            pages.append({"offset": offset, "rows": len(page), "api_total": total, "response_data_sha256": digest})
            rows.extend(page)
            if len(rows) == total:
                break
            if len(page) < self.page_size:
                raise IntegrityError("Short page before declared API total")
        return rows, {"endpoint": endpoint, "parameters": params, "api_total": total, "pages": pages,
                      "started_utc": started, "finished_utc": utc_now()}
