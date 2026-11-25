from datetime import date
from types import SimpleNamespace

import pytest
import requests

from fdic_ews.api import FDICClient
from fdic_ews.contracts import IntegrityError


class Session:
    def __init__(self, responses):
        self.headers = {}
        self.responses = iter(responses)
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(status_code=item[0], headers={}, json=lambda: item[1])


def page(total, certs):
    return 200, {"meta": {"total": total}, "data": [{"data": {"CERT": c}} for c in certs]}


def client(responses, size=2):
    session = Session(responses)
    return FDICClient(session=session, page_size=size, sleep=lambda _: None, attempts=2), session


def test_pagination():
    c, session = client([page(3, [1,2]), page(3,[3])])
    rows, meta = c.fetch("financials", date(2006,12,31))
    assert [r["CERT"] for r in rows] == [1,2,3]
    assert meta["api_total"] == 3 and [p["offset"] for p in meta["pages"]] == [0,2]
    assert session.calls[0][1]["params"]["filters"] == "REPDTE:20061231"
    assert session.calls[1][1]["params"]["offset"] == 2


def test_reference_unfiltered_and_event_fields():
    c, s = client([page(2, [1,2])])
    _, m = c.fetch("failures")
    assert "filters" not in m["parameters"]
    assert "RESTYPE" in m["parameters"]["fields"] and "FIN" in m["parameters"]["fields"]


@pytest.mark.parametrize("responses", [[page(4,[1,2]), page(5,[3,4])],
    [page(4,[1,2]), page(4,[])], [page(4,[1,2]), page(4,[1,2])], [page(4,[1])],
    [(200,{"meta":{"total":2},"data":[{}]})], [(200,{"meta":{"total":True},"data":[]})]])
def test_invalid_pagination(responses):
    c, _ = client(responses)
    with pytest.raises(IntegrityError):
        c.fetch("institutions")


def test_transient_retry():
    c, s = client([(429,{}),page(1,[1])])
    assert len(c.fetch("institutions")[0]) == 1 and len(s.calls)==2


def test_timeout_retry():
    c, s = client([requests.Timeout("fixture"),page(1,[1])])
    assert len(c.fetch("institutions")[0]) == 1 and len(s.calls)==2


def test_permanent_error():
    c, s = client([(400,{})])
    with pytest.raises(IntegrityError, match="HTTP 400"):
        c.fetch("institutions")
    assert len(s.calls)==1


def test_failure_page_guard():
    c, _ = client([page(3,[1,2])])
    with pytest.raises(IntegrityError, match="one page"):
        c.fetch("failures")
