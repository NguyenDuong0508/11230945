from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from fdic_ews.contracts import (START, END, FIELDS, IntegrityError, number, normalize,
                                quarter_ends, report_date, restore_schema, sha256, write_json, read_json)


def test_calendar():
    quarters = quarter_ends()
    assert len(quarters) == 170 and quarters[0] == START and quarters[-1] == END
    assert len(set(quarters)) == 170


@pytest.mark.parametrize("start,end", [(date(1984, 1, 1), END), (END, START)])
def test_invalid_calendar(start, end):
    with pytest.raises(ValueError):
        quarter_ends(start, end)


@pytest.mark.parametrize("value", ["20060331", 20060331, "2006-03-31", "03/31/2006"])
def test_reporting_dates(value):
    assert report_date(value) == date(2006, 3, 31)


@pytest.mark.parametrize("value", [None, "", "20060230", "2006-13-31"])
def test_invalid_reporting_dates(value):
    with pytest.raises(IntegrityError):
        report_date(value)


def test_numeric_preservation():
    assert number("0.0") == 0
    assert number("-1.25") == -1.25
    assert number(None) is None and number("") is None
    assert number("9007199254740993", integer=True) == 9007199254740993


@pytest.mark.parametrize("value", [True, "NaN", "inf", "1.2.3", "1,2"])
def test_reject_invalid_number(value):
    with pytest.raises(IntegrityError):
        number(value)


@pytest.mark.parametrize("value", ["0.1", "9223372036854775808"])
def test_required_integer_precision(value):
    with pytest.raises(IntegrityError):
        number(value, integer=True)


def test_financial_normalization():
    q = date(2006, 12, 31)
    rows, quality = normalize("financials", [{"CERT": 7213, "REPDTE": "20061231", "ASSET": "0", "ROA": None}], q)
    assert rows[0]["ASSET"] == 0 and rows[0]["ROA"] is None
    assert quality["absent_key_counts"]["DEP"] == 1 and quality["absent_key_counts"]["ROA"] == 0
    assert set(rows[0]) == set(FIELDS["financials"])


@pytest.mark.parametrize("rows", [[{"CERT": None, "REPDTE": "20061231"}],
    [{"CERT": 7213, "REPDTE": "20060930"}], [{"CERT": 7213, "REPDTE": "20061231"}]*2])
def test_bad_financial_keys(rows):
    with pytest.raises(IntegrityError):
        normalize("financials", rows, date(2006,12,31))


def test_assistance_and_ambiguous_reference_retained():
    rows, audit = normalize("failures", [{"CERT": 7213, "FAILDATE": "20081123", "RESTYPE": "ASSISTANCE"},
                                         {"CERT": None, "FAILDATE": "unresolved", "RESTYPE": None},
                                         {"CERT": 7213, "RESTYPE": "FAILURE"}])
    assert len(rows) == 3 and audit["null_cert"] == 1
    assert rows[0]["RESTYPE"] == "ASSISTANCE" and rows[1]["FAILDATE"] == "unresolved"
    assert "Y_FAIL_4Q" not in rows[0]


def test_atomic_json(tmp_path):
    path = tmp_path / "new" / "manifest.json"
    write_json(path, {"a": 1})
    h = sha256(path)
    write_json(path, {"a": 2})
    assert read_json(path) == {"a": 2} and sha256(path) != h
    assert list(path.parent.iterdir()) == [path]


class TableContractDouble:
    """Test the schema-restoration guard independently of the Arrow runtime."""
    column_names = ["CERT"]
    def __init__(self, null_count=0, dtype="int64"):
        self.col = SimpleNamespace(type=dtype, null_count=null_count)
        self.casts = []
    def __getitem__(self, key):
        return self.col
    def cast(self, expected, safe):
        self.casts.append((expected, safe))
        return self


class SchemaContractDouble(list):
    names = ["CERT"]


def test_restore_schema_guard():
    expected = SchemaContractDouble([SimpleNamespace(name="CERT", type="int64", nullable=False)])
    ok = TableContractDouble()
    assert restore_schema(ok, expected) is ok and ok.casts == [(expected, True)]
    for bad in (TableContractDouble(1), TableContractDouble(dtype="string")):
        with pytest.raises(IntegrityError):
            restore_schema(bad, expected)
        assert not bad.casts
