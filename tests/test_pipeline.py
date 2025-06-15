from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime
import timefhuman.main as tfm
from app.pipeline import _parse_date


def test_parse_iso_date():
    ts = _parse_date("2020-01-02T03:04:05")
    assert ts == datetime.fromisoformat("2020-01-02T03:04:05").timestamp()


def test_parse_human_date(monkeypatch):
    fixed = datetime(2020, 1, 10, 12, 0, 0)
    monkeypatch.setattr(tfm.DEFAULT_CONFIG, "now", fixed)
    ts = _parse_date("yesterday")
    assert ts == datetime(2020, 1, 9).timestamp()


def test_parse_human_between(monkeypatch):
    fixed = datetime(2020, 1, 10, 12, 0, 0)
    monkeypatch.setattr(tfm.DEFAULT_CONFIG, "now", fixed)
    rng = _parse_date("between jan 1 and jan 31")
    assert rng == (
        datetime(2020, 1, 1).timestamp(),
        datetime(2020, 1, 31).timestamp(),
    )
