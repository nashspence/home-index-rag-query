from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime
import timefhuman.main as tfm
from app.pipeline import _parse_date, _geocode


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


def test_parse_after(monkeypatch):
    fixed = datetime(2020, 1, 10, 12, 0, 0)
    monkeypatch.setattr(tfm.DEFAULT_CONFIG, "now", fixed)
    rng = _parse_date("after jan 5")
    assert rng == (datetime(2020, 1, 5).timestamp(), None)


def test_parse_before(monkeypatch):
    fixed = datetime(2020, 1, 10, 12, 0, 0)
    monkeypatch.setattr(tfm.DEFAULT_CONFIG, "now", fixed)
    rng = _parse_date("before jan 5")
    assert rng == (None, datetime(2020, 1, 5).timestamp())


def test_parse_on(monkeypatch):
    fixed = datetime(2020, 1, 10, 12, 0, 0)
    monkeypatch.setattr(tfm.DEFAULT_CONFIG, "now", fixed)
    rng = _parse_date("on jan 5")
    assert rng == (
        datetime(2020, 1, 5).timestamp(),
        datetime(2020, 1, 5).timestamp() + 24 * 60 * 60,
    )


def test_geocode_success(monkeypatch):
    class Dummy:
        latitude = 1.23
        longitude = 4.56

    class DummyLocator:
        def geocode(self, name):
            return Dummy()

    import app.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "_geolocator", None, raising=False)
    monkeypatch.setattr("app.pipeline.Nominatim", lambda user_agent: DummyLocator())
    lat, lon = _geocode("somewhere")
    assert lat == 1.23 and lon == 4.56


def test_geocode_failure(monkeypatch):
    class DummyLocator:
        def geocode(self, name):
            return None

    import app.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "_geolocator", None, raising=False)
    monkeypatch.setattr("app.pipeline.Nominatim", lambda user_agent: DummyLocator())
    lat, lon = _geocode("unknown")
    assert lat is None and lon is None


def test_query_pipeline_radius_miles(monkeypatch):
    from app.pipeline import query_pipeline, FileDocument
    import app.pipeline as pipeline_module

    class DummyRunnable:
        def __or__(self, other):
            return self

        def invoke(self, _):
            return FileDocument(location="loc", radius_miles=1.0)

    monkeypatch.setattr(pipeline_module, "PROMPT", DummyRunnable())
    monkeypatch.setattr(pipeline_module, "load_llm", lambda *a, **kw: DummyRunnable())
    monkeypatch.setattr(pipeline_module, "JsonOutputParser", lambda *a, **kw: DummyRunnable())
    monkeypatch.setattr(pipeline_module, "_geocode", lambda name: (1.0, 2.0))

    captured = {}

    def dummy_search_index(index, query, limit=5, **params):
        captured.update(params)
        return []

    monkeypatch.setattr(pipeline_module, "search_index", dummy_search_index)

    query_pipeline("q")

    assert captured.get("filter") == "_geoRadius(1.0, 2.0, 1609)"
