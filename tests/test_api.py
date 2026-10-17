import csv
import io

import pytest
from fastapi.testclient import TestClient

from api.main import app
from pata.client import Completion
from pata.prompt import target_text
from pata.rules import triage
from pata.schema import TriageInput

ORDER = {"market": "AE", "raw_address": "Villa 12, Street 7, Al Barsha 2, Dubai"}


class StubClient:
    model = "stub"
    schema_hint = False
    max_tokens = 512

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, 0

    async def complete(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return Completion(self.reply, "stub", 300, 120, "stop", 5.0)

    async def aclose(self):
        pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("MODEL_URL", raising=False)
    with TestClient(app) as c:
        yield c


def test_rules_only_mode(client):
    assert client.get("/healthz").json()["mode"] == "rules-only"
    j = client.post("/triage", json=ORDER).json()
    assert j["source"] == "rules"
    assert j["output"]["decision"] == "DELIVERABLE"


def test_model_answer_used_when_valid(client):
    # The stub answers CALL_FIRST so we can tell the model's answer from the rules'.
    out = triage(TriageInput(market="AE", raw_address="behind Lulu, Al Barsha, Dubai"))
    app.state.client = StubClient(reply="```json\n" + target_text(out) + "\n```")
    j = client.post("/triage", json=ORDER).json()
    assert j["source"] == "model"
    assert j["output"]["decision"] == "CALL_FIRST"
    assert j["rules_decision"] == "DELIVERABLE"
    assert j["input_tokens"] == 300


@pytest.mark.parametrize(
    "stub,reason",
    [
        (StubClient(reply='{"decision": "MAYBE"}'), "schema_invalid"),
        (StubClient(reply="I think it is fine."), "schema_invalid"),
        (StubClient(error=TimeoutError()), "model_error:TimeoutError"),
    ],
)
def test_fallback_to_rules(client, stub, reason):
    app.state.client = stub
    j = client.post("/triage", json=ORDER).json()
    assert j["source"] == "rules"
    assert j["fallback_reason"].startswith(reason)
    assert j["output"]["decision"] == "DELIVERABLE"


def test_batch(client):
    orders = [ORDER, {"market": "IN", "raw_address": "Bandra West, Mumbai 400050"}]
    j = client.post("/triage/batch", json={"orders": orders}).json()
    assert [r["output"]["decision"] for r in j["results"]] == ["DELIVERABLE", "LIKELY_FAIL"]


def test_csv_round_trip(client):
    body = "raw_address,payment\nVilla 12 Street 7 Al Barsha 2 Dubai,COD\nnear the mosque,PREPAID\n"
    r = client.post("/triage/csv", files={"file": ("o.csv", body)}, data={"market": "AE"})
    assert r.status_code == 200
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == 2
    assert rows[0]["payment"] == "COD"
    assert {"decision", "driver_note", "building", "source"} <= rows[0].keys()


def test_csv_without_address_column(client):
    r = client.post("/triage/csv", files={"file": ("o.csv", "foo\n1\n")})
    assert r.status_code == 422


def test_index_served(client):
    assert "Pata" in client.get("/").text
