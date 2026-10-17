from data.ae_generator import PATTERNS, generate
from pata.schema import Normalised, TriageInput


def test_deterministic_for_seed():
    assert generate(200, seed=3) == generate(200, seed=3)
    assert generate(200, seed=3) != generate(200, seed=4)


def test_every_row_validates():
    for x in generate(500, seed=11):
        TriageInput.model_validate(x["input"])
        Normalised.model_validate(x["truth"])
        assert x["input"]["market"] == "AE"
        assert x["truth"]["postcode"] is None
        assert x["intended_decision"] in {"DELIVERABLE", "CALL_FIRST", "LIKELY_FAIL"}


def test_test_pool_names_never_in_train_pool():
    rows = generate(3500, seed=7)
    train = {n for x in rows if x["pool"] == "train" for n in x["_names"]}
    test = {n for x in rows if x["pool"] == "test" for n in x["_names"]}
    assert test, "expected some held-out rows"
    assert not train & test


def test_every_pattern_present():
    assert {x["pattern"] for x in generate(500, seed=1)} == set(PATTERNS)


def test_truth_fields_appear_in_raw_text():
    for x in generate(500, seed=5):
        raw = x["input"]["raw_address"]
        for fld in ("area", "street", "building", "unit", "landmark"):
            assert x["truth"][fld] in raw, (fld, x)
        if x["truth"]["makani"]:
            assert x["truth"]["makani"][:5] in raw
        if x["truth"]["po_box"]:
            assert x["truth"]["po_box"] in raw
