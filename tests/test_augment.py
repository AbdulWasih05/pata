from data.augment import augment
from data.label import label


def _rec(i: int, pool: str = "train") -> dict:
    return {
        "id": f"in-test-{i}",
        "pool": pool,
        "input": {"market": "IN", "raw_address": "Flat 12, Shanti Apartments, MG Road, "
                  "Kothrud, Pune 411038", "phone_present": True, "payment": "PREPAID",
                  "declared_pincode": None, "city_hint": None},
        "truth": {"country": "IN", "admin1": "Maharashtra", "city": "Pune", "area": "Kothrud",
                  "street": "MG Road", "building": "Shanti Apartments", "unit": "Flat 12",
                  "landmark": "", "postcode": "411038", "makani": None, "po_box": None},
    }


def test_deterministic():
    assert augment(_rec(1), seed=3) == augment(_rec(1), seed=3)


def test_truth_text_fields_still_appear_in_text():
    for i in range(300):
        out = augment(_rec(i), seed=1)
        text = out["input"]["raw_address"].lower()
        for k in ("building", "unit", "landmark"):
            v = out["truth"][k]
            assert not v or v.lower() in text, (k, v, text)
        if out["truth"]["postcode"] is None:
            assert "411038" not in text.replace(" ", "")


def test_test_pool_uses_test_landmarks():
    from data.augment import LANDMARKS

    for i in range(300):
        lm = augment(_rec(i, pool="test"), seed=1)["truth"]["landmark"]
        assert not any(t in lm for t in LANDMARKS["train"])


def test_label_output_is_valid_and_consistent():
    for i in range(100):
        row, why = label(augment(_rec(i), seed=2))
        if row is None:
            assert why
            continue
        out = row["output"]
        assert (out["call_script"] is not None) == (out["decision"] == "CALL_FIRST")
