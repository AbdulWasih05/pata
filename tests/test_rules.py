import pytest

from pata.rules import extract_features, triage
from pata.schema import Decision, Market, Payment, ReasonCode, TriageInput, TriageOutput

D, C, F = Decision.DELIVERABLE, Decision.CALL_FIRST, Decision.LIKELY_FAIL
R = ReasonCode


def run(addr, market="AE", phone=True, payment="PREPAID", pin=None, hint=None) -> TriageOutput:
    return triage(
        TriageInput(
            market=Market(market),
            raw_address=addr,
            phone_present=phone,
            payment=Payment(payment),
            declared_pincode=pin,
            city_hint=hint,
        )
    )


# ---------------------------------------------------------------- UAE


def test_ae_villa_street_area_is_deliverable():
    out = run("Villa 12, Street 7, Al Barsha 2, Dubai")
    assert out.decision == D
    assert out.reason_codes == [R.OK]
    assert out.normalised.admin1 == "Dubai"
    assert out.normalised.area == "Al Barsha 2"
    assert out.call_script is None


def test_ae_flat_and_named_building_is_deliverable():
    out = run("Flat 1203, Marina Pinnacle Tower, Dubai Marina")
    assert out.decision == D
    assert "1203" in out.normalised.unit
    assert out.normalised.building == "Marina Pinnacle Tower"


def test_ae_named_building_without_flat_is_missing_unit():
    out = run("Marina Pinnacle Tower, Dubai Marina")
    assert out.decision == C
    assert R.MISSING_UNIT in out.reason_codes
    assert out.call_script and "flat" in out.call_script


def test_ae_landmark_only():
    out = run("behind Lulu Hypermarket, Al Barsha, Dubai")
    assert out.decision == C
    assert R.LANDMARK_ONLY in out.reason_codes
    assert out.normalised.landmark == "behind Lulu Hypermarket"


def test_landmark_building_is_not_customer_building():
    out = run("near Al Noor Building, Al Nahda 2, Dubai")
    assert out.decision == C
    assert R.LANDMARK_ONLY in out.reason_codes
    assert out.normalised.building == ""


def test_ae_landmark_only_without_phone_escalates():
    out = run("near Al Noor mosque, Deira", phone=False)
    assert out.decision == F
    assert R.NO_PHONE in out.reason_codes


def test_ae_nothing_identifiable_fails():
    out = run("Dubai, UAE, deliver fast")
    assert out.decision == F
    assert R.NO_BUILDING_ID in out.reason_codes


def test_ae_po_box_only_fails():
    out = run("PO Box 12345, Deira, Dubai")
    assert out.decision == F
    assert R.PO_BOX_ONLY in out.reason_codes
    assert out.normalised.po_box == "12345"


def test_ae_makani_counts_as_building():
    out = run("Makani 12345 67890, Al Quoz 3, Dubai, warehouse gate")
    assert out.normalised.makani == "1234567890"
    assert R.NO_BUILDING_ID not in out.reason_codes


def test_ae_ambiguous_al_nahda():
    out = run("Villa 4, Street 12, Al Nahda")
    assert out.decision == C
    assert R.AMBIGUOUS_AREA in out.reason_codes


def test_ae_ambiguity_resolved_by_city_hint():
    out = run("Villa 4, Street 12, Al Nahda", hint="Sharjah")
    assert out.decision == D


def test_ae_khalifa_city_needs_sector():
    assert R.AMBIGUOUS_AREA in run("Villa 33, Street 5, Khalifa City").reason_codes
    assert run("Villa 33, Street 5, Khalifa City A, Abu Dhabi").decision == D


def test_ae_call_before_coming():
    out = run("Villa 9, Street 3, Mirdif, call before coming")
    assert out.decision == C
    assert R.CALL_REQUESTED in out.reason_codes


def test_ae_whatsapp_location():
    out = run("Flat 5, Al Fardan Building, Al Majaz, will share location on WhatsApp")
    assert R.CALL_REQUESTED in out.reason_codes


def test_ae_mixed_arabic_english():
    out = run("شقة 5 بناية الفردوس near metro, Al Nahda Sharjah")
    assert out.decision == C
    assert R.MIXED_SCRIPT in out.reason_codes


def test_ae_arabizi():
    out = run("fela 7, shar3 12, wara el masjed, Mirdif")
    assert R.MIXED_SCRIPT in out.reason_codes


def test_ae_uae_mobile_not_read_as_makani():
    out = run("Villa 12, Street 7, Al Barsha 2, Dubai, 0501234567")
    assert out.normalised.makani is None
    assert out.decision == D


def test_cod_weak_address_calls_first():
    out = run("Villa 12, Mirdif", payment="COD")
    assert out.decision == C
    assert R.COD_HIGH_RISK in out.reason_codes


def test_cod_strong_address_is_deliverable():
    out = run("Villa 12, Street 7, Al Barsha 2, Dubai", payment="COD")
    assert out.decision == D


def test_too_short_fails():
    out = run("Dubai")
    assert out.decision == F
    assert R.TOO_SHORT in out.reason_codes


# ---------------------------------------------------------------- India


def test_in_full_address_deliverable():
    out = run(
        "Plot No. 12, Sector 5, IMT Manesar, Gurgaon, Haryana 122050", market="IN"
    )
    assert out.decision == D
    assert out.normalised.postcode == "122050"
    assert out.normalised.admin1 == "Haryana"
    assert out.normalised.makani is None


def test_in_pincode_mismatch_fails():
    out = run("H.No 45, MG Road, Bengaluru 560001", market="IN", pin="110001")
    assert out.decision == F
    assert R.PINCODE_MISMATCH in out.reason_codes


def test_in_pincode_spaced_and_matching():
    out = run("H.No 45, MG Road, Bengaluru 560 001", market="IN", pin="560001")
    assert R.PINCODE_MISMATCH not in out.reason_codes
    assert out.normalised.postcode == "560001"


def test_in_phone_not_read_as_pincode():
    f = extract_features(
        TriageInput(market="IN", raw_address="12, MG Road, Pune 411001, ph 9876543210")
    )
    assert f.text_pins == ["411001"]


def test_in_no_pincode_is_ambiguous():
    out = run("Flat 3B, Shanti Apartments, MG Road, Pune", market="IN")
    assert out.decision == C
    assert R.AMBIGUOUS_AREA in out.reason_codes


def test_in_flat_code_counts_as_unit():
    out = run("B-204, Shanti Apartments, Kothrud, Pune 411038", market="IN")
    assert out.decision == D


def test_in_landmark_hinglish():
    out = run("hanuman mandir ke paas, Kothrud, Pune 411038", market="IN")
    assert out.decision == C
    assert R.LANDMARK_ONLY in out.reason_codes


def test_in_village_is_call_first():
    out = run("Village Rampur, Post Khurd, Tehsil Sadar, Distt Meerut, UP 250001", market="IN")
    assert out.decision == C
    assert R.NO_BUILDING_ID in out.reason_codes


def test_in_nothing_identifiable_fails():
    out = run("Bandra West, Mumbai 400050", market="IN")
    assert out.decision == F


def test_in_devanagari_mixed():
    out = run("H.No 12, गली नंबर 4, Laxmi Nagar, Delhi 110092", market="IN")
    assert R.MIXED_SCRIPT in out.reason_codes


def test_in_call_me_hinglish():
    out = run("H.No 12, Gali 4, Laxmi Nagar, Delhi 110092, aane se pehle call karna", market="IN")
    assert R.CALL_REQUESTED in out.reason_codes


# ---------------------------------------------------------------- contract


@pytest.mark.parametrize(
    "addr,market",
    [
        ("x", "AE"),
        ("near the mosque", "AE"),
        ("Flat 1203, Marina Pinnacle Tower, Dubai Marina, call me", "AE"),
        ("PO Box 1, Dubai", "AE"),
        ("123 " * 200, "IN"),
        ("गली नंबर 4 दिल्ली", "IN"),
    ],
)
def test_output_always_valid(addr, market):
    out = run(addr, market=market, payment="COD", phone=False)
    TriageOutput.model_validate(out.model_dump())
    assert len(out.driver_note) <= 120


def test_deterministic():
    a = run("Villa 4, Street 12, Al Nahda", payment="COD")
    b = run("Villa 4, Street 12, Al Nahda", payment="COD")
    assert a == b
