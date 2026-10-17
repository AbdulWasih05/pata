import pytest

from pata.prompt import build_messages, parse_output, target_text
from pata.rules import triage
from pata.schema import TriageInput

INP = TriageInput(market="AE", raw_address="Flat 1203, Marina Pinnacle Tower, Dubai Marina")


def test_round_trip():
    out = triage(INP)
    assert parse_output(target_text(out)) == out


@pytest.mark.parametrize(
    "wrap",
    [
        "{}",
        "```json\n{}\n```",
        "<think>\nhmm\n</think>\n\n{}",
        "Sure, here it is: {} Hope that helps.",
    ],
)
def test_parse_tolerates_wrappers(wrap):
    out = triage(INP)
    assert parse_output(wrap.format(target_text(out))) == out


def test_parse_rejects_invalid():
    with pytest.raises(ValueError):
        parse_output("no json here")
    with pytest.raises(ValueError):  # pydantic ValidationError subclasses ValueError
        parse_output('{"decision": "MAYBE"}')


def test_messages_shape():
    msgs = build_messages(INP)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "JSON keys" not in msgs[0]["content"]
    assert "JSON keys" in build_messages(INP, schema_hint=True)[0]["content"]
