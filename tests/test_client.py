import asyncio

import pytest

from pata.client import TINKER_OAI_URL, Completion, ModelClient, thinking_off
from pata.prompt import extract_json


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("MODEL_URL", "MODEL_API_KEY", "TINKER_API_KEY", "HOSTED_API_KEY", "PATA_MODEL"):
        monkeypatch.delenv(k, raising=False)


def test_thinking_off_per_host():
    assert thinking_off(TINKER_OAI_URL) == {"reasoning_effort": False}
    assert thinking_off("http://10.0.0.5:8000/v1") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert thinking_off("https://generativelanguage.googleapis.com/v1beta/openai") == {}


def test_no_url_means_rules_only():
    assert ModelClient.from_env() is None


def test_tinker_without_key_means_rules_only(monkeypatch):
    monkeypatch.setenv("MODEL_URL", TINKER_OAI_URL)
    assert ModelClient.from_env() is None


def test_tinker_key_is_picked_up(monkeypatch):
    monkeypatch.setenv("MODEL_URL", TINKER_OAI_URL)
    monkeypatch.setenv("TINKER_API_KEY", "k")
    monkeypatch.setenv("PATA_MODEL", "tinker://x/sampler_weights/final")
    c = ModelClient.from_env()
    assert c is not None and c.model == "tinker://x/sampler_weights/final"


def test_reasoning_blocks_are_skipped():
    assert extract_json('<thought>maybe {"decision": "X"}</thought>{"decision": "Y"}') == {
        "decision": "Y"
    }


class Scripted(ModelClient):
    """_once replaced by a script of (delay seconds, exception or None), one per call."""

    def __init__(self, script, hedge_ms=50):
        super().__init__("http://x/v1", None, "m", hedge_ms=hedge_ms)
        self.script, self.calls = list(script), 0

    async def _once(self, messages):
        self.calls += 1
        n = self.calls
        delay, err = self.script[n - 1]
        await asyncio.sleep(delay)
        if err:
            raise err
        return Completion(f"call{n}", "m", 1, 1, "stop", delay * 1000)


def run(coro):
    return asyncio.run(coro)


def test_fast_first_call_is_not_hedged():
    c = Scripted([(0.0, None)])
    out = run(c.complete([]))
    assert (c.calls, out.hedged) == (1, False)


def test_slow_first_call_is_hedged_and_second_wins():
    c = Scripted([(1.0, None), (0.0, None)])
    out = run(c.complete([]))
    assert (c.calls, out.hedged, out.hedge_won, out.text) == (2, True, True, "call2")
    assert out.latency_ms < 500


def test_hedge_survives_one_failure():
    c = Scripted([(0.2, None), (0.0, RuntimeError("boom"))])
    out = run(c.complete([]))
    assert (out.hedged, out.hedge_won, out.text) == (True, False, "call1")


def test_both_failing_raises():
    c = Scripted([(0.2, RuntimeError("late")), (0.0, RuntimeError("boom"))])
    with pytest.raises(RuntimeError):
        run(c.complete([]))


def test_hedging_on_by_default_only_for_tinker():
    assert ModelClient(TINKER_OAI_URL, "k", "m").hedge_ms > 0
    assert ModelClient("http://10.0.0.5:8000/v1", "k", "m").hedge_ms == 0
