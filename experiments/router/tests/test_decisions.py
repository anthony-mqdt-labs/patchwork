"""Rung-3 decision tier: contract, policy, and cascade wiring. No models loaded.

`mlx.core` is a hard dependency of this project (mlx-lm is the tier runtime),
so a crafted logits tensor exercises the whole micro-scorer path — including
renormalisation and the engage gate — without touching a checkpoint. The http
backend is exercised against a real loopback server, so the contract is tested
rather than assumed.
"""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx
import pytest

from darkcore import cascade, decisions, surface, verifiers
from tests.conftest import FakePool, minimal_params

VOCAB = 16
YES_ID, NO_ID = 7, 8


class FakeTokenizer:
    """Maps every yes/no surface form to its single verdict id."""

    def __init__(self, yes=YES_ID, no=NO_ID, prompt_ids=(1, 2, 3)):
        self._yes, self._no, self._prompt = yes, no, list(prompt_ids)

    def encode(self, s):
        if s in decisions.YES_FORMS:
            return [self._yes]
        if s in decisions.NO_FORMS:
            return [self._no]
        return list(self._prompt)


class FakeModel:
    """Returns one logits row. `other_logit` puts mass outside yes/no so
    renormalisation is actually exercised rather than trivially satisfied."""

    def __init__(self, yes_logit, no_logit, other_logit=-1.0):
        row = [other_logit] * VOCAB
        row[YES_ID], row[NO_ID] = yes_logit, no_logit
        self._row = mx.array(row, dtype=mx.float32)

    def __call__(self, _ids):
        return self._row.reshape(1, 1, VOCAB)


def scorer(yes_logit=math.log(3.0), no_logit=0.0, other_logit=-1.0):
    return decisions.MicroScorer(
        FakeModel(yes_logit, no_logit, other_logit), FakeTokenizer())


# ------------------------------------------------------------------- the head
def test_noul_renormalises_over_the_verdict_tokens():
    """Raw p(' Yes') is 0.328 when a third of the mass sits elsewhere; the
    decision is p(yes | the model chose yes or no) = 0.75."""
    d = scorer().noul("STATE", "Statement.")
    z = 3.0 + 1.0 + (VOCAB - 2) * math.exp(-1.0)
    assert d["p_yes"] == pytest.approx(3.0 / z, abs=1e-4)
    assert d["p_either"] == pytest.approx(4.0 / z, abs=1e-4)
    assert d["noul"] == pytest.approx(0.75, abs=1e-3)
    assert d["passes"] == 1


def test_no_single_token_verdict_form_is_unavailable_not_wrong():
    """A tokenizer that splits 'Yes' into several tokens must refuse to score
    rather than sum a partial surface form."""
    class Splitting(FakeTokenizer):
        def encode(self, s):
            if s in decisions.YES_FORMS or s in decisions.NO_FORMS:
                return [1, 2]
            return [3]

    with pytest.raises(decisions.DecisionUnavailable):
        decisions.MicroScorer(FakeModel(1.0, 0.0), Splitting())


def test_choice_is_order_independent_and_one_pass_per_option():
    s = scorer()
    opts = ["agentic", "reasoning", "emotional", "default"]
    a = s.choice("STATE", "Which class?", opts)
    b = s.choice("STATE", "Which class?", list(reversed(opts)))
    assert a["probabilities"] == b["probabilities"]
    assert a["passes"] == len(opts)
    assert abs(sum(a["probabilities"].values()) - 1.0) < 0.01


def test_score_is_the_expected_level_index():
    s = scorer()
    out = s.score("STATE", "Urgency?", ["low", "normal", "high"])
    assert 0.0 <= out["score"] <= 2.0
    assert out["passes"] == 3


def test_distribution_softmaxes_independent_probabilities():
    d = decisions._to_distribution({"a": 0.9, "b": 0.5, "c": 0.1})
    assert d["choice"] == "a"
    assert d["probabilities"]["a"] > d["probabilities"]["b"] > d["probabilities"]["c"]
    assert abs(sum(d["probabilities"].values()) - 1.0) < 0.01
    assert d["confidence"] == d["probabilities"]["a"]


# ---------------------------------------------------------------- the verifier
class ScriptedBackend:
    def __init__(self, noul, p_either=1.0):
        self._d = {"noul": noul, "p_yes": noul, "p_no": 1 - noul,
                   "p_either": p_either, "passes": 1, "forward_ms": 1.5}

    def noul(self, state, statement, template=None):
        return dict(self._d)


def arm(monkeypatch, backend):
    monkeypatch.setattr(decisions, "resolve_backend",
                        lambda *a, **k: backend)


def ctx(**thresholds):
    return {"expected": [], "thresholds": thresholds}


def test_confident_yes_passes(monkeypatch):
    arm(monkeypatch, ScriptedBackend(0.91))
    passed, d = decisions.decision_judge("q", "a", None, "T0", ctx())
    assert passed and d["rung"] == 3 and d["check"] == "decision_judge"
    assert d["decision_tokens"] == 0 and d["p_yes"] == 0.91


def test_confident_no_fails(monkeypatch):
    arm(monkeypatch, ScriptedBackend(0.12))
    passed, d = decisions.decision_judge("q", "a", None, "T0", ctx())
    assert not passed and d["engaged"]


def test_threshold_is_configurable(monkeypatch):
    """A conservative class can demand 0.9 where the default 0.5 would pass."""
    arm(monkeypatch, ScriptedBackend(0.72))
    assert decisions.decision_judge("q", "a", None, "T0", ctx())[0] is True
    assert decisions.decision_judge(
        "q", "a", None, "T0", ctx(threshold=0.9))[0] is False


def test_unengaged_model_fails_even_with_a_high_p_yes(monkeypatch):
    """p(Yes)/p(No) can look decisive while the model put almost no mass on
    either token — that is not a verdict, and rung 3 must not pass it."""
    arm(monkeypatch, ScriptedBackend(1.0, p_either=0.01))
    passed, d = decisions.decision_judge("q", "a", None, "T0", ctx())
    assert not passed and not d["engaged"]


def test_backend_failure_falls_back_to_the_llm_judge(monkeypatch):
    """A missing/broken certificate degrades to rung 4, visibly."""
    def boom(*a, **k):
        raise decisions.DecisionUnavailable("no service")
    monkeypatch.setattr(decisions, "resolve_backend", boom)
    pool = FakePool({"T1": {"answer": "no verdict word here"}})
    passed, d = decisions.decision_judge("q", "a", pool, "T0", ctx())
    assert passed is False                       # unparseable judge -> escalate
    assert d["fallback"] == "llm_judge" and d["judge_tier"] == "T1"
    assert d["decision_error"] == "DecisionUnavailable"


# ------------------------------------------------------- cascade integration
@pytest.fixture()
def armed_cascade(monkeypatch):
    """Register decision_judge in the real verifier REGISTRY with a scripted
    verdict, and drive it through cascade.run — the actual wiring."""
    def make(verdicts):
        it = iter(verdicts)
        def fn(query, answer, pool, tier, ctx_):
            ok = next(it)
            return ok, {"rung": 3, "check": "decision_judge", "backend": "micro",
                        "p_yes": 0.9 if ok else 0.1, "p_either": 1.0,
                        "threshold": 0.5, "engaged": True, "passes": 1,
                        "decision_tokens": 0, "forward_ms": 1.2, "verify_ms": 1.3}
        return fn

    def factory(verdicts):
        fn = make(verdicts)
        monkeypatch.setitem(verifiers.REGISTRY, "decision_judge", fn)
        return fn
    return factory


def test_cascade_escalates_on_a_rung3_fail(armed_cascade, log_events):
    armed_cascade([False, True])
    p = minimal_params(verifier_config={
        "default": {"rung": 3, "verifier": "decision_judge",
                    "thresholds": {"backend": "micro"}}})
    out = cascade.run(FakePool({"T0": {}, "T1": {"answer": "better"}}),
                      "q", "qh", "rid", "default", "T0", p)
    assert out["final_tier"] == "T1" and out["escalations"] == 1
    assert [a["outcome"] for a in out["attempts"]] == ["fail", "pass"]
    assert [a["rung"] for a in out["attempts"]] == [3, 3]

    ev = [e for e in log_events() if e["event"] == "verifier_result"]
    assert ev and all(e["rung"] == 3 for e in ev)
    assert ev[0]["p_yes"] == 0.1 and ev[0]["decision_tokens"] == 0
    # PII rule: numbers only, never the question or the answer
    assert not ({"query", "answer", "text", "prompt", "content"} & set(ev[0]))


def test_surface_accepts_the_new_verifier_but_still_rejects_strangers():
    cfg = surface.DEFAULTS
    candidate = {"params": {"verifier_config": {"default": {
        "rung": 3, "verifier": "decision_judge", "thresholds": {}}}}}
    merged = surface._merge(cfg["params"], candidate["params"])
    assert not [v for v in surface.validate({**cfg, "params": merged})
                if "verifier" in v]

    bad = surface._merge(cfg["params"], {"verifier_config": {"default": {
        "rung": 3, "verifier": "made_up_judge", "thresholds": {}}}})
    assert any("unknown verifier 'made_up_judge'" in v
               for v in surface.validate({**cfg, "params": bad}))


# ------------------------------------------------------------- rung-3 triage
def spy_judge(monkeypatch, verdict=True):
    calls = []

    def fake(query, answer, pool, tier, ctx_):
        calls.append(tier)
        return verdict, {"rung": 4, "check": "judge", "judge_tier": "T9",
                         "judge_tokens": 3, "verify_ms": 1.0}
    monkeypatch.setattr(verifiers, "next_tier_judge", fake)
    return calls


def test_triage_confident_pass_short_circuits_the_judge(monkeypatch):
    arm(monkeypatch, ScriptedBackend(0.95))
    calls = spy_judge(monkeypatch)
    passed, d = decisions.decision_judge("q", "a", None, "T0",
                                         ctx(bypass_threshold=0.9))
    assert passed and d["role"] == "bypass_pass" and calls == []


def test_triage_confident_fail_short_circuits_the_judge(monkeypatch):
    """A confident FAIL escalates immediately — the judge's tokens are saved
    in both directions, not just on the pass side."""
    arm(monkeypatch, ScriptedBackend(0.02))
    calls = spy_judge(monkeypatch)
    passed, d = decisions.decision_judge("q", "a", None, "T0",
                                         ctx(fail_threshold=0.1))
    assert passed is False and d["role"] == "bypass_fail" and calls == []


def test_triage_ambiguous_hands_the_decision_to_the_judge(monkeypatch):
    """The whole point: a rung-3 head must not decide the middle of its own
    distribution. Evidence is preserved alongside the judge's verdict."""
    arm(monkeypatch, ScriptedBackend(0.62))
    calls = spy_judge(monkeypatch, verdict=True)
    passed, d = decisions.decision_judge(
        "q", "a", None, "T0",
        ctx(bypass_threshold=0.9, fail_threshold=0.1))
    assert passed is True and calls == ["T0"]          # judge ran
    assert d["role"] == "triage_inconclusive"
    assert d["p_yes"] == 0.62 and d["judge_tier"] == "T9"
    assert d["decision_tokens"] == 0


def test_triage_unengaged_never_passes_even_above_bypass(monkeypatch):
    arm(monkeypatch, ScriptedBackend(1.0, p_either=0.005))
    calls = spy_judge(monkeypatch, verdict=False)
    passed, d = decisions.decision_judge("q", "a", None, "T0",
                                         ctx(bypass_threshold=0.9))
    assert passed is False and calls == ["T0"]
    assert d["role"] == "triage_inconclusive" and not d["engaged"]


def test_two_way_mode_is_the_default_and_is_labelled(monkeypatch):
    arm(monkeypatch, ScriptedBackend(0.8))
    passed, d = decisions.decision_judge("q", "a", None, "T0", ctx())
    assert passed and d["role"] == "two_way"


# ------------------------------------------------------------ server dialects
def test_response_dialects_normalise():
    q = ["team", "urgent"]
    wrapped = {"results": {"team": {"choice": "billing"},
                           "urgent": {"noul": 0.8}}}
    flat = {"team": {"choice": "billing"}, "urgent": {"noul": 0.8}}
    for body in (wrapped, flat):
        out = decisions._normalise(body, q)
        assert out["team"]["choice"] == "billing" and out["urgent"]["noul"] == 0.8
    assert decisions._normalise({"unrelated": 1}, q) == {}
    with pytest.raises(decisions.DecisionUnavailable):
        decisions._normalise(["not", "an", "object"], q)


def test_service_client_speaks_the_published_contract(monkeypatch):
    """The request body must match the documented /v1/systemone shape."""
    seen = {}

    class Resp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b'{"results": {"_verdict": {"noul": 0.77}}}'

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = __import__("json").loads(req.data.decode())
        return Resp()

    monkeypatch.setattr(decisions.urllib.request, "urlopen", fake_urlopen)
    svc = decisions.DecisionService("http://localhost:8080", model="openjev")
    out = svc.noul("STATE", "The answer resolves the question.")
    assert seen["url"] == "http://localhost:8080/v1/systemone"
    assert seen["body"]["state"] == "STATE"
    assert seen["body"]["model"] == "openjev"
    assert seen["body"]["questions"]["_verdict"]["type"] == "noul"
    assert out["noul"] == 0.77


def test_service_unreachable_is_unavailable_not_a_verdict(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(decisions.urllib.request, "urlopen", boom)
    with pytest.raises(decisions.DecisionUnavailable):
        decisions.DecisionService("http://localhost:9").noul("s", "t")


# ------------------------------------------------- loopback service (real HTTP)
class _Handler(BaseHTTPRequestHandler):
    payload: dict = {}
    status: int = 200
    last_body: dict | None = None
    hits: int = 0

    def do_POST(self):
        _Handler.hits += 1
        n = int(self.headers.get("Content-Length", 0))
        _Handler.last_body = json.loads(self.rfile.read(n).decode())
        self.send_response(_Handler.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(_Handler.payload).encode())

    def log_message(self, *a):  # keep pytest output clean
        pass


@pytest.fixture()
def fake_service():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Handler.hits, _Handler.status = 0, 200
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_http_backend_round_trip_over_loopback(fake_service):
    """The out-of-process path (Laya / OpenJev / Tiny-Jev) end to end."""
    _Handler.payload = {"results": {
        "_verdict": {"noul": 0.88},
        "_choice": {"choice": "billing", "confidence": 0.8,
                    "probabilities": {"billing": 0.8, "technical": 0.2}}}}
    svc = decisions.DecisionService(fake_service, model="openjev")

    assert svc.noul("STATE", "Statement.")["noul"] == 0.88
    c = svc.choice("STATE", "Which team?", ["billing", "technical"])
    assert c["choice"] == "billing" and c["confidence"] == 0.8
    assert abs(sum(c["probabilities"].values()) - 1.0) < 1e-9

    body = _Handler.last_body
    assert body is not None
    assert body["state"] == "STATE"
    assert body["model"] == "openjev"
    assert body["questions"]["_choice"]["criteria"] == {
        "billing": "billing", "technical": "technical"}


def test_http_backend_tolerates_a_dialect_without_a_distribution(fake_service):
    """A service that returns only the pick still yields a usable distribution."""
    _Handler.payload = {"_choice": {"choice": "billing"}}  # flat, no probabilities
    c = decisions.DecisionService(fake_service).choice("s", "q", ["billing", "x"])
    assert c["choice"] == "billing"
    assert c["probabilities"] == {"billing": 1.0, "x": 0.0}


def test_http_5xx_is_unavailable_so_the_cascade_falls_back(fake_service):
    _Handler.status = 500
    with pytest.raises(decisions.DecisionUnavailable):
        decisions.DecisionService(fake_service).noul("s", "t")


def test_resolve_backend_http_caches_by_url_and_needs_a_url(monkeypatch):
    decisions._BACKENDS.clear()
    monkeypatch.setenv("DARKCORE_DECISION_URL", "http://127.0.0.1:1234")
    a = decisions.resolve_backend("http", None, None, {})
    b = decisions.resolve_backend("http", None, None, {})
    assert isinstance(a, decisions.DecisionService)
    assert a is b and a.base_url == "http://127.0.0.1:1234"
    monkeypatch.delenv("DARKCORE_DECISION_URL", raising=False)
    decisions._BACKENDS.clear()
    with pytest.raises(decisions.DecisionUnavailable):
        decisions.resolve_backend("http", None, None, {})
