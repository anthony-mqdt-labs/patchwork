"""Rung 3 — the decision tier: typed decisions from a resident tier, no generation.

A System One / "Jev-shaped" decision is a *fixed-shape* answer: a choice over N
options, a score on an ordered scale, or noul — p(true) of a statement about a
state. One forward pass, no autoregressive loop, nothing to parse, so the
output cannot be malformed. The family is open now (Laya, OpenJev, Tiny-Jev,
Jev-Style); `research/system-one-decision-models.md` is the survey.

This module implements the cheapest useful member of that family — the
**micro-scorer**: read the next-token distribution of a tier that is *already
resident* and treat it as a decision head. No new weights, no new dependency
(mlx-lm is already the tier runtime), no extra load: the model that just
answered is the model that verifies, so the certificate costs one forward pass
instead of a judge generation. Measured on this box: 1.2-2.3 ms/pass.

Why rung 3 and not rung 4 (`docs/routing-architecture.md` §6): one small
learned pass with no generation *is* the taxonomy's "learned lightweight
judge", and that document flags the rung as **the danger rung** — cheap plus
unreliable buys false security. So this module is built to be audited, never
trusted on latency:

  - every verdict reports its own evidence (p_yes, p_either, threshold), so the
    control plane can quarantine it like any other rung-3 label;
  - an ill-formed or unengaged pass counts as FAIL (escalate conservatively),
    matching the rung-4 judge's unparseable-verdict rule;
  - any backend failure degrades to the rung-4 LLM judge rather than blocking a
    route (dark-operable, the same posture as the class-prior predictor);
  - its safety is a *measured* number — false-pass rate on labelled pairs, from
    `decision_bench.py` — not a claim. Adopt where that number is 0.

Raw next-token probability is not a calibrated head. `decision_bench.py`
measures how far off it is, which is what decides whether fine-tuning is
required for threshold-gated use (see `research/decision-finetune-path.md`).
"""
import json
import os
import time
import urllib.error
import urllib.request

TELEMETRY_SAFE = True  # detail dicts carry numbers only (PII rule, telemetry.py)

# Surface forms of the two verdict tokens. Tokenizers differ in how they encode
# the leading space and the case, so every form is tried and only single-token
# forms vote (a multi-token "Yes" would be summed wrongly — better to abstain).
YES_FORMS = (" Yes", "Yes", " YES", "YES", " yes", "yes")
NO_FORMS = (" No", "No", " NO", "NO", " no", "no")

# Probe-validated phrasing (2026-09-25, Ternary-Bonsai-8B-mlx-2bit, cached):
# good answer -> p(' Yes') 0.748; bad answer -> p(' No') 0.689 / p(' Yes') 0.178.
NOUL_TEMPLATE = "{state}\n\n{statement}\nAnswer:"
DEFAULT_JUDGE_STATEMENT = "Does the ANSWER correctly and adequately resolve the QUESTION?"

DEFAULT_THRESHOLD = 0.5    # threshold on the renormalised p_yes
DEFAULT_MIN_ENGAGE = 0.10  # below this the model didn't really choose Yes or No
DEFAULT_TIMEOUT_S = 5.0


class DecisionUnavailable(Exception):
    """Backend missing, unreachable, or unable to score — caller decides."""


# --------------------------------------------------------------------- micro
def _surface_ids(tokenizer, forms):
    """Single-token ids for the given surface forms (empty when none encode)."""
    out = []
    for f in forms:
        e = tokenizer.encode(f)
        if len(e) == 1:
            out.append(int(e[0]))
    return sorted(set(out))


class MicroScorer:
    """A decision head made of one resident tier's next-token distribution.

    `noul` asks a yes/no statement about a state; `choice` scores each option
    independently and softmaxes the resulting logits (the OpenJev algorithm:
    independent probabilities -> logits -> a proper distribution), which keeps
    options order-independent and needs no option-length bookkeeping.
    """

    kind = "micro"

    def __init__(self, model, tokenizer, model_id=None):
        self.model = model
        self.tok = tokenizer
        self.model_id = model_id
        self.yes_ids = _surface_ids(tokenizer, YES_FORMS)
        self.no_ids = _surface_ids(tokenizer, NO_FORMS)
        if not self.yes_ids or not self.no_ids:
            raise DecisionUnavailable("tokenizer has no single-token yes/no form")

    # -- the one primitive everything else is built from -------------------
    def _verdict_probs(self, prompt):
        """(p_yes, p_no, p_either) after `prompt` — one forward pass, no gen."""
        import mlx.core as mx

        ids = self.tok.encode(prompt)
        if not ids:
            raise DecisionUnavailable("empty prompt")
        logits = self.model(mx.array([ids]))[0, -1]
        probs = mx.softmax(logits.astype(mx.float32))
        mx.eval(probs)
        p_yes = sum(float(probs[i]) for i in self.yes_ids)
        p_no = sum(float(probs[i]) for i in self.no_ids)
        return p_yes, p_no, p_yes + p_no

    def noul(self, state, statement, template=None):
        """p(statement is true | state), renormalised over the yes/no tokens."""
        prompt = (template or NOUL_TEMPLATE).format(state=state, statement=statement)
        t0 = time.perf_counter()
        p_yes, p_no, p_either = self._verdict_probs(prompt)
        n = p_yes + p_no
        return {
            "noul": round(p_yes / n, 4) if n > 0 else 0.0,
            "p_yes": round(p_yes, 5),
            "p_no": round(p_no, 5),
            "p_either": round(p_either, 5),
            "passes": 1,
            "forward_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def choice(self, state, question, options, mode="binary"):
        """A distribution over `options` (order-independent)."""
        if not options:
            raise DecisionUnavailable("no options to choose from")
        t0 = time.perf_counter()
        if mode == "first_token":
            # One pass: score each option by the probability of its first token.
            prompt = f"{state}\n\n{question}\nOptions: {', '.join(options)}.\nAnswer:"
            import mlx.core as mx
            ids = self.tok.encode(prompt)
            probs = mx.softmax(self.model(mx.array([ids]))[0, -1].astype(mx.float32))
            mx.eval(probs)
            raw, passes = {}, 1
            for o in options:
                e = self.tok.encode(o.lstrip())
                raw[o] = float(probs[int(e[0])]) if e else 0.0
        else:
            # N passes: each option scored on its own (robust to shared prefixes).
            raw, passes = {}, 0
            for o in options:
                r = self.noul(state, f"The best answer is: {o}.", template=(
                    "{state}\n\n{statement}\nAnswer:"))
                raw[o] = r["p_yes"]
                passes += 1
        return {
            **_to_distribution(raw),
            "passes": passes,
            "mode": mode,
            "forward_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def score(self, state, question, levels):
        """Expected level index on an ordered scale (may land between levels)."""
        d = self.choice(state, question, list(levels), mode="binary")
        probs = d["probabilities"]
        exp = sum(i * probs[l] for i, l in enumerate(levels))
        return {"score": round(exp, 4), "probabilities": probs,
                "confidence": d["confidence"], "passes": d["passes"]}


def _to_distribution(raw):
    """Independent probabilities -> logits -> softmax (OpenJev's shape)."""
    import math

    eps = 1e-6
    logits = {k: math.log(max(v, eps) / max(1.0 - min(v, 1.0 - eps), eps))
              for k, v in raw.items()}
    m = max(logits.values()) if logits else 0.0
    exp = {k: math.exp(v - m) for k, v in logits.items()}
    z = sum(exp.values()) or 1.0
    probs = {k: round(v / z, 4) for k, v in exp.items()}
    pick = max(probs.items(), key=lambda kv: kv[1])[0] if probs else None
    return {"choice": pick, "probabilities": probs,
            "confidence": probs.get(pick, 0.0) if pick else 0.0}


# ---------------------------------------------------------------------- http
class DecisionService:
    """Out-of-process backend speaking the ecosystem's /v1/systemone contract.

    Exists so a bigger System One model (Laya, OpenJev, Tiny-Jev) can verify
    without dragging torch or foreign weights into the router venv — the same
    reason the tier roster is a roster and not a hard-coded import.

    NOTE: request shape follows the published OpenJEV/Laya docs; response
    normalisation is deliberately tolerant because the served dialects differ
    in the wrapper key. Live-verified against a running service only — until
    then the micro backend is the tested path.
    """

    kind = "http"

    def __init__(self, base_url, model="openjev", timeout_s=DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def request(self, state, questions):
        """questions: {id: {type, instructions, criteria|options}} -> {id: dict}."""
        payload = {"state": state, "questions": questions}
        # Some services (the hosted OpenJEV relay) require `model`; a local
        # `laya serve` rejects an unknown one ("unknown model 'openjev'").
        # Omitting the key is the portable default.
        if self.model:
            payload["model"] = self.model
        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                body = json.loads(r.read().decode())
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise DecisionUnavailable(f"{type(e).__name__}: {e}") from e
        return _normalise(body, list(questions))

    def noul(self, state, statement, template=None):
        t0 = time.perf_counter()
        res = self.request(state, {"_verdict": {
            "type": "noul", "instructions": statement}})
        d = res.get("_verdict") or {}
        p = float(d.get("noul", 0.0))
        return {"noul": round(p, 4), "p_yes": round(p, 5), "p_no": round(1 - p, 5),
                "p_either": 1.0, "passes": 1,
                "forward_ms": round((time.perf_counter() - t0) * 1000, 2),
                "backend_model": d.get("model")}

    def choice(self, state, question, options, mode="binary", descriptions=None):
        """`criteria` is a {label: description} map in the served dialects; the
        label is a serviceable description of itself when the caller has none."""
        t0 = time.perf_counter()
        criteria = {o: (descriptions or {}).get(o, o) for o in options}
        res = self.request(state, {"_choice": {
            "type": "choice", "instructions": question,
            "criteria": criteria}})
        d = res.get("_choice") or {}
        probs = d.get("probabilities") or {}
        probs = {k: float(v) for k, v in probs.items() if k in options}
        if not probs:  # dialect without a distribution: reconstruct from one-hot
            pick = d.get("choice")
            probs = {o: (1.0 if o == pick else 0.0) for o in options}
        pick = d.get("choice") or max(probs.items(), key=lambda kv: kv[1])[0]
        return {"choice": pick, "probabilities": probs,
                "confidence": float(d.get("confidence", probs.get(pick, 0.0))),
                "passes": 1, "mode": "service",
                "forward_ms": round((time.perf_counter() - t0) * 1000, 2)}


def _normalise(body, ids):
    """Tolerate the served dialects: {results|answers|data: {...}} or flat."""
    if not isinstance(body, dict):
        raise DecisionUnavailable("non-object response")
    for key in ("results", "answers", "data", "decisions"):
        if isinstance(body.get(key), dict):
            body = body[key]
            break
    out = {}
    for qid in ids:
        v = body.get(qid)
        if isinstance(v, dict):
            out[qid] = v
    return out


# ----------------------------------------------------------------- backends
_BACKENDS = {}


def resolve_backend(backend, pool, tier, thresholds):
    """Return a scored-decision backend. Tests monkeypatch this."""
    if backend == "http":
        url = thresholds.get("url") or os.environ.get("DARKCORE_DECISION_URL")
        if not url:
            raise DecisionUnavailable("no decision service url configured")
        key = ("http", url, thresholds.get("model", "openjev"))
        if key not in _BACKENDS:
            _BACKENDS[key] = DecisionService(
                url, model=thresholds.get("model", "openjev"),
                timeout_s=float(thresholds.get("timeout_s", DEFAULT_TIMEOUT_S)))
        return _BACKENDS[key]

    # micro: verify with a tier that is already resident — the answering tier by
    # default, so the certificate costs a forward pass and not a model load.
    verify_tier = thresholds.get("tier") or tier
    acquire = getattr(pool, "acquire", None)
    if acquire is None:
        raise DecisionUnavailable("pool cannot expose a raw model for scoring")
    model, tok, _load_ms = acquire(verify_tier)
    key = ("micro", verify_tier, id(model))
    sc = _BACKENDS.get(key)
    if sc is None or getattr(sc, "model", None) is not model:
        sc = MicroScorer(model, tok, model_id=verify_tier)
        _BACKENDS[key] = sc
    return sc


# ----------------------------------------------------------------- verifier
def decision_judge(query, answer, pool, tier, ctx):
    """Rung 3 verifier: does the ANSWER resolve the QUESTION? One small pass.

    Contract matches the other verifiers: (passed, telemetry-safe detail).

    Two modes, chosen by the thresholds:

    **Two-way (default).** pass iff p_yes >= `threshold`. Measured on 18
    labelled pairs this is UNSAFE as a pass authority: at 0.5 it passed 2 of 9
    plausible-but-wrong answers (false-pass 22%), and the smallest threshold
    with zero false passes (0.75) then rejects 6 of 9 good answers — the head's
    ranking, not its calibration, is what fails (`decision_bench.py`).

    **Three-way triage (`bypass_threshold` / `fail_threshold`).** The safe mode,
    and the one the evidence supports: a confident verdict short-circuits the
    judge in either direction, and everything ambiguous falls through to the
    rung-4 judge instead of being decided by a rung-3 head. This is the same
    rule rung 0 already uses (`inconclusive -> judge`), applied to confidence
    instead of to structure. Unengaged passes can never PASS in either mode.

    Backend failure -> the rung-4 judge, flagged so the fallback is never
    invisible.
    """
    from .verifiers import next_tier_judge  # local import: avoid an import cycle

    th = ctx.get("thresholds") or {}
    backend = th.get("backend", "micro")
    threshold = float(th.get("threshold", DEFAULT_THRESHOLD))
    min_engage = float(th.get("min_engage", DEFAULT_MIN_ENGAGE))
    bypass = th.get("bypass_threshold")
    fail_at = th.get("fail_threshold")
    triage = bypass is not None or fail_at is not None
    statement = th.get("statement", DEFAULT_JUDGE_STATEMENT)
    t0 = time.perf_counter()

    try:
        scorer = resolve_backend(backend, pool, tier, th)
        d = scorer.noul(judge_state(query, answer), statement)
    except Exception as e:  # noqa: BLE001 — a certificate must never break a route
        passed, detail = next_tier_judge(query, answer, pool, tier, ctx)
        detail.update({"check": "judge_decision_fallback", "backend": backend,
                       "fallback": "llm_judge", "decision_error": type(e).__name__,
                       "verify_ms": round((time.perf_counter() - t0) * 1000, 2)})
        return passed, detail

    p_yes, engage = float(d["noul"]), float(d.get("p_either", 1.0))
    engaged = engage >= min_engage
    evidence = {
        "rung": 3, "check": "decision_judge", "backend": backend,
        "p_yes": p_yes, "p_either": round(engage, 5), "threshold": threshold,
        "min_engage": min_engage, "engaged": engaged,
        "passes": d.get("passes", 1), "decision_tokens": 0,
        "forward_ms": d.get("forward_ms", 0.0),
        "verify_ms": round((time.perf_counter() - t0) * 1000, 2),
    }

    if triage:
        if bypass is not None and engaged and p_yes >= float(bypass):
            return True, {**evidence, "role": "bypass_pass"}
        if fail_at is not None and (not engaged or p_yes <= float(fail_at)):
            return False, {**evidence, "role": "bypass_fail"}
        # ambiguous: this head does not decide — the judge does (rung 0's rule)
        passed, detail = next_tier_judge(query, answer, pool, tier, ctx)
        detail.update({**evidence, "role": "triage_inconclusive"})
        detail["verify_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return passed, detail

    passed = engaged and p_yes >= threshold
    return passed, {**evidence, "role": "two_way"}


def judge_state(query, answer):
    """The state the verifier scores a statement against (never telemetrised)."""
    return f"QUESTION:\n{query}\n\nANSWER:\n{answer}"


def decide(state, questions, backend="micro", pool=None, tier=None, thresholds=None):
    """General entry for the decision tier (bench + the tuner's L1 admission).

    questions: [{"kind": "noul"|"choice"|"score", "instructions": str,
                 "criteria": [str, ...]}] — mirrors the served contract.
    """
    th = dict(thresholds or {})
    th.setdefault("backend", backend)
    scorer = resolve_backend(th["backend"], pool, tier, th)
    out = []
    for q in questions:
        kind = q["kind"]
        if kind == "noul":
            out.append({"kind": kind,
                        **scorer.noul(state, q["instructions"])})
        elif kind == "choice":
            out.append({"kind": kind,
                        **scorer.choice(state, q["instructions"], q["criteria"])})
        elif kind == "score":
            out.append({"kind": kind,
                        **scorer.score(state, q["instructions"], q["criteria"])})
        else:
            raise ValueError(f"unknown question kind {kind!r}")
    return out
