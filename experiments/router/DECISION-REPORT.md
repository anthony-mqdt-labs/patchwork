# Decision Tier — Live-Fire Report (rung 3, System One verifier)

**Agent:** penny (Hermes, profile `penny`)
**Date:** 2026-09-25
**Status:** measured; adopt-where-safe recommendations below, everything else is a proposal
**Artifact:** `DECISION-BENCH.json` (numbers only — the PII rule applies to bench output too)
**Module under test:** `darkcore/decisions.py` (new), registry entry `decision_judge` (rung 3)
**Machine:** M2 MacBook Air, 16 GB, macOS; tier `T1` = `prism-ml/Ternary-Bonsai-8B-mlx-2bit`
(2.2 GB, ternary 2-bit, already resident in the HF cache — **no model download was
needed for this bench**)
**Reproduce:** `cd experiments/router && .venv/bin/python decision_bench.py --judge`

> **Read this as measurements, not as a sales sheet.** The research survey
> (`research/system-one-decision-models.md`) says these models are fast and
> cheap; this report is what happened when one of them was pointed at *our*
> verifier job on *this* machine.

---

## What was built

`darkcore/decisions.py` gives the cascade a rung-3 certificate with no new
weights and no new dependency: it reads the next-token distribution of a tier
that is **already resident** and treats it as a decision head.

| piece | what it does |
|---|---|
| `MicroScorer.noul` | p(statement true) as `p(yes)/(p(yes)+p(no))`, one forward pass, zero generated tokens |
| `MicroScorer.choice` | a distribution over options — `binary` (N passes, independent scoring softmaxed) or `first_token` (1 pass, argmax over option first-token probability) |
| `MicroScorer.score` | expected level index on an ordered scale |
| `DecisionService` | the same three primitives over `POST /v1/systemone`, so Laya / OpenJev / Tiny-Jev can verify without dragging torch or foreign weights into the router venv |
| `decision_judge` | rung-3 verifier: two-way `threshold` mode, and a three-way `bypass_threshold`/`fail_threshold` triage mode |

29 model-free tests (`tests/test_decisions.py`, 20 of them) cover the head, the
policy, the cascade wiring, the PII rule on the detail dict, the response
dialects, and the fallback to the rung-4 judge. Full suite: **76 passed, 3
skipped** (the opt-in embedder parity tests).

---

## 1. Safety — the gate (18 labelled pairs, `fixtures/decision-pairs.json`)

Rung 3 is the taxonomy's *danger rung*: cheap plus unreliable buys false
security. So the question is the dangerous error rate — how many answers that
do **not** resolve the query would it pass?

| metric | value |
|---|---|
| AUC (raw p_yes) | **0.914** |
| at threshold 0.50 | false-pass **22.2 %** (2 of 9 bad answers passed), false-fail 0 % |
| smallest threshold with zero false passes | 0.75 → but false-fail **66.7 %** |

The two bad answers that slip through at 0.5 are worth naming, because they are
exactly the failure the cascade exists to prevent:

- `R2-bad` — "You bought 20 apples and 4 oranges" (wrong; answer is 18 and 6) → p_yes **0.7466**
- `R4-bad` — the bat-and-ball "ball costs $0.10" (the trap answer) → p_yes **0.5548**

Both are *plausible and confident*. The head ranks them above several correct
answers, so no single threshold separates good from bad: raising the bar to
0.75 to stop them also rejects 6 of the 9 good answers.

**Verdict: the raw head must not be the pass authority.** It is a good ranker
with no usable operating point.

## 2. Post-hoc calibration does not fix it

Leave-one-out Platt scaling (temperature + bias, fitted on 17 pairs, applied to
the held-out one):

| | raw | calibrated (LOO) |
|---|---|---|
| AUC | 0.914 | **0.864** |
| at 0.50 | false-pass 22.2 % / false-fail 0 % | false-pass 22.2 % / false-fail 11.1 % |
| smallest safe threshold | 0.75 (66.7 % false-fail) | **none in 0.05–0.95** |

This is structural, not bad luck: calibration is a **monotone** transform, and
monotone transforms preserve ranking. The failure is a *ranking* failure, so
calibration cannot repair it — it only relabels the same ordering. Fitted
temperatures ran 1.95–4.00 (median 2.20), i.e. the head is systematically
under-confident, but rescaling an ordering that already puts a wrong answer
above a right one just moves both.

## 3. What the evidence *does* support: three-way triage

Because the head is a decent ranker and a poor decision-maker, the safe design
is to let it **short-circuit the judge when it is confident, and abstain
otherwise** — the same rule rung 0 already uses (`inconclusive -> judge`), with
confidence in place of structure.

| | in-sample sweep | **out-of-sample (LOO)** |
|---|---|---|
| judge calls skipped | 9/18 (50 %) at bypass 0.8 / fail 0.2 | **8/18 (44 %)** |
| false-pass by rung 3 | 0 | **0** |
| false-fail by rung 3 | 0 | **0** |

**Quote the LOO row.** The in-sample sweep picks thresholds on the same pairs it
scores — that is selection on the test set and will always flatter.

Rule of three for the honesty of the zero: 0 errors in 18 trials is consistent
with a true error rate as high as ~17 % (95 %). The triage mode therefore ships
**off by default**, is labelled in telemetry (`role: bypass_pass | bypass_fail |
triage_inconclusive`), and should be re-measured on the 0002 tuner's label
stream before it is trusted to gate anything.

## 4. The two verifiers err *differently* (this is the most interesting result)

| pair | label | p_yes | rung 3 @0.5 | rung-4 judge |
|---|---|---|---|---|
| `R2-bad` | 0 | 0.7466 | **PASS (wrong)** | FAIL ✓ |
| `R4-bad` | 0 | 0.5548 | **PASS (wrong)** | FAIL ✓ |
| `R1-good` | 1 | 0.7197 | PASS ✓ | **FAIL (wrong)** |

Agreement was 15/18, but "83 % as good" is the wrong reading. Each verifier
catches what the other misses: rung 3 is blind to *plausible-but-wrong*, and the
judge rejected the "missing dollar" explanation that is actually correct and
precise. Neither dominates, and both are wrong in ways worth remembering.

**Independent corroboration.** A parallel investigation
(`research/router-and-guard-models-brief.md`) ran a *different* candidate — the
best-in-class zero-shot NLI cross-encoder
(`MoritzLaurer/ModernBERT-large-zeroshot-v2.0`) — on the same question and hit
the same wall from the other direction: it scored a correct answer **0.884** and
a wrong answer **0.506** against "correctly and completely resolves the
question", scored **0.022 vs 0.032** for "the answer is wrong or unfounded"
(no signal), and fired on *both* answers (0.824 / 0.874) for "the answer
addresses the question" — i.e. it measures **topical relevance, not factual
correctness**. Two independent zero-shot approaches, two different substrates,
one conclusion: **the verifier job is not solved by an off-the-shelf model.**
That is what makes the fine-tune verdict in `research/decision-finetune-path.md`
a real finding rather than a preference.

This also means the triage mode's abstain region is not a concession — it is
where the costly judgement actually belongs.

## 5. Step 0 executed: a *real* shipped System One model, on the same 18 pairs

`research/decision-finetune-path.md` said to bench a shipped calibrated model
before writing any training code. That is now done, not planned: **Laya** (the
421 M ModernBERT encoder, RLCD-trained against proper scoring rules, Apache-2.0)
served torch-free via the `ggmlc` Metal binary and `mys/laya-GGUF`
(`laya_english_q8_0.gguf`), driven through `darkcore/decisions.py`'s own
`/v1/systemone` client:

```bash
./laya serve models/laya_english_q8_0.gguf --port 8123 --device metal
.venv/bin/python decision_bench.py --backend http --url http://127.0.0.1:8123
```

Numbers preserved in `LAYA-BENCH.json`; the two runs are compared below.

| on the 18 labelled pairs | micro-scorer (8 B ternary) | **Laya (421 M)** |
|---|---|---|
| AUC | **0.914** | 0.728 |
| false-pass @0.5 | 22.2 % | 22.2 % |
| false-fail @0.5 | 0 % | **22.2 %** |
| smallest zero-false-pass threshold | 0.75 → 66.7 % false-fail | 0.70 → 77.8 % false-fail |
| Platt-calibrated AUC | 0.864 | 0.642 |
| **LOO triage judge calls skipped** | **8/18 (44 %)** | **0/18 (0 %)** |
| wall per verdict | ~869 ms | **101 ms** |

Laya passed `R3-bad` (the wrong 17×23) and `A1-bad` ("just restart nginx"), and
*failed* `R2-good` and `R4-good` — both correct answers. Its LOO triage found no
threshold pair with zero error at all.

**So the purpose-built, calibrated, published-accuracy decision model is worse
than a raw next-token head from our own resident tier at this job** — better
calibration per the card, better engineering, and still no discrimination between
a correct answer and a plausible wrong one. Note the documented caveat that
Laya's `noul` is its weakest primitive on the English checkpoint, which is
precisely the primitive this benchmark uses.

**Three independent approaches have now failed the same job:** the micro-scorer
(AUC 0.914, 22 % false-pass), a zero-shot NLI cross-encoder (0.884 vs 0.506 —
relevance, not correctness), and real Laya (AUC 0.728). That convergence is the
finding. The verifier job needs a head trained on our own verdicts, or it keeps
the LLM judge.

## 6. Choice (the predictor job): the decision model is excellent at *its* job

| mode | accuracy on battery.jsonl (12 queries) | wall per query |
|---|---|---|
| Laya (service, its designed job) | **12/12 (100 %)** | **53 ms** |
| micro-scorer `first_token` (1 pass) | 10/12 (83.3 %) | ~1.5 s |
| micro-scorer `binary` (N passes) | 6/12 (50.0 %) | ~3.3 s |

The split is the story: **a model built to make choices is good at choices and
bad at verdicts.** Laya matches the incumbent embedder kNN exactly (12/12) at
53 ms — but its confidences on the correct `reasoning` picks sat at 0.16–0.46,
because its `confidence` is `1 − normalised entropy` (a *concentration*
measure), not a probability that the answer is right. Gating on it would
discard correct answers.

The `binary` mode is systematically biased — it answered `emotional` for 8 of 12
queries, including every non-emotional one it missed, with confidences that never
leave 0.37–0.76. Scoring each option independently with a *statement* ("The best
answer is: X.") invites the model to agree with whatever statement it is shown;
asking the question **once** and reading the next token does not.

Against the incumbent: the embedder kNN class prior is documented at **12/12**
(BENCH-REPORT v0.2, LOO over its own exemplar snapshot). Laya ties it; the
micro-scorer does not beat it. *Protocol caveat: the kNN number is leave-one-out
over its store; the numbers here are zero-shot with no fitting.*

**Verdict: keep the kNN for class prediction** — Laya ties rather than wins, and
adds a service dependency for the same accuracy. But the tie is worth knowing:
this is a job where the decision-model family delivers on the tin, unlike the
verifier job.

## 7. Wiring notes found by running a real service

The client needed two dialect fixes that only appear against a live server:

- **`model` must be omittable.** A local `laya serve` *rejects* an unknown model
  (`unknown model 'openjev'`); the hosted relay requires one. The client now
  omits the key by default and takes it from config.
- **`choice.criteria` is a `{label: description}` map**, not a list. Passing a
  list 422s. The client now sends the map, using the label as its own
  description when the caller has none.

Both are exercised by the loopback test and now by a real service, which is the
difference between a contract and a guess.

## 8. Cost — and why the marketing number did not survive contact

| path | generated tokens | wall per verification |
|---|---|---|
| rung-3 micro pass, long real state | **0** | 0.39 – 1.10 s |
| rung-4 judge, same tier, same pairs | 2/pair (36 total) | 0.84 – 2.52 s |
| rung-3 micro pass, warm 55-token prompt (probe) | 0 | **1.4 ms** |

The 1.4 ms figure is real but it is a *short-prompt best case*, measured on a
warm model with ~55 tokens of prompt. On real query+answer states the pass costs
0.4–1.1 s, because **prefill dominates and the state is long** — the decision is
one forward pass, but it is one forward pass *over the whole state*.

So on this box and against this baseline the honest win is: **100 % of generated
tokens, ~40 % of wall time.** The dramatic token saving in the literature
assumes the alternative is an LLM *generating a structured answer*
(schema names, formatting and all) or a thinking-tier judge narrating CoT —
which is what patchwork's T2 does (256–512 judge tokens, and the measured judge
tax of 24.5 %).

**Documented deviation:** T2 is not downloaded on this box (7.9 GB, and the disk
is at 93 %), so the judge baseline here is `T1` **judging itself**. That makes
this comparison a measurement of *cost and cache shape*, not of judge quality.
Re-run with T2 resident before quoting any judge-tax saving.

## Threats to validity

- **n = 18.** Nine good, nine bad, hand-written by the agent that then measured
  them. The pairs are committed in `fixtures/` so the labels can be
  independently reviewed or replaced — that is the fix, not a bigger claim.
- **One model, one machine.** A 2-bit ternary 8B on an M2 with the disk at 93 %.
  The paging history in this repo (27B evicts the embedder, "never trust solo
  latency measurements on this box") applies here too.
- **Judge baseline is self-judging** (above).
- **Choice comparison mixes protocols** (kNN LOO vs zero-shot) — stated inline.
- The triage LOO's zero-error intervals are wide (rule of three) — it is a
  *signal to instrument*, not a certificate.

## Recommendations (ranked)

1. **Ship rung 3 as triage-only, default off.** The mode is safe by
   construction (confident verdicts short-circuit, ambiguity goes to the judge)
   and measured 44 % of judge calls avoided with zero observed error. Gate the
   default on a larger labelled set from the 0002 tuner.
2. **Do not swap the class predictor.** Laya ties the kNN rather than beating
   it, and adds a service dependency; the micro-scorer is worse than both.
3. **Try the decision head on the tier decision**, not the class decision — the
   interesting, unexploited question in this repo is whether difficulty
   prediction improves with a ranker that is good at *ordering* (a good ranker
   and a poor decider is exactly the shape you want for a λ input).
4. **Stop hunting off-the-shelf verifiers.** Step 0 is closed and negative: real
   Laya on our own pairs scored AUC 0.728 with 22 % false-pass *and* 22 %
   false-fail, and its LOO triage skipped 0/18. The zero-shot NLI cross-encoder
   failed the same way from the other direction (0.884 vs 0.506). Three
   independent approaches, one conclusion — this job is not solved by a model
   someone else trained for a different purpose.
5. **If rung 3 must become the pass authority, train the head on our own
   verdicts** — it is the *head*, not the model, that needs training, and the
   labels already exist in the rung-4 judge's output. Process and acceptance
   gate in `research/decision-finetune-path.md`. Until then the LLM judge keeps
   the pass authority and rung 3 stays a default-off bypass.
6. **Adopt the family where it demonstrably works: choice.** Laya scored 12/12
   at 53 ms on the routing job — tying the kNN, ~16× faster than our 8 B pass —
   and it opens a question the router currently cannot ask cheaply: tool
   selection for the agentic class, or the tier decision itself. That is the
   adoption path with positive evidence behind it, and it is the inverse of
   where this session started looking.
