# Decision Heads — Is Fine-Tuning Necessary, and What Is the Process?

## Investigation: does a System One decision head need training for patchwork's jobs?
**Agent:** penny (Hermes, profile `penny`)
**Date:** 2026-09-25
**Status:** answered for the three jobs we actually have; process documented; **no training run performed**
**Evidence:** `experiments/router/DECISION-REPORT.md`, `DECISION-BENCH.json`,
`fixtures/decision-pairs.json` (18 labelled pairs), `darkcore/decisions.py`

> Where this document says "measured", it means the bench in this repository on
> this machine. Where it says "reported", it means a model publisher's own
> number, cited with its source. They are not the same thing.

### Hypothesis
That adopting a decision model for patchwork's verifier job requires
fine-tuning one on our own labels — i.e. that off-the-shelf weights cannot be
trusted to decide whether a small tier's answer resolves the question.

### Method
18 hand-written labelled pairs (9 competent, 9 that fail the query) scored by
the rung-3 micro-scorer on the cached ternary 8B; raw threshold sweep; leave-one-
out Platt calibration; leave-one-out three-way triage. Plus a review of the
published training recipes (LoRA + marker-token head, span-scoring head,
soft-target losses, post-hoc temperature) and a verification of this machine's
own fine-tuning CLI (`mlx_lm lora`).

### Results
**No fine-tune is needed to rank** (AUC 0.914) or to **triage** (LOO 8/18 judge
calls skipped, 0 false-pass, 0 false-fail). **A fine-tune is needed only if the
head must be the pass authority**: 22.2 % false-pass at 0.5, and the smallest
zero-false-pass threshold rejects 66.7 % of good answers. Calibration cannot
repair it, because Platt scaling is monotone and preserves the ranking that is
wrong.

### Implications
Three jobs, three different answers — and the same conclusion arrived
independently from a zero-shot NLI cross-encoder
(`research/router-and-guard-models-brief.md`), which scored a correct answer
0.884 against a wrong answer's 0.506: the *verifier* job is not solved
off-the-shelf. Bench a shipped calibrated model on our own pairs (Step 0) before
writing any training code; if training is needed, the acceptance gate in §3 is
the thing to write down first.

---

## 1. Verdict — three jobs, three different answers

| job | fine-tuning needed? | evidence | use instead |
|---|---|---|---|
| **Rank / choose among options** (predictor) | **No** | AUC 0.914 zero-shot; `first_token` choice 10/12 on the battery | nothing — the existing bge-small kNN is documented 12/12; do not swap |
| **Triage bypass** (verifier, judge skipped only when confident) | **No** | LOO: 8/18 judge calls skipped, 0 false-pass, 0 false-fail | ship default-off, re-fit thresholds on your own labels |
| **Pass authority** (rung 3 decides pass/fail alone) | **Yes** | 22.2 % false-pass at 0.5; smallest zero-false-pass threshold rejects 66.7 % of good answers; Platt calibration does not repair it | rung-4 judge, or a trained head |

The distinction that matters: the raw head is a **good ranker and a poor
decider**. It orders answers sensibly (AUC 0.914) but puts two confident,
plausible, *wrong* answers above several correct ones — and no threshold can
separate them, because the error is in the ordering, not in the scale.

## 2. Why calibration cannot fix it (and why this is not a tuning problem)

`p(yes)` from a base model is the **continuation** probability of the token
`Yes`. A trained decision head does something structurally different: it scores
**option spans against the question** and normalises within the question's
option group.

- `com-kotobalabs/open-jev-deberta-v3-large`: the head scores
  `[mean(question tokens); mean(option tokens); product]` per option, softmaxed
  per question group; loss was cross-entropy + Brier, with temperature fitted
  post hoc on a validation split.
- `lostargon/Tiny-Jev`: a marker token plus a 1-dimensional head over the
  hidden state at each option marker; soft cross-entropy against
  **probability-vector targets**, then temperature calibration.
- `argos1111/modernbert-ja-310m-jev`: a cross-encoder reading
  (question + state, candidate) as one sequence, softmaxed across candidates.

Continuation asks "is this a plausible next token"; a span head asks "does this
option answer *this* question". Plausible-but-wrong is precisely where the two
disagree — which is exactly the failure we measured (`R2-bad` at p_yes 0.747,
`R4-bad` at 0.555).

Calibration is a **monotone** transform, and monotone transforms preserve
ranking. Fitted temperatures ran 1.95–4.00 (median 2.20), i.e. the head is
under-confident, but scaling an ordering that already misplaces a wrong answer
only moves both answers together. **Calibration fixes the scale; only training
fixes the order.**

## 3. The process, in the order it should be attempted

### Step 0 — Bench a shipped calibrated model first (minutes, no training)
Before writing any training code, point the bench at a model whose head was
already trained and calibrated by someone else:

```bash
.venv/bin/python decision_bench.py --backend http --url http://127.0.0.1:8080
```

- `lostargon/Tiny-Jev` (0.6B, Apache-2.0; reports 95.8 % on its test split with
  ECE 0.004, and 99.8 % accuracy at ≥0.9 confidence covering 90 % of items),
- `convaiinnovations/laya` (421 M ModernBERT encoder, Apache-2.0, 100+ languages)
  — note its fine-tuning recipe **is** documented (see the ledger in Step 2c),
- `chaoliangUNSW/Jev-Style-0.8B-Decision-v3-MLX` (0.8B, MLX-native, reports 79.2 %
  on 2 000 typed decisions).

`--backend http` is implemented, covered by loopback tests, **and has now been
pointed at a real service on this machine.** Result: **negative.** Real Laya
(torch-free via ggmlc, Metal) scored AUC 0.728 on our 18 pairs with 22 %
false-pass *and* 22 % false-fail, and its LOO triage skipped 0/18 judge calls —
worse than a raw next-token head from our own tier (AUC 0.914, 8/18 skipped).
See `DECISION-REPORT.md` §5 for the side-by-side.

**So this step is closed, and no shipped model cleared the plausible-but-wrong
cases.** With the zero-shot NLI cross-encoder also failing (0.884 vs 0.506 —
relevance, not correctness), three independent off-the-shelf approaches have now
failed the verifier job. The remaining paths are (2b) a head trained on our own
verdicts, or keeping the rung-4 judge.

### Step 1 — Get labels (the cascade already manufactures them)
The rung-4 judge emits PASS/FAIL on real routes, and spec 0002's tuner already
plans L1 label admission (judge labels quarantined + corroborated). Export
`(state, statement, verdict)` triples from that stream. Two requirements:

- **Deliberate hard negatives.** Include plausible-but-wrong answers
  specifically — that is the observed failure mode. A dataset of obvious errors
  will train a head that passes subtle ones.
- **Negation and scoping pairs.** "asks for a refund" / "does not ask" /
  "asks only for a refund" — the published recipes use contrastive sets for a
  reason, and Tiny-Jev's own limitation note says negations are read literally.

Scale, from published recipes: 18 000 states / 42 000 questions was enough for
the DeBERTa span head (1 epoch, one H100, 229 s, ≈ $0.25); Tiny-Jev used ≈100 k
synthetic states. Hundreds of pairs is enough to *calibrate*; thousands to train
a head.

### Step 2a — Cheap path: LoRA the continuation (works today, no new architecture)
Verified on this machine (`python -m mlx_lm lora --help`, mlx-lm 0.31.3):

```bash
# data dir holds {train,valid,test}.jsonl
python -m mlx_lm lora --model <local-or-hf-model> --train \
  --data <dir> --fine-tune-type lora --num-layers 16 \
  --batch-size 1 --iters <N> --mask-prompt --adapter-path adapters/decision
```

`--mask-prompt` is the load-bearing flag: it trains the completion
(`" Yes"` / `" No"`) without also training the model to reproduce the state.
This teaches the *continuation* to answer the verifier's question — it is the
cheapest thing that can move the ranking, and it is worth doing as a baseline
before building a head. It is **not** the published recipe; it is the subset
mlx-lm supports directly.

### Step 2b — Real path: train a decision head
The published shape (Tiny-Jev, Apache-2.0, recipe stated on its model card):
LoRA (r = 32) on all linear layers **plus a new marker token and a
1-dimensional head**, soft cross-entropy against probability-vector targets,
one epoch, then post-hoc temperature calibration.

Two consequences worth knowing before choosing this path:

1. **You need soft targets, not labels.** The loss is against a distribution, so
   either multiple annotators or an ensemble/judge-derived distribution is
   required. Single PASS/FAIL labels are a degraded substitute.
2. **It is a new model artifact** — weights, head, marker token, calibration
   file, versioning — which is a larger commitment than the entire rung-3 tier
   as currently built (a module with no new weights at all).

Hardware on this machine: LoRA on a 0.6–2B model is feasible on an M2 with
16 GB. A full span-head train at the published scale is a rented-GPU job.

**Read one published ablation before betting on the marker-token recipe.**
`com-kotobalabs` reports that a **fresh marker-token head does not learn** — it
sits at the label prior — while scoring the *mean of the option's text tokens*
(`[mean(question); mean(option); product]`) is what actually works. Tiny-Jev's
marker-token + linear head did work for them, so the two published recipes
differ on the central architectural choice. Budget for testing both, or start
from the span-scoring head, which has a documented ablation behind it.

### Step 2c — The published fine-tuning ledger (what each family actually ships)

| family | training code? | cost / constraints |
|---|---|---|
| `convaiinnovations/laya` | **Yes** — a Kaggle notebook running the full loop: dataset build, **RLCD** (proper-scoring-rule rewards, GRPO-style policy gradient), temperature fitting, eval, Hub push | 2×T4, **~4–5 h for 4 epochs over ~30 k questions**; needs torch + CUDA |
| `zefan-cai/Open-Jev` | **Yes, best documented** — `pip install -e '.[train]'`, multi-domain recipes, row schema `id/group_id/state/instructions/primitive/criteria`, public dataset (326 619 rows, 147 139 train) | GPU/Linux; LoRA r8/α16 |
| `kotoba-lang/typed-decisions` | **Yes but Modal-only** — `python -m typed_decisions.data --out data`, `modal run modal_app.py::encoder …`; CE + Brier; question augmentation at p = 0.7 | 237 s / **$0.26** on one H100 for 18 k states × 1 epoch; no local/CPU path |
| `apus-ailab/APUS-OpenJev-v1` | **Provenance only** — the SFT curriculum, loss (`0.5·CE_low + 0.5·CE_high + 0.1·KL`), LoRA r8/α16 are documented, but the adapters are in a separate access-restricted repo | 5 949-record curriculum, 1 epoch |
| `chaoliangUNSW/Jev-Style` | **No** — no script or dataset on any card; v2's card gives the envelope: rank-32 LoRA, **36.9 min on one H100 80 GB** | — |
| `razorback16/openjev`, HF `openjev/openjev` | **No** — server and frozen readout constants only | — |

**The pattern across the ledger:** every documented path is GPU + torch. There is
no torch-free way to train one of these, which for this project means the
training step (if it is ever taken) happens elsewhere and only the *inference*
artefact comes home — which is exactly what `--backend http` exists to carry.

### Step 3 — Calibrate and gate (the step that decides adoption)
1. Fit the threshold **leave-one-out** on the labelled set — never in-sample.
2. Report false-pass at the operating point, with the count, not just a rate.
3. Apply the rule of three: 0 errors in n trials bounds the true rate by ≈ 3/n.
   18 pairs cannot bound it below ~17 %; ~150 pairs bounds it below 2 %.

**Write the acceptance gate down before training** (otherwise the result will be
argued into usefulness):

| criterion | threshold |
|---|---|
| false-pass on hard negatives | 0, on n large enough that 3/n < 2 % |
| coverage at that operating point | ≥ 50 % of judge calls skipped |
| cost | ≤ 1 forward pass, 0 generated tokens |

Miss it and rung 3 stays default-off with the rung-4 judge as the authority —
which is a perfectly good outcome, and the one the current evidence supports.

## 4. What not to do

- **Do not train on the 18 bench pairs.** They are the test set. A head fitted
  on them will look excellent and mean nothing.
- **Do not quote an in-sample threshold sweep.** This repository has already
  paid for that lesson once (the mlx-port myth: a "fix" that did not fix it
  because the measurement was not isolated). The in-sample triage sweep said
  9/18 skipped; LOO said 8/18 — close, but the *only* defensible number is LOO.
- **Do not fine-tune a chat model to emit JSON.** Structured-output failure is
  the thing this whole model family exists to eliminate; teaching it back in is
  a step backwards.
- **Do not adopt a head before checking the ranking failure specifically.**
  Accuracy on easy pairs will look fine. The only question that matters is
  whether plausible-but-wrong still slips through.

## 5. Open questions

1. Does a shipped calibrated model (Laya, Tiny-Jev, Jev-Style) clear the two
   plausible-but-wrong pairs? (Step 0 — unrun.)
2. Is the ranking failure intrinsic to a 2-bit ternary substrate, or does it
   persist on a full-precision base? A cheap follow-up: re-run the bench against
   a bf16 small model.
3. Does the `first_token` choice mode (the winner at 10/12) hold up at larger
   option counts, or does it break as options start sharing a first token?
4. Does a head trained on the judge's own verdicts inherit the judge's errors —
   specifically its rejection of `R1-good` (a correct explanation)? Distilling a
   judge that is wrong 1/18 on good answers propagates that error in a new place.
