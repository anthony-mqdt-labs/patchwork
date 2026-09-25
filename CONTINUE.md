# Patchwork — CONTINUE.md

> **Handoff snapshot.** Read this first when resuming work in a new session.
> **Rule:** Overwrite at end of session. Never append. This is a snapshot, not a history.

> **Session 2026-09-25:** rung-3 **decision tier** built, tested, and live-fired
> (`experiments/router/darkcore/decisions.py` + `decision_bench.py` + `DECISION-REPORT.md` +
> `research/system-one-decision-models.md` + `research/decision-finetune-path.md`).
> **Committed** as `feat(router): rung-3 decision tier` — see §0.

---

## ⚠️ 0. Session state a returning agent MUST know

**Repo shape — read before scoping work.** Patchwork holds **many** small-model / ML
constructs; the tiered cascade router is project **0001**, not the repository's
identity. Read `AGENTS.md` §0–§1 before treating anything here as router-wide, and see
D011 in `MEMORY.md`. The work described below is project 0001's.

**GitHub preflight (D009) — PASSED, after a detour.** At session start
`gh api user --jq .login` returned a *different* account than this repository
requires, which correctly blocked all commits; the operator switched to
`anthony-mqdt-labs` and the preflight then returned it. The lesson stands and is
the reason D009 exists: **check the account before writing anything**, and never
commit under a shared CLI session whose identity you have not verified. The
account this repo requires is the one named in `AGENTS.md`; do not assume the
active one.

**Session 2026-09-25 work: committed** (`feat: rung-3 decision tier`). The tree
is clean and the handoff below describes the committed state.

| path | state |
|---|---|
| `experiments/router/darkcore/decisions.py` | NEW — decision tier (rung 3) |
| `experiments/router/darkcore/surface.py` | MODIFIED — `decision_judge` added to `VERIFIER_REGISTRY` |
| `experiments/router/tests/test_decisions.py` | NEW — 24 model-free tests (incl. a loopback http round-trip) |
| `experiments/router/decision_bench.py` | NEW — the live-fire harness |
| `experiments/router/fixtures/decision-pairs.json` | NEW — 18 hand-labelled pairs (the safety test set) |
| `experiments/router/DECISION-REPORT.md` | NEW — measured results |
| `experiments/router/DECISION-BENCH.json` | NEW — the numbers |
| `experiments/router/LAYA-BENCH.json` | NEW — the real-Laya live fire (§5) |
| `research/system-one-decision-models.md` | NEW — the landscape survey |
| `research/decision-finetune-path.md` | NEW — fine-tune verdict + process |
| `research/README.md`, `MEMORY.md` | MODIFIED — index + D010 + an OPEN issue |

**Suite state:** 85 passed, 3 skipped (the opt-in parity tests).
`uv run pytest` / `.venv/bin/python -m pytest` both work.

**Do not start a new thread before committing this**, or the survey and the
measurements will rot in the tree.

---

## 1. What this session established (the short version)

The question was whether System One / "Jev-style" decision models (Laya,
OpenJev, Tiny-Jev, Jev-Style) can cut patchwork's token and latency bill. They
can — but **only as a triage bypass, never as a verdict**, and the survey's
optimism does not survive contact with our own verifier job:

- A raw next-token head is a **good ranker** (AUC 0.914) and a **poor decider**
  (22.2 % false-pass at 0.5). No threshold separates the plausible-but-wrong
  answers from the correct ones, and calibration cannot fix it (a monotone
  transform preserves ranking).
- The safe design is **three-way triage** — confident PASS/FAIL short-circuits
  the judge, everything ambiguous still goes to the rung-4 judge (rung 0's own
  `inconclusive -> judge` rule, applied to confidence). LOO: **8/18 judge calls
  skipped, 0 false-pass, 0 false-fail**. Ships **default off** (rule of three:
  0/18 bounds the true error rate only at ≈17 %).
- **Do not replace the class predictor**: `first_token` choice scored 10/12
  against the embedder kNN's 12/12.
- **Cost is prefill-bound**, not 1 ms: 0.39–1.10 s/pass on real query+answer
  states (vs 1.4 ms on a warm 55-token probe). Against a two-token judge the win
  is ~40 % wall time and 100 % of generated tokens; the big token win only
  materialises against a CoT-narrating thinking-tier judge (T2, not downloaded).
- **Step 0 is DONE and it is NEGATIVE.** A real shipped calibrated model (Laya,
  421 M, torch-free via the ggmlc Metal binary + mys/laya-GGUF) was driven
  through our own `POST /v1/systemone` client on the **same 18 pairs**: AUC 0.728,
  22 % false-pass *and* 22 % false-fail, LOO triage 0/18 — worse than a raw
  next-token head from our own tier. With the zero-shot NLI cross-encoder also
  failing (0.884 vs 0.506), three independent off-the-shelf approaches have now
  failed the verifier job. **Stop looking for one.** Either train the head on the
  judge's own verdicts, or keep the rung-4 judge.
- **Where the family *did* deliver:** the choice job. Laya scored **12/12 at
  53 ms** (tying the kNN, ~16× faster than our 8 B pass). That — routing, tool
  selection, the tier decision — is the adoption path with positive evidence
  behind it.

**Memory of the two verifiers, if nothing else survives:** each catches what the
other misses — rung 3 passed both wrong answers the judge caught, and the judge
rejected the one correct "missing dollar" explanation that rung 3 accepted.

---

## 2. Bootstrap Sequence (Do This First, In Order)

```bash
cd .
git status && git log --oneline -5      # expect dark-core v0.2 commits at HEAD

# THE conceptual entry point:
cat docs/routing-architecture.md        # §6 is the rung table the decision tier lives in

# The JOURNEY — evidence→decision→next-step chain, episodes 1–9:
cat specs/0001-tiered-cascade-router/journal.md

# Router bench findings (v0 → v0.2 comparison up top):
cat experiments/router/BENCH-REPORT.md

# NEW — the decision tier: what was measured, and what it does not license:
cat experiments/router/DECISION-REPORT.md

# Watch it happen on the gauge board (~2 min):
cd experiments/router
uv sync   # first time only — pinned env (.venv) via uv.lock
uv run darkcore board --replay --speed 30
```

---

## 3. Standing state (unchanged this session)

**dark-core v0.2: built, benched three times, all 5 thresholds pass.** Spec 0001
`ready`; mlx embedder port done (parity EXACT, snapshot v2, config v4);
predictor live in class-prior mode (class detection 12/12); rung-0 with
arg-shape + `inconclusive→judge`; judge caps (T1 256 / T2 512) + `skip_start`;
escalation visibility; server + CLI seams 1–6 live-verified; spark owns the
router (`spark darkcore-router`).

Open/residual: **27B evicts the co-resident embedder** (mmap is mmap; mitigated
by rewarm-at-the-evicting-route's-tail; S4 passes with margin). Rung-0 stays
blind to *strategy* (A4, by design).

## 4. What's Next (Prioritized)

1. **Commit this session's work** — after the GitHub account switch (D009).
2. ~~Bench a shipped calibrated decision model on the same 18 pairs
   (`--backend http`).~~ **DONE 2026-09-25 — NEGATIVE** (see §1). The runner-up
   candidates remain untried and are the only ones worth a look: **Jev-Style
   0.8 B v3** (MLX/GGUF, Apache-2.0 — the family's top pick for a no-torch 16 GB
   box) and **Tiny-Jev 0.6 B** (Apache-2.0, best-documented calibration), both of
   which need torch or a llama.cpp fork. Licence facts corrected: `pip install
   laya` hard-requires torch (use the ggmlc GGUF route instead); SiliconLabAI's
   micro-scorer **is MIT** (not unlicenced); `openjev.sh` is a $JEV-funded
   *hosted relay* to TypeSafe's Jev, not an open model; and the HF
   `openjev/openjev` **weights are CC-BY-NC-4.0** while `razorback16/openjev` is
   Apache-2.0 — two different projects sharing a name.
3. **Grow the labelled set** before enabling triage by default — 18 pairs cannot
   bound a zero error rate below ≈17 % (rule of three). The 0002 tuner's L1
   label stream is the natural source.
4. **Then** the standing backlog: 0002 (tuner; needs operator sign-off on 4 open
   items), 0003 (orchestrator; attention budget), phase-0 exemplar corpus
   (n ≫ 21), residual seams via the tuner.
5. Optional and cheap: re-run the decision bench against a **bf16** small model
   to test whether the ranking failure is intrinsic or an artefact of the 2-bit
   ternary substrate.

## 5. Gotchas

- **`experiments/router` is a standalone uv project**: own `.venv` + `uv.lock`,
  `darkcore` editable, console script `uv run darkcore …`. Tests:
  `uv run pytest` (85 model-free ~2.5 s + 3 opt-in parity via `DARKCORE_PARITY=1`).
- **The 27B evicts the embedder** — any new resident component must assume T2
  wipes the page cache; put rewarms at the evicting route's tail, and never
  trust solo latency measurements on this box. The decision tier's timings were
  measured with T1 resident only.
- **Disk is at 93 %** (≈33 GB free). T2 (7.9 GB) is deliberately not downloaded;
  the judge baseline in `DECISION-BENCH.json` is therefore T1 judging itself —
  a documented deviation, re-run with T2 resident before quoting judge-tax numbers.
- **PII rule is enforced at the sink** (`telemetry.emit` asserts on content
  keys). New verifier details must carry numbers only — `decision_judge`'s
  detail dict is asserted clean by `tests/test_decisions.py`.
- **A new verifier name must be added to `surface.VERIFIER_REGISTRY`** or
  `validate()` rejects the config (I6). Covered by a test.
- Config is at **v4**; `set_config` is the one write path and journals every
  change. The decision tier deliberately did **not** change the shipped config —
  no class uses `decision_judge` yet.
- Bench answer snapshots live in `experiments/router/bench-answers/`;
  `DECISION-BENCH.json` follows the same rule (numbers only).
- **Decision-tier model weights live in the HF cache at pinned revisions**, and the
  `ggmlc` tool (which runs Laya torch-free) is at `~/.local/share/laya/laya`. Revisions,
  SHA-256s and the tool's trust caveats are recorded in
  `research/system-one-decision-models.md` §Provenance — verify, do not assume.
- **The 27B generation drift:** `experiments/router/darkcore/config.json` asks for
  prism-ml/Ternary-Bonsai-27B-mlx-2bit while spark's registry points at
  prism-ml/Ternary-Bonsai-2-27B-mlx-2bit (a newer build). Neither is downloaded; pick
  deliberately rather than letting one arrive by accident.
- Eval substrate lives in **spark**: `../spark/MODEL-EVAL-2026-07-15.md`.
