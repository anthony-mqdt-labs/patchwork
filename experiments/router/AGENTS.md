# Project 0001 — Tiered Cascade Router (agent context)

> **Supplements the root `AGENTS.md` — read that first.** Patchwork holds many
> small-model and ML constructs; this file is scoped to **project 0001 only**. Repo-wide
> rules (GitHub account preflight, licensing, the commit gate, disk discipline) are
> stated once in the root file and deliberately not repeated here.
> Paths in this file are **repository-root-relative**, per the repo convention.

## Status — 2026-09-25

- Spec 0001 is **ready**: built, benched, all five thresholds pass. T0/T1 live; **T2 is
  not downloaded on this machine**.
- **Rung-3 decision tier** added 2026-09-25 (`experiments/router/darkcore/decisions.py`).
  It ships **three-way triage, default off**, and never holds pass authority (D010 in
  `MEMORY.md`).
- **OPEN defect:** terminal-tier unavailability fails *open* — an unavailable last tier
  returns the previous tier's rejected answer with `flagged=False`. Logged in `MEMORY.md`.
- Control plane: 0002 config tuner (planned, awaiting operator sign-off), 0003
  supervisory orchestrator (scaffold). This project is the **data plane**.

## The project's thesis

Current serving is monolithic: every query pays 27B-class latency and RAM even when a
1.7B would answer. This project treats individual models as **expert components**,
composed at inference time:

- **Routing** — a lightweight classifier dispatches a query to specialised modules
- **Latent bridging** ("telepathy") — learned projections pass compressed hidden states
  between modules (*paused thread*, planned in `plans/index.md`)
- **Memory tiering** — only active modules are resident in RAM; cold modules live on disk
  and swap in ~1 second at 4×7B scale on an NVMe
- **LoRA injection** — adapters plug into any module slot without reloading the base

**The target:** ~30B-equivalent capability at ~16 GB RAM, using 4× 7B or 2× 14B dense
models at IQ4_XS, composed modularly rather than merged into a single weight matrix.

The spine is a **cascade, not a predictor**: the spark eval (`../spark/MODEL-EVAL-2026-07-15.md`)
established that the small tier is *confidently wrong* — fluent and structured and wrong on
anything needing judgement. So a router that only predicts sends bad answers downstream
with no safety net; it has to verify and escalate.

## Question backlog

The project's open questions, kept here rather than in the root bootstrap (they are this
project's, not the repository's).

### Model selection
- Is Qwen2.5 the right family? What about Gemma-2, Llama-3.2, DeepSeek-V2-Lite?
- Do the models need to share the same tokenizer for latent bridging, or can a learned
  projection handle mismatch?
- What is the minimum viable module size? 7B? 3B? 1.5B?

### Routing
- Per-token routing vs. per-phrase vs. per-layer? What latency/quality trade-offs?
- What routing classifier architecture? Small transformer? Learned hash? N-gram LM?
- Is the router learned offline or adaptive online?

### Verification (the certificate rungs)
- Which jobs admit a cheap certificate, and which force the expensive judge?
- Rung 3 measured as a good **ranker** (AUC 0.914) and a poor **decider** (22.2 %
  false-pass). Is that a property of the job or of the model class? (Three off-the-shelf
  attempts have failed; the working assumption is *the job*.)

### Latent bridge ("telepathy")
- Does a simple linear projection between last-layer hidden states of model A and
  first-layer inputs of model B suffice?
- Does the bridge need attention, or is an MLP bottleneck enough?
- Can we use cross-attention between modules instead of sequential bridging?
- Is the bridge trained end-to-end on a small corpus, or post-hoc on cached representations?
- How does the bridge interact with KV-cache — does bridging invalidate cache?

### Memory tiering
- Colibrì's streaming insight (experts on disk, LRU cache, RAM/VRAM/DRAM tier) — can we
  apply it at the *module* level instead of the *expert* level?
- How many modules can be resident at IQ4_XS in 16 GB? 4×7B? 2×14B?
- What is the cold-swap latency from a fast NVMe?

### Quantisation strategy
- IQ4_XS (from GGUF) is the baseline. Does bridge quality degrade at 4-bit?
- Can the bridge / router live at higher precision (8-bit) for better signal?

### Composition patterns
- **Merge:** SLERP / TIES / DARE of same-arch models
- **Stripe:** route tokens through alternating modules layer-by-layer
- **Stack:** run module A's full forward pass, then pass its last hidden state to module B
- **Ensemble:** run all modules in parallel, weighted vote on output
- **MoE-style:** router dispatches per token to one or more modules
- **Adaptive:** router decides the composition strategy per prompt

## Code map — `experiments/router/darkcore/`

This is a map, not documentation: read the module before trusting the one-liner.

| Module | Role |
|---|---|
| `experiments/router/darkcore/surface.py` | the control surface — hard interface to the control plane; holds the `VERIFIER_REGISTRY` allowlist |
| `experiments/router/darkcore/prefilter.py` | cheap pre-filters applied before any model call |
| `experiments/router/darkcore/predictor.py` | class-prior start-tier prediction (`ClassPrior.classify`) |
| `experiments/router/darkcore/exemplars/` | published exemplar snapshot backing the class prior (config v3+) |
| `experiments/router/darkcore/embedder_mlx.py` | MLX fp32 bge-small embedder — torch-free since 2026-07-18 |
| `experiments/router/darkcore/models.py` | tier lifecycle + answer extraction (handles bare `</think>`) |
| `experiments/router/darkcore/verifiers.py` | the rung ladder (0–5), incl. `next_tier_judge` |
| `experiments/router/darkcore/decisions.py` | rung 3: micro-scorer, `/v1/systemone` client, `decision_judge` triage |
| `experiments/router/darkcore/cascade.py` | the verify-and-escalate loop (the spine) |
| `experiments/router/darkcore/router.py` | arbitration between candidate routes |
| `experiments/router/darkcore/server.py` | standalone OpenAI-compatible endpoint, dark-operable |
| `experiments/router/darkcore/telemetry.py` | caller-visible escalation stream |
| `experiments/router/darkcore/cli.py`, `experiments/router/darkcore/tui.py` | entry points; gauge-board TUI |

## How to run

```bash
cd experiments/router
uv run pytest                                   # model-free suite: 85 passed, 3 skipped
uv run python verify.py                         # spec thresholds — all five pass
uv run python darkcore_bench.py                 # cascade bench
uv run python decision_bench.py                 # rung-3 micro-scorer live fire
uv run python decision_bench.py --backend http --url http://127.0.0.1:PORT
```

- Server, relay and integration: `experiments/router/ROUTER-SERVER.md`,
  `experiments/router/INTEGRATION-GUIDE.md`, `experiments/router/QUICKSTART.md`.
- **There is no torch in this venv** (dropped 2026-07-18 with the MLX embedder port).
  Anything requiring torch runs out-of-process in its own throwaway venv — do not add it
  to this project's dependencies.
- The spark relay (`../spark/src/spark/runtimes/relay.py`) can proxy to this server, so
  an agent harness can reach the router directly or through spark.

## Invariants — do not break these

1. **Rung 3 never holds pass authority.** Triage only, default off, and gated on a larger
   labelled set than the current 18 pairs (D010).
2. **The control surface is the only interface** between this project and the control
   plane. No side channels, no shared mutable state.
3. **Verifiers are allowlisted** in `surface.py`. Adding one means registering it *and*
   giving it a test.
4. **Weights are pinned.** Model files live in the Hugging Face cache at a recorded
   revision with a verified SHA-256 — see `research/system-one-decision-models.md`
   §Provenance. Permissive licences only.
5. **Measurements name the machine.** The dev box is an M2 / 16 GB; the spec's target
   profile is an M4 Max. Both kinds of number appear in these reports — label them.
6. **Fail loud, never silently certify.** A tier that cannot run must flag the route. See
   the OPEN fail-open issue in `MEMORY.md`; it is the current counter-example.

## Gotchas

- **T2 residency evicts the co-resident embedder** (page cache, not a runtime bug — mmap
  is mmap). Mitigated by rewarming at the evicting route's tail; worst overhead 22.31 ms.
- **Bonsai-27B leaks chain-of-thought** as prose. Fixed at answer extraction, not generation.
- **T2 is not downloaded on this machine**, which is what makes the fail-open defect
  reachable in ordinary use rather than only in tests.
- **Prebuilt Laya GGUFs bake in `max_opts=16`** — more than 16 options aborts (SIGABRT).
- **The `ggmlc` runtime that runs Laya is unaudited** (unsigned, no attestation, no
  licence file, binds all interfaces). It exists here as a measurement harness and must
  never become a shipping dependency.
- **Off-the-shelf verifiers measured worse than our own raw head** (Laya AUC 0.728 vs
  0.914). The hunt is closed on measurement, not on taste — see
  `experiments/router/DECISION-REPORT.md`.

## Documents

| Document | What it holds |
|---|---|
| `specs/0001-tiered-cascade-router/prd.md` | problem, role, goals, non-goals |
| `specs/0001-tiered-cascade-router/design.md` | the design |
| `specs/0001-tiered-cascade-router/control-surface.md` | the hard interface between the planes |
| `specs/0001-tiered-cascade-router/journal.md` | the narrative evidence→decision→next-step record; why the backlog is ordered as it is |
| `specs/0001-tiered-cascade-router/decisions.md` | spec-level decision log |
| `specs/0001-tiered-cascade-router/results.md`, `tests.md`, `status.yaml` | measurements, test plan, status |
| `docs/routing-architecture.md` | master architecture: planes, taxonomy, certificate rungs, DAG, glossary |
| `experiments/router/BENCH-REPORT.md` | cascade bench results |
| `experiments/router/DECISION-REPORT.md` | rung-3 measured results (incl. the negative result) |
| `experiments/router/DECISION-BENCH.json`, `experiments/router/LAYA-BENCH.json` | the numbers |
| `experiments/router/README.md` | harness index |
