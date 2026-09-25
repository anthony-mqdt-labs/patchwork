# Patchwork Agent Bootstrap

> **Purpose:** Every agent entering this workspace reads this file first. It says what
> patchwork is, which projects live inside it, and the procedures that apply to all of
> them.
> **Workspace:** `./` (this repository)
> **Shape:** a workshop for small models and ML constructs. It holds **many** projects,
> side by side, at different stages of life. The tiered cascade router is the first one
> to get a real spec — it is not the point of the repository.

## GitHub account preflight (required before project work)

This repository uses the GitHub account `anthony-mqdt-labs` (account ID
`81272454`), which also owns the repository remote.

Before doing any project work, first run this read-only check:

```bash
gh api user --jq .login
```

The required result is exactly:

```text
anthony-mqdt-labs
```

If GitHub CLI is unauthenticated, its credential is invalid, or another account
is active, stop before doing project work and direct the user to authenticate or
switch accounts. Do not log out, switch accounts, or start an interactive login
without the user's approval because GitHub authentication is shared across
repositories.

Preferred commands for the user:

```bash
# If the account is already stored by gh:
gh auth switch -h github.com -u anthony-mqdt-labs

# If the account has not been added yet:
gh auth login -h github.com -p https -w
```

After a switch or login, rerun `gh api user --jq .login` and begin work only
after it returns `anthony-mqdt-labs`.

Use this repository-local commit identity:

```text
user.name  = Anthony
user.email = 81272454+anthony-mqdt-labs@users.noreply.github.com
```

Do not change global Git identity settings for this project.

---

## 0. What patchwork is — and what it is not

Patchwork is a place to build small-model and ML constructs and keep the trail of
how they went. One repository, many projects, most of them unfinished by design.

**A router is one project. There will be others, and they need not resemble it.**
The workspace is explicitly open-ended: dozens of projects, indefinitely — a
routing layer now, and later whatever construct is worth building next. Routing is
the current occupant, not the organising principle.

In practice that means:

- **Do not read this repo's purpose as "routing."** If you are asked for a new
  construct, it does not need to compose with the router, reuse its interfaces, or
  justify itself as a routing improvement.
- **Do not export the router's abstractions by default.** The control surface,
  certificate rungs, tier ladder and cascade spine exist because *that* project
  needed them. A new project owns its own interfaces until it has a reason not to.
- **Composition is one technique among several** — not the house style, and not a
  precondition for anything living here.
- **What genuinely is shared:** the agent interface (`docs/agent-guide/`,
  `scripts/agent-tools/`, the commit gate), the standing constraints in §2, the
  per-project layout in §1, and the durable records in `MEMORY.md` and
  `CONTINUE.md`.

When this framing drifts — an agent describing the repo as a router project, or
growing a router-shaped abstraction into unrelated work — fix it here, where it
gets read, rather than only in chat.

---

## 1. Projects

| # | Project | Where it lives | State |
|---|---------|----------------|-------|
| 0001 | **Tiered cascade router** | `specs/0001-tiered-cascade-router/`, `experiments/router/`, `logs/router/` | **Active.** Spec ready (built, benched, thresholds pass); rung-3 decision tier added 2026-09-25, ships triage-only and default off |
| 0002 | **Config tuner** | `specs/0002-config-tuner/` | Planned — prd + design done, awaiting operator sign-off |
| 0003 | **Supervisory orchestrator** | `specs/0003-supervisory-orchestrator/` | Scaffold |
| — | Latent-bridge / composition line | `plans/index.md` | Paused, not abandoned (decisions D001–D004 stay tentative) |
| — | Tier performance measurements | `experiments/inference-bench/`, `logs/swap-econ/` | Measurement thread, feeds project 0001 |
| — | MiniCPM5 bench | `experiments/minicpm5_bench.py`, `experiments/minicpm5-results.json`, `experiments/bench-answers-minicpm5/` | One-off measurement |
| — | *(next project)* | `specs/NNNN-<slug>/` + `experiments/<slug>/` | **Unclaimed.** This is where a new construct goes |

### Landing a new project (the convention)

1. **Spec first.** Create `specs/NNNN-<slug>/` with a `prd.md` (problem, role,
   goals, non-goals) and a `design.md` when the approach firms up. A one-off
   measurement can skip the spec and be an experiment with a README instead.
2. **Build in `experiments/<slug>/`** with its own README. Give it its own
   virtualenv and `pyproject.toml` if it needs dependencies — do not assume
   another project's venv, and do not add dependencies to it.
3. **File findings in `research/<topic>.md`**, and add a row to the table in
   `research/README.md` naming the project they belong to. Negative results are
   the most valuable rows.
4. **Log to `logs/<slug>/`** if the project produces runtime logs or journals.
5. **Register it** in the table above and as a dated `D` entry in `MEMORY.md`.
   An unregistered project is invisible to the next agent.
6. **Never delete another project's research.** Add a section or a new file.

Projects share the repository, not a runtime. Promote code to a shared location
only when two projects independently need it, and say so in `MEMORY.md` when you
do.

---

## 2. Standing constraints (apply to every project here)

- **Target machine:** ~16 GB RAM, macOS (Apple Silicon). The dev box is an
  **M2 / 16 GB**; an M4 Max is the eventual target. Published measurements must
  say which machine produced them.
- **No GPU requirement** — CPU-only or Metal inference is the baseline.
- **No cloud dependencies** — everything runs locally. Hosted relays of
  proprietary models are not part of any shipped path.
- **Quantisation:** IQ4_XS (GGUF) or MLX 2-bit/4-bit as the defaults; weight
  quality at low precision is an open question worth measuring, not assuming.
- **All models must be permissively licensed** (Apache-2.0 / MIT). Non-commercial
  weights — CC-BY-NC, Llama community, Gemma terms — are out, and licence traps
  in a family that shares a name deserve a note in `research/`.
- **Disk is tight.** Anything over ~2 GB should be justified before download, and
  model weights belong in the Hugging Face cache at a pinned revision, not in a
  scratch directory. See `research/system-one-decision-models.md` §Provenance for
  the pattern (revision + SHA-256, verified after download).

---

## 3. Project 0001 — the composition line (active)

> This section is scoped to project 0001. It is *not* a description of patchwork.

The insight driving it: current frontier inference is monolithic — one big model
does everything. Project 0001 inverts that, treating individual models as **expert
components** composed at inference time via:

- **Routing** — a lightweight classifier dispatches a query to specialised modules
- **Latent bridging** ("telepathy") — learned projections pass compressed hidden
  states between modules (the paused thread)
- **Memory tiering** — only active modules are resident; cold modules live on disk
  and swap in at ~1 s for 4×7B on an NVMe
- **LoRA injection** — adapters plug into a module slot without reloading a base

**The target:** ~30B-equivalent capability at ~16 GB RAM, from 4× 7B or 2× 14B
dense models at IQ4_XS, composed rather than merged into one weight matrix.

> **⇒ Active workstream:** start at `docs/routing-architecture.md` (planes,
> taxonomy, certificate rungs, DAG, glossary), then the specs —
> `specs/0001-tiered-cascade-router/` (data plane, **ready** — built, benched, all
> thresholds pass), `specs/0002-config-tuner/` (planned — prd + design done,
> awaiting operator sign-off), `specs/0003-supervisory-orchestrator/` (scaffold).
> Session handoff: `CONTINUE.md`.

### Open questions — project 0001

**Routing:** per-token, per-phrase or per-layer? What classifier architecture —
small transformer, learned hash, n-gram LM? Learned offline or adaptive online?

**Verification (rungs):** which jobs admit a cheap certificate? Rung 3 measured as
a good *ranker* and a poor *decider* — is that a property of the job or of the
model class?

**Latent bridge:** does a linear projection between last-layer hidden states
suffice, or does it need attention? Trained end-to-end or post-hoc on cached
representations? Does bridging invalidate the KV-cache?

**Memory tiering:** can Colibrì's expert-streaming insight apply at the *module*
level? How many modules stay resident at IQ4_XS in 16 GB? Cold-swap latency?

**Quantisation:** does bridge quality degrade at 4-bit? Should the bridge and
router live at higher precision while the modules stay low?

**Composition patterns:** merge (SLERP/TIES/DARE), stripe (alternate modules
layer-by-layer), stack (A's last hidden state → B), ensemble (parallel weighted
vote), MoE-style dispatch, adaptive (router picks the pattern per prompt).

---

## 4. Directory layout

The shape is **per project**; project 0001 is the worked example.

```
patchwork/
├── AGENTS.md                         # This file — read first
├── MEMORY.md                         # Durable state, decisions, known issues
├── CONTINUE.md                       # Session handoff (overwritten per session)
├── docs/
│   ├── agent-guide/                  # Agent-facing procedures (never runbooks)
│   ├── routing-architecture.md       # Project 0001's master architecture doc
│   └── references.md                 # External papers, projects, tools
├── specs/                            # One directory per project
│   ├── 0001-tiered-cascade-router/   # data plane — ready 2026-07-17
│   ├── 0002-config-tuner/            # control plane — planned
│   └── 0003-supervisory-orchestrator/# control plane — scaffold
├── experiments/                      # One directory per construct, each with a README
│   ├── router/                       # project 0001's harnesses + darkcore/ (the router)
│   │   ├── darkcore/                 # surface/prefilter/predictor/models/verifiers/cascade/router/server/cli/tui
│   │   ├── plans/                    # router-scoped idea-stage plans
│   │   ├── battery.jsonl             # labelled battery + probes
│   │   ├── closure.py, swap_econ.py  # Exp 1–3, Exp 4 harnesses
│   │   ├── darkcore_bench.py, verify.py, BENCH-REPORT.md
│   │   ├── tests/                    # model-free suite (uv run pytest)
│   │   └── fixtures/
│   ├── inference-bench/              # tier perf measurements
│   ├── bench-answers-minicpm5/       # stored answer sets (numbers-only rule)
│   └── minicpm5_bench.py, minicpm5-results.json
├── logs/                             # One directory per project that logs
│   ├── router/
│   └── swap-econ/
├── plans/index.md                    # latent-bridge thread (paused)
└── research/                         # Findings, filed by project
    └── README.md                     # research artifact index
```

---

## 5. Standard agent workflow

1. **Read this file** (`AGENTS.md`) — the workspace framing, then the project you
   are working on.
2. **Read** `MEMORY.md` — durable decisions, current state, known issues.
3. **Read** `CONTINUE.md` — what the last agent was doing and where it stopped.
4. **Read** the spec and the plan documents for the project in question.
5. **Check** `research/` for existing findings — do not duplicate work, and do not
   re-derive a negative result someone already paid for.
6. **Begin work.** Prefer reading existing artifacts over running new experiments.

### When adding findings

- Add structured notes to the relevant `research/` document, or start a new one.
- Update `MEMORY.md` with durable decisions (`D` entries) and known issues.
- Update `CONTINUE.md` at session end with current state and the next action.
- Do NOT delete another agent's research — add a section or a new file.

**Commit gate (doc references).** `scripts/hooks/pre-commit` (wired via
`core.hooksPath`, so it is not in `.git/hooks/`) extracts the backticked spans in
`AGENTS.md`, `MEMORY.md`, `CONTINUE.md` and `plans/index.md`, and every span that
looks like a path must resolve **from the repo root** — not from the directory
the doc sits in, and a bare endpoint path is read as an absolute filesystem path
and fails. Spans containing a space or angle brackets are ignored, which is why
`POST /v1/systemone` and `~/.hermes/profiles/<profile>/...` are fine.

**But a green report is not proof.** The extractor pairs backticks blindly, so
every fenced code block shifts the pairing and produces paragraph-grabbing spans
that are then skipped for containing spaces. Measured coverage of the four docs:
`MEMORY.md` ~100 % (no fences), `CONTINUE.md` ~57 %, `plans/index.md` ~21 %,
`AGENTS.md` ~20 %. So the gate catches what it happens to see, not everything —
fix references properly rather than relying on it, and run
`scripts/agent-tools/update-agents.py --report` before committing.

### When experimenting

- Place prototypes in `experiments/<project>/` with a clear README.
- Document what you tested, what data you used, the result, and what it implies.
- If an experiment disproves a hypothesis, say so plainly — that is as valuable as
  a positive result, and it is the finding most likely to be repeated by accident.

---

## 6. Cross-profile notes

This workspace is designed for multiple agents to work in parallel. Avoid conflicts by:

- Working in separate files or clearly marked sections
- Writing research findings, not personal notes, in `research/`
- Leaving `plans/` documents as living documents — add, don't replace
- Using `CONTINUE.md` as the session handoff, not `MEMORY.md`

---

## 7. Cross-references (project 0001's lineage)

- **Colibrì:** `JustVugg/colibri` — expert streaming from disk, tiered memory, the
  primary inspiration for the memory tiering approach
- **llama.cpp:** `ggml` — quantization kernels (IQ4_XS), GGUF format, model loader
- **MergeKit:** model merging (SLERP, TIES, DARE) — the "merge" composition pattern
- **FrankenMoE / MoEfication:** converting dense models to MoE — expert-level routing
- **Sakana AI:** model merging research (evolutionary merging, cross-tokenizer bridges)
- **Spark:** `../spark/` — sibling workspace; owns the model fleet API and the eval
  substrate (`../spark/MODEL-EVAL-2026-07-15.md`) that project 0001's tiers came from
