# Research Directory

> Structured findings from investigations. Each subdirectory or file is one investigation thread.

> **Filings are per project.** The threads below belong to **project 0001** (tiered
> cascade router) unless a row says otherwise. A new project's findings get their own
> file and their own row; prototypes live in `experiments/<slug>/`. The convention for
> landing a project is in `AGENTS.md` §1 — patchwork holds many of them, and routing is
> only the first.

## Contents

| File | Topic | Status |
|------|-------|--------|
| `small-specialist-landscape.md` | Survey of sub-4B specialist models for composed architectures (SmolLM3 3B, Phi-4-mini, Qwen3.5-4B, tiny routers/drafters) | Complete |
| `system-one-decision-models.md` | The System One / Jev-style decision family (Laya, OpenJev ×3, Tiny-Jev, Jev-Style, com-kotobalabs, Needle 3): the typed-decision contract, use patterns, local runnability on a 16 GB M2, licence traps | Complete (survey; live-fire in the router experiment) |
| `decision-finetune-path.md` | Is a decision head worth training? Verdict per job (rank / triage / pass-authority), why calibration cannot fix a ranking failure, and the labelled-data → LoRA/marker-head → LOO-calibration process with an acceptance gate | Complete (no training run performed) |
| `router-and-guard-models-brief.md` | Non-Jev alternatives for the same two jobs: bge-small + nearest-centroid for routing (measured 0.808 zero-shot → 0.841 @ 21 exemplars on a banking77 proxy), ONNX Runtime as a torch-free path, why zero-shot NLI cross-encoders fail at correctness judging (measured), GLiNER2/SetFit/guard classifiers/TinyRouter | Complete (agent-measured; numbers are on a banking77 proxy, not our battery) |
| `scale-resolution-and-architecture.md` | Coastline paradox / fractal dimension as a lens on architecture: scaling exponents as Richardson plots, superposition vs. allocated notions, architecture as free symmetries, precision as a focus knob | In-progress (conceptual, no experiments) |
| `latent-bridge-survey.md` | Cross-model latent communication techniques | Not started |
| `routing-literature.md` | Token routing approaches (MoE, MoA, etc.) | Not started |
| `memory-tiering-analysis.md` | Colibrì-inspired tiering at module level | Not started |
| `quant-impact.md` | Quantization-aware bridge quality | Not started |

## How to contribute

- Create a new file for each investigation thread
- Use this frontmatter format:

```
## Investigation: [Brief Title]
**Agent:** [profile name]
**Date:** YYYY-MM-DD
**Status:** in-progress | complete | inconclusive

### Hypothesis
What we expected to find.

### Method
What we did.

### Results
What we found. Include numbers, data, and failure modes.

### Implications
What this means for the design.
```

- Add negative results clearly. They save the next agent from repeating dead ends.
- After completing an investigation, update the status in `plans/index.md`.
