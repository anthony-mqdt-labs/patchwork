# System One / Jev-Style Decision Models — Landscape and Use

## Investigation: what exists in the "make a decision, not a paragraph" model family, and how should patchwork use it?
**Agent:** penny (Hermes, profile `penny`)
**Date:** 2026-09-25
**Status:** complete for the survey; adoption recommendations in
`experiments/router/DECISION-REPORT.md`, training process in
`research/decision-finetune-path.md`
**Method:** primary sources only — Hugging Face model API + raw model cards,
PyPI metadata, project sites — read 2026-09-25. **No model other than the
already-cached ternary 8B was downloaded or run for this survey.**
Provenance markers: **[HF]** = fetched from the source just now · **[rep]** =
a publisher's own reported number · **[docs]** = project documentation.

### Hypothesis
That the open "System One" family — models that return typed decisions instead
of text — can supply patchwork's cheap certificates, replacing either the rung-4
LLM judge (24.5 % of token spend) or the embedder class-prior, at one forward
pass and zero generated tokens.

### Method
Primary-source survey, 2026-09-25: the Hugging Face model API and raw model
cards, PyPI metadata, and project sites. No model was downloaded or run for this
survey; the quantitative live-fire against our own verifier job is reported
separately in `experiments/router/DECISION-REPORT.md`.

### Results
Eleven distinct open implementations found, spanning 29 M to 26 B parameters and
three different mechanisms (bidirectional encoder + head, discrete diffusion,
and orchestration over chat LLMs). Three licence traps were found in a family
that reads as uniformly permissive: one **CC-BY-NC-4.0** build, one
**CC-BY-SA-4.0** build, and one project with **no licence file at all**. Local
runnability on a 16 GB M2 is narrow: the 26 B MoE cannot run here, and most
others need either torch (which this venv deliberately lacks) or a service
process. The cost model is **prefill-bound**, so vendor latency figures do not
transfer to long states.

### Implications
Adopt the **contract**, not a model: patchwork's cheapest useful member of this
family is a decision head made from a tier that is already resident, which is
what `darkcore/decisions.py` implements. Do not replace the class predictor.
Use the tier for triage, not for verdicts.

---

## 1. The contract (what makes a model one of these)

A System One model answers **typed questions about a state** with fixed-shape
values, in one forward pass, without generating text:

| primitive | input | returns |
|---|---|---|
| **choice** | a question + a set of candidate options | the pick, a probability per option, a confidence |
| **score** | a question + ordered levels | an expected level index (may land between levels) + a distribution |
| **noul** | a statement about the state | p(true), 0–1 |

The consequence that matters operationally: **the answer space is the option
list you pass in**, so there is no parse, no retry, and no schema error — the
model cannot emit a value outside the set you defined. That is the entire value
proposition, and it is why these belong next to an LLM, not instead of one.
They cannot plan, reason step by step, write, or count.

Jev (TypeSafe AI, Sep 2026) defined the contract and remains closed (waitlist,
32 K context, no published architecture; 70–500 ms and 20–200× / 40–400× cost
claims **[docs]**). Everything below is an open implementation of that contract.

## 2. The family, as verified today

| model | base | params | licence | local verdict |
|---|---|---|---|---|
| `convaiinnovations/laya` | ModernBERT-large | 421 M | **Apache-2.0** | **`pip install laya` hard-requires torch** (core dep; the `onnx` extra does *not* remove it — `laya.onnx_agent` imports `laya.common`, which imports torch). Torch-free path = the third-party `ggmlc` GGUF build, **verified running on this machine**. Fitted temperatures shipped. |
| `akshatbindal/laya-typed-decisions` | Laya | 421 M (F16) | **Apache-2.0** | the benchmark fine-tune; 0.757 acc vs Jev 0.727, 146 ms p50 **[rep]** |
| `lostargon/Tiny-Jev` | Qwen3-0.6B | 596 M | **Apache-2.0** | torch + `trust_remote_code`; best-documented calibration; guardrail training |
| `chaoliangUNSW/Jev-Style-0.8B-Decision-v3-MLX` | Qwen3.5 | 0.8 B | **Apache-2.0** | **MLX-native** — best fit for this stack; 79.2 % on 2 000 typed decisions **[rep]** |
| `com-kotobalabs/open-jev-deberta-v3-large` | DeBERTa-v3-large | ≈435 M | Apache-2.0 weights; GitHub repo **NOASSERTION** | span-scoring head; 512-token context (state cut to 256); install is **git-only** and needs torch; 1.8 s / 4 questions on M1 Max CPU **[rep]** |
| `apus-ailab/APUS-OpenJev-v1` | Qwen3.5 | 4 B / 9 B / 35 B-A3B | **Apache-2.0** | native runtime, `effort=low/high`, GGUF builds |
| `razorback16/openjev` (codiv) | DiffusionGemma 26B-A4B | 26 B MoE | Apache-2.0 (code **and** weights) | discrete-diffusion reader, `POST /v1/systemone`. **vLLM PR #57250 is merged** (2026-09-22), though the image still applies two local patches. MLX 4-bit weights are **16.54 GB** — not runnable on 16 GB |
| `openjev/openjev` (HF) — *a different project, same name* | **Qwen3.8-27B** | 27.4 B | **weights CC-BY-NC-4.0** (helpers Apache-2.0) | 54.7 GB bf16 / 14.09 GiB at 4-bit; frozen readout constants, no training docs |
| `heman10x/rlcd-modernbert-151m` ("Verdict") | ModernBERT | 151 M | see repo | 512 tokens, ≤24 options; another Jev-shaped head |
| `Mapika/decider` | Qwen3.5 fine-tunes | 2 B | see repo | speaks `POST /v1/systemone`; calibration-aware RL; ~4 GB CUDA |
| `zefan-cai/open-jev` | Qwen 2 B / 9 B / 27 B | — | code MIT **[docs]** | LoRA adapter + a decision head, published demo catalogue |
| `SiliconLabAI/OpenJev` | *any* chat LLM | — | **MIT** (`LICENSE.md`) | not weights: TypeScript orchestration. Local API is `POST /api/evaluate` on :3001, **not** `/v1/systemone`; N calls per choice question with the state re-sent each time |
| `openjev.sh` | *hosted relay to TypeSafe's Jev* | — | no weights to license | **not a model and not open**: a $JEV-funded proxy to the proprietary API (`Authorization: Bearer`, `api.openjev.sh`) |
| `argos1111/modernbert-ja-310m-jev` | ModernBERT-ja 310 M | 310 M | CC-BY-SA-4.0 | cross-encoder backend for `jev_local`; Japanese |
| `Cactus-Compute/needle3` | own (LSAN) | 29–121 M | **Apache-2.0** | *not* a decision model: tool calls + extraction + embeddings, 8–29 MB, `pip install cactus-needle` (3.0.5, LoRA support) **[HF]** |

Adoption signal **[HF]**: `Cactus-Compute/needle3` 69.6 k downloads, `mys/laya-GGUF`
5.5 k, `Jev-Style-…-GGUF` 4.5 k, `com-kotobalabs/open-jev-deberta-v3-large` 2.1 k.
`convaiinnovations/laya` shows 3 467 likes — high, but its reported download
count is 0, which usually means a mirror/distribution pattern rather than direct
Hub pulls. Treat the likes as enthusiasm, not as usage.

## 3. How to use them (the patterns that actually matter)

1. **One pass, no generation → structured-output error is 0 by construction.**
   Verify that claim on your data anyway: the failure mode moves from
   "malformed output" to "confidently wrong pick".
2. **Gate on confidence, and measure coverage separately.** The useful pair is
   *accuracy at ≥ c* and *coverage at ≥ c* — Tiny-Jev reports 99.8 % accuracy at
   ≥0.9 confidence covering 90 % of items **[rep]**. A model can be right when
   confident and useless if it is never confident (our own micro-scorer rarely
   exceeded 0.9, which caps its usefulness as a bypass).
3. **Per-option independent scoring is not always better than one pass.**
   The OpenJev micro-scorer fires one call per option and softmaxes the results
   **[docs]**; on our substrate the single-pass "read the next token" mode
   **beat** it 10/12 to 6/12 on the same queries, at 1 pass instead of N
   (`DECISION-REPORT.md` §5).
4. **Keep the option list short.** Laya's own guidance: ≤20 options for tool
   choice, shortlist first if the catalogue is bigger **[docs]**. Quality
   degrades with option count and with options sharing a first token (our
   `first_token` mode is vulnerable to exactly that).
5. **Write questions like unit tests.** These models read literally; negations
   and scoping are taken at face value **[rep]**. "Does the ANSWER resolve the
   QUESTION?" is better than "Judge the answer".
6. **Treat the state as untrusted.** Content inside the state can argue for its
   own label; Tiny-Jev was trained against this, and you should still assume it
   **[rep]**.
7. **Re-measure calibration on your own distribution.** The publisher's
   calibration does not transfer: the DeBERTa head reports in-domain 0.854 vs
   OOD 0.690, and its ordered-scale behaviour on *new* level sets is barely above
   majority **[rep]**. This is the single most reproducible finding in the whole
   survey, and our own bench reproduced its shape.
8. **Understand the cost model: it is prefill-bound.** A decision is one forward
   pass *over the entire state*, so a long state costs like a long prefill. Our
   measured micro-scorer: 1.4 ms on a warm 55-token prompt, **0.39–1.10 s on
   real query+answer states** (ternary 8 B, M2). Real Laya measured locally
   (421 M, Metal, torch-free): **61–155 ms per noul, 53 ms per choice** — close
   to the vendor figure, and still ~8× faster than our resident 8 B tier's own
   pass on the same states. The headline "33 ms" / "70 ms" numbers remain
   short-input, warm-model, vendor-hardware figures; only a local run tells you
   what your states cost.
9. **Do not use them for** planning, multi-step reasoning, open-ended writing,
   long-context synthesis, arithmetic, or dates. Every publisher says this.

## 4. Local runnability on this machine (M2, 16 GB, MLX, no torch, ~33 GB free)

Ranked by setup cost, cheapest first. **Items 1 and 2 were run here, not
inferred** — item 2 is the only shipped System One model measured end-to-end on
this box, and it needed no torch.

1. **The micro-scorer already built here** (`darkcore/decisions.py`) — zero new
   weights, zero new dependencies, one forward pass on the resident tier.
   Live-fired; see `DECISION-REPORT.md`.
2. **Laya, torch-free, via `ggmlc` + `mys/laya-GGUF`** — `pip install laya` is
   **off the table** (torch is a core dependency and the `onnx` extra does not
   remove it), but a prebuilt Metal binary plus one ~450 MB Q8_0 GGUF runs the
   real model with **zero Python dependencies**. Both are now durable: the tool at
   `~/.local/share/laya/laya`, the weights in the HF cache at pinned revisions
   (see §Provenance below for hashes and revisions):
   ```bash
   # tool (GitHub release, NOT on Hugging Face): v0.9.5 macos-arm64-metal
   mkdir -p ~/.local/share/laya && cd ~/.local/share/laya
   curl -sL https://github.com/monatis/ggmlc/releases/download/v0.9.5/laya-macos-arm64-metal.tar.gz | tar xz
   # weights (Hugging Face, revision-pinned so the download is reproducible)
   hf download mys/laya-GGUF --revision 713ae6f6e39fb54835e010485656e4484e5ec411 \
     --include "laya_english_q8_0.gguf"
   L="$HOME/.cache/huggingface/hub/models--mys--laya-GGUF/snapshots/713ae6f6e39fb54835e010485656e4484e5ec411/laya_english_q8_0.gguf"
   ~/.local/share/laya/laya serve "$L" --port 8123 --device metal
   ```
   `laya serve` exposes **`POST /v1/systemone`** — the same contract
   `darkcore/decisions.py` already speaks, so `decision_bench.py --backend http`
   drives it with no adapter. Measured here: **53 ms per choice, 61–155 ms per
   noul** (not the advertised 33 ms, but ~8× faster than the resident 8 B tier's
   own pass). Landmine: `max_opts=16` is baked into the prebuilt GGUFs, and a
   larger question set raises an **uncaught C++ exception (SIGABRT)** rather than
   a catchable error — keep label sets small or compile your own GGUF (which
   needs torch).

   **Trust notes for the tool (verified 2026-09-25, not assumed):** it is
   unsigned (adhoc/linker-signed, no TeamIdentifier) and `spctl` rejects it; there
   is no build attestation (`gh attestation verify` → 404) and **no licence file
   in the repo** despite an MIT badge in the README; it binds **all interfaces**
   (`TCP *:PORT (LISTEN)`), not just localhost; its only baked-in network
   endpoints are `127.0.0.1`/`localhost` plus documentation links, and no outbound
   TCP was observed while serving. Treat it as untrusted tooling: short-lived
   sessions, nothing sensitive in the state you send.
3. **Jev-Style 0.8 B v3** (MLX and GGUF builds, Apache-2.0) — the family the
   deeper survey ranks first for a no-torch 16 GB box: a genuine one-pass typed-
   probability model with published accuracy, 25 600-token inputs, 51 languages.
   Its GGUF scoring route wants a `llama.cpp` fork build, so budget an afternoon.
4. **Tiny-Jev 0.6 B** — the best-documented calibration and the closest match to
   our verifier job (its training set explicitly covers guard-railing another
   model's answer). Requires torch → separate process/venv.
5. **APUS-OpenJev-v1-4B** (MLX-4bit ≈ 5.5 GB, peak ~7.4 GB per its card) — the
   higher-quality second choice; the 9 B MLX build is *too* tight on this box.
6. **Ruled out by memory:** every DiffusionGemma-based OpenJev (26 B, 4-bit MLX
   weights are **16.54 GB**) and HF `openjev/openjev` (Qwen3.8-27B, 14.09 GiB at
   4-bit, **CC-BY-NC-4.0** weights). Also `com-kotobalabs/open-jev-deberta-v3-large`:
   excellent documentation and a real span head, but torch-only, git-only install,
   and 1.8 s for 4 questions on M1 Max CPU **[rep]** — too slow for a router hot
   path even if torch were allowed.

**Deferred, not dismissed — the open-jev arm that never got benched.** APUS-OpenJev-v1-4B
is shortlisted at §4 above, Apache-2.0, and an MLX-4bit build exists
(`apus-ailab/APUS-OpenJev-v1-4B-MLX-4bit`; the family also publishes 4B / 9B / 35B-A3B in
GGUF and in MLX at 4-bit and 8-bit). It was left out of the measured round on budget: the
round was scoped to sub-1 B heads to test whether *any* small decision model can carry a
verdict, and that answer came back task-shaped rather than size-shaped — at 0.6–0.8 B the
choice head was solved 12/12 while the verdict head stayed weak. That is a judgement, not
evidence, and APUS-4B is the one open-jev arm that could still falsify it. Price it
properly before running it: its decision runtime is a Python package (`openjet_runtime/`,
plus `deployment/serve_vllm.py`), vLLM-and-torch shaped on a box that runs MLX — so
adopting it means writing a shim, not downloading a file. The remaining open-jev arms were
rejected on grounds, not on size: `SiliconLabAI/OpenJev` is orchestration rather than
weights (and its per-option scoring *lost* to single-pass here, 6/12 vs 10/12), `openjev.sh`
is a hosted relay to the proprietary model, and `zefan-cai/open-jev`, `heman10x/rlcd-*` and
`argos1111/…-jev` (Japanese) publish nothing we can adopt as-is.

## 5. Licensing, for a project that requires permissive only

`AGENTS.md` requires permissively licensed models. Scanning the family:

- **Apache-2.0 / MIT:** Laya (all variants found), Tiny-Jev, Jev-Style,
  com-kotobalabs DeBERTa, APUS, Needle 3, ggmlc (MIT), zefan-cai code (MIT).
- **CC-BY-NC-4.0 — non-commercial:** the HF `openjev/openjev` **weights** (the
  54.7 GB bf16 repo, its FP8/MLX/GGUF quantisations, and the 4-bit MLX build) —
  only its `helper/` and `serve/` *code* is Apache-2.0. Note this is a different
  project from `razorback16/openjev`, whose code *and* DiffusionGemma weights are
  Apache-2.0. Easy trap, and the two share a name.
- **CC-BY-SA-4.0 — share-alike:** `argos1111/modernbert-ja-310m-jev` **[HF]**.
- **MIT:** `SiliconLabAI/OpenJev`'s TypeScript micro-scorer (`LICENSE.md`) — the
  earlier "no licence file" reading came from a secondary source and is wrong.
- **Nothing to license, and not open at all:** `openjev.sh` is a $JEV-funded
  hosted *relay* to TypeSafe's proprietary Jev — no weights, no local execution.
- **NOASSERTION:** the `kotoba-lang/typed-decisions` GitHub repo has no
  recognised SPDX file even though the HF weights are Apache-2.0.

## 6. What patchwork should take from this

- **Adopt the contract, not necessarily a model.** The typed-decision interface
  is the durable part; `darkcore/decisions.py` implements it with no new weights,
  and it already speaks the wire format that `laya serve` and the codiv server
  expose.
- **Do not replace the class predictor.** Real Laya *ties* the embedder kNN
  (12/12) rather than beating it, and adds a service dependency to do so.
- **Use rung 3 as triage, not as a verdict.** `bypass_threshold` /
  `fail_threshold` short-circuit the judge when confident and defer when not —
  the same rule rung 0 already uses for structure.
- **Step 0 is done and the answer was no.** A real shipped calibrated model
  (Laya, 421 M, torch-free via ggmlc) was benched on our own 18 pairs: AUC 0.728,
  22 % false-pass *and* 22 % false-fail, LOO triage 0/18 — worse than a raw
  next-token head from our own tier at the verifier job. Three independent
  off-the-shelf approaches have now failed it, so stop looking for one
  (`DECISION-REPORT.md` §5).
- **The family's real win is the choice job at ~53 ms** — routing, tool
  selection, and the tier decision are where it delivers, and where a shipped
  encoder is both faster and more accurate than our resident 8 B.
- **Watch Needle 3 separately.** It is not a decision model, but a 14 MB
  pip-installable tool-caller with a documented LoRA path is directly relevant to
  the *agentic* tier's function-selection problem, which is a different question
  from verification.

## Provenance (verified 2026-09-25)

Every weight this survey recommends and the decision work measured is now durable in the
Hugging Face cache at a **pinned revision**, and the one piece that is not an HF artifact
(the `ggmlc` runtime) is pinned by SHA-256. Verification method: Hugging Face stores each
file's SHA-256 as its LFS object id
(`/api/models/<id>/tree/<rev>?expand=true&recursive=true`), so a local hash matching the
repo's object id proves the bytes came from that repo — rather than assuming it from a URL.
Byte-identity was confirmed against the original scratch copies before those were deleted.

| artifact | source | revision / tag | size |
|---|---|---|---|
| `laya_english_q8_0.gguf` | HF `mys/laya-GGUF` | `713ae6f6e39fb54835e010485656e4484e5ec411` | 451.5 MB |
| `laya_multilingual_q8_0.gguf` | HF `mys/laya-multilingual-GGUF` | `3b645ae5428115fa5fd1c453070d22c3ce4b987d` | 361.7 MB |
| `onnx/model_quantized.onnx` (int8) | HF `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | `a51e07b524299e309dd2b88d48b0cfa2bd9ec598` | 398.1 MB |
| `onnx/model_quantized.onnx` + `_data` (int8) | HF `onnx-community/bge-small-en-v1.5-ONNX` | `4a9a46c7b88fa408e650a571a1800243f26309bd` | 33 MB |
| `laya` + its release tarball | GitHub `monatis/ggmlc` — **not** Hugging Face | tag `v0.9.5` | 3.2 MB + 1.1 MB |

```
6243305fb16cf22e53b932c220bd029be2adbfbec6ce120356517b8ae2f38382  laya_english_q8_0.gguf
757c1a4b1f0f41824113dde76d6cd06b0881d37b796103a338de77f9f9b935c3  laya_multilingual_q8_0.gguf
4fcd879d3433e2fff506ac86221b4656f6a960653752ee87d087efd3a64cc128  model_quantized.onnx        (ModernBERT zs int8)
4d46eaef91aaa13133a32aafba250a05265c220d1e526797fb55359bfb5a953a  model_quantized.onnx        (bge-small int8 graph)
bf3c4475ba4cac85907418ef34a48dbdfcf628ee91c85415ca6b0c3e70a94cbb  model_quantized.onnx_data   (bge-small int8 weights)
cb20401a39eff32169d6eaf310bfb60cde33cdd68613a802ebf2f1f6a23fde57  laya                        (ggmlc binary)
a1ccbcbcf8d35997c0882cbbe2dae84cf82132b3208918ade3f8da7a4bfd19c5  laya-v0.9.5-macos-arm64-metal.tar.gz
```

Local paths: the four weight sets resolve under
`$HOME/.cache/huggingface/hub/models--<org>--<name>/snapshots/<revision>/`, and the tool
lives at `~/.local/share/laya/laya`. Re-downloads are reproducible with the pinned
revisions, e.g.
`hf download mys/laya-GGUF --revision 713ae6f6e39fb54835e010485656e4484e5ec411 --include "laya_english_q8_0.gguf"`.

Two notes for anyone verifying later: only the **int8 subsets** were fetched from the ONNX
repos, not the whole repository, so a full-tree hash check will not match; and the tool's
trust posture (unsigned, no attestation, no licence file) is recorded in §4 above — it is
pinned for integrity, not vouched for.

## 7. Sources

- Laya: huggingface.co/convaiinnovations/laya · huggingface.co/akshatbindal/laya-typed-decisions ·
  huggingface.co/mys/laya-GGUF (ggmlc compiler route, **not** llama.cpp) ·
  github.com/NandhaKishorM/laya · laya.studio/learn/decision-models-for-ai-agents ·
  pypi.org/project/laya (torch is a core dep) ·
  github.com/monatis/ggmlc/releases (the `laya-macos-arm64-metal.tar.gz` binary used here)
- Tiny-Jev: huggingface.co/lostargon/Tiny-Jev
- Jev-Style: huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16 (+ v3 0.8B repo)
- OpenJev (diffusion): huggingface.co/openjev/openjev · github.com/razorback16/openjev · codiv.ai/docs/models ·
  vllm-project/vllm PR #57250 (merged 2026-09-22) · mlx-community/diffusiongemma-26B-A4B-it-4bit
- OpenJev (micro-scorer): github.com/SiliconLabAI/OpenJev (MIT, homepage jev.opendef.com) ·
  openjev.sh (hosted relay to TypeSafe's Jev, $JEV-funded — not open weights)
- Additional family members: github.com/Mapika/decider · heman10x/rlcd-modernbert-151m ("Verdict")
- Open-Jev (Qwen + head): zefan-cai.github.io/open-jev · github.com/zefan-cai/open-jev
- APUS: huggingface.co/apus-ailab/APUS-OpenJev-v1-4B (+ 9B, 35B-A3B)
- DeBERTa span head: huggingface.co/com-kotobalabs/open-jev-deberta-v3-large · github.com/kotoba-lang/typed-decisions
- Jev Local (ModernBERT backend): github.com/Argos1111/jev_local
- Needle 3: huggingface.co/Cactus-Compute/needle3 · pypi.org/project/cactus-needle
- Jev (closed reference): typesafe.ai · docs.typesafe.ai/llms.txt
