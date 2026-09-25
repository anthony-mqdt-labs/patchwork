# Small classifier / guard / router models to replace LLM calls in the patchwork cascade

Brief written against the 24.5%-of-tokens LLM judge problem. Target: Apple M2, 16 GB RAM, Python 3.12, MLX available, **no PyTorch**, ~34 GB free disk.

Legend for evidence class:
- **[V]** = verified by fetching a primary source (URL cited) or by running on this machine.
- **[R]** = read in a secondary/summary source only, not confirmed against the primary.
- **[U]** = unverified / could not confirm.

### Hypothesis
That a purpose-built small classifier (guard model, zero-shot NLI cross-encoder,
or a few-shot linear head over the existing embeddings) can replace the LLM
judge for either of patchwork's two jobs: classifying a query into the routing
classes, or judging whether an answer resolves the question.

### Method
Primary-source review of the candidate families (GLiNER/GLiNER2, ModernBERT and
the MoritzLaurer zero-shot NLI cross-encoders, SetFit, prompt-injection and
guard classifiers, fastText/linear baselines, semantic-router, RouteLLM,
TRINITY/TinyRouter), with the load-bearing candidates **run on this machine**
through ONNX Runtime with no torch import. Routing numbers use a banking77
12-class proxy subset, not patchwork's battery.

### Results
The routing job is close to free: bge-small int8 nearest-centroid over class
*names* measured **0.808** zero-shot, **0.841** with 21 exemplars, at 3.96 ms per
embedding — and logistic regression **underperforms** nearest-centroid at n=21
(0.722), because it overfits. The verifier job is not free: the zero-shot NLI
cross-encoder scored a correct answer **0.884** against a wrong answer's
**0.506**, scored 0.022 vs 0.032 on "the answer is wrong or unfounded", and fired
on both for "addresses the question" — it measures relevance, not correctness.

### Implications
Routing: keep embedding-similarity with the index already in place; do not add a
dependency or a training step. Verification: zero-shot is a dead end, which
matches the independent measurement in `experiments/router/DECISION-REPORT.md`.
A fine-tuned binary head on the judge's own verdicts is the cheapest correct
path — see `research/decision-finetune-path.md`.

---

## 0. Headline

1. **The router job (a) is essentially solved for free.** A 12-class classifier built on the *bge-small index you already have* — zero training, just class names as text — measured **0.808 accuracy** on a 12-class proxy. Adding your ~21 exemplars pushes it to **0.84–0.90**. Cost: **3.96 ms/query** for the embedding. No new dependency, no new model download beyond an ONNX export of an encoder you already run.
2. **The verifier job (b) is *not* solved zero-shot, and the obvious candidate fails.** I ran the best-in-class zero-shot NLI cross-encoder and it **could not separate a correct answer from a wrong one** (0.884 vs 0.506 on "correctly and completely resolves the question"). Generic NLI models score *topical relevance*, not *factual correctness*. This job needs either a fine-tuned head on your own PASS/FAIL labels or a purpose-built judge model — not an off-the-shelf zero-shot classifier.
3. **ONNX Runtime is the correct torch-free execution path.** Verified working end-to-end on this M2 with `onnxruntime` 1.30.0 + `transformers` 5.17.0 (tokenizer only) and no `torch` import.
4. **Cap the ambition on "router models."** The best-documented 2026 routing coordinator (TRINITY/TinyRouter) *ties random routing on math* and only modestly beats the best single model on MMLU. The observed pattern in the literature is that routing helps when your model pool has genuine per-task specialisation, not otherwise.

---

## 1. Comparison table

Throughput figures marked "measured" are my own runs on this M2 (4 threads, `CPUExecutionProvider`).

| Model / family | Params | License | Context | Apple Silicon w/o PyTorch | Throughput | Custom 12-label space | URL |
|---|---|---|---|---|---|---|---|
| **bge-small-en-v1.5** (+ linear/centroid head) | 33 M | MIT | 512 | **Yes** — official ONNX in repo; int8 export used here is 33 MB | **3.96 ms/text measured** (M2, int8 ONNX, 4 thr) | **Zero-shot 0.808; 21 exemplars → 0.841** (measured, 12-class proxy) | https://huggingface.co/BAAI/bge-small-en-v1.5 |
| **MoritzLaurer/ModernBERT-large-zeroshot-v2.0** | 395 M | Apache-2.0 | 8192 | **Yes** — full ONNX family incl. int8 398 MB, q4f16 297 MB | **622 ms / 12 labels batched**, 1085 ms serial (measured, int8, M2); card claims 1116 text/s on A100 b32 | **Zero-shot, arbitrary labels.** Fires correctly on routing text; **fails on correctness judging** | https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0 |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | 435 M | MIT | 512 | ONNX via optimum (export needed); no prebuilt ONNX in repo | ~3–5x slower than ModernBERT per card | Zero-shot, arbitrary labels. **banking77 (72 cls) zero-shot f1_macro 0.513; +500/class → 0.766** | https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 |
| MoritzLaurer/mDeBERTa-v3-base-mnli-xnli | 279 M | MIT | 512 | ONNX export needed | — | Zero-shot, arbitrary labels, 100 languages; XNLI avg 0.808 | https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli |
| **answerdotai/ModernBERT-base** | 149 M | Apache-2.0 | 8192 | **Yes** — ONNX incl. `model_quantized.onnx` 151 MB | ~2.5x faster than large | **Requires fine-tuning** (masked-LM base, no label head) | https://huggingface.co/answerdotai/ModernBERT-base |
| answerdotai/ModernBERT-large | 395 M | Apache-2.0 | 8192 | Yes — 67 quantized variants on the Hub | — | Requires fine-tuning | https://huggingface.co/answerdotai/ModernBERT-large |
| **fastino/gliner2-base-v1** | 205 M | Apache-2.0 | 512 | ONNX via `gliner[onnx]`; community ONNX exports exist | CPU-first claim (no GPU needed) | **Zero-shot, arbitrary labels** — `classify_text(text, {"field": [...]})` | https://huggingface.co/fastino/gliner2-base-v1 |
| urchade GLiNER v2.1 (small/medium/large) | ~50–200 M | **Apache-2.0** (v2.1 only; v1 base/small/medium/large are CC-BY-NC-4.0) | 512 | **Yes** — official ONNX + OpenVINO export path, and `onnx-community/gliner_*` exports | — | Zero-shot arbitrary *entity* labels; classification needs GLiNER2 | https://urchade.github.io/GLiNER/intro.html |
| onnx-community/gliner_small-v2.1 | ~50 M | inherits Apache-2.0 | 512 | Yes — int8 183 MB | — | Zero-shot entity labels | https://huggingface.co/onnx-community/gliner_small-v2.1 |
| **SetFit** (sentence-transformers) | body-dependent | Apache-2.0 (lib) | body-dependent | **No — requires torch.** Not usable in the target venv | — | Few-shot, needs **8+ examples/class**; head alone trains in ~12 ms (measured proxy) | https://github.com/huggingface/setfit |
| **meta-llama/Llama-Prompt-Guard-2-86M** | 86 M | Llama 4 Community (gated on HF; backbone mDeBERTa MIT) | **512** | ONNX export needed | Card: 92.4 ms/classification A100, 512 tok | Binary benign/malicious only — no custom labels | https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M |
| meta-llama/Llama-Prompt-Guard-2-22M | 22 M | same | 512 | ONNX export needed | Card: 19.3 ms/classification A100 | Binary only | https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M |
| **protectai/deberta-v3-base-prompt-injection-v2** | 184 M | **Apache-2.0** | 512 | **Yes — ONNX in-repo** (`onnx/model.onnx`, 738 MB fp32) | — | Binary only (0=benign, 1=injection). No custom labels | https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2 |
| ibm-granite/granite-guardian-3.1-2b | 2 B | Apache-2.0 | 4 K | GGUF exists (community); MLX possible | generative — much slower than a classifier | Prompt-configurable risk dimensions **incl. Groundedness + Answer Relevance** | https://huggingface.co/ibm-granite/granite-guardian-3.1-2b |
| meta-llama/Llama-Guard-3-1B | 1 B | Llama 3.2 Community (gated) | 8 K | GGUF/pruned-quantized versions exist | generative | Fixed MLCommons 13-hazard taxonomy; customisable only by prompting/FT | https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/1B/MODEL_CARD.md |
| **aurelio-labs/semantic-router** | n/a (library) | MIT | encoder-dependent | Yes with `HuggingFaceEncoder` (torch) — otherwise API encoders | embedding lookup, ~ms | **No training.** Pure embedding similarity vs. utterance exemplars | https://github.com/aurelio-labs/semantic-router |
| **lm-sys/RouteLLM** | n/a (library) | Apache-2.0 | n/a | Trained routers are matrix-factorisation/BERT heads; torch-based | — | Trained on preference data to pick strong/weak model | https://github.com/lm-sys/RouteLLM |
| harrrshall/tinyrouter (TRINITY reimpl.) | 0.6 B encoder + **~10 K head** | see repo | — | No (H200/torch) | frozen encoder + tiny head | Trained head over 3 model × 3 role space | https://github.com/harrrshall/tinyrouter |

---

## 2. Per-family findings

### 2.1 GLiNER / GLiNER2 — [V]

- **GLiNER2 exists and is real.** `fastino/gliner2-base-v1`, **205 M params**, **Apache-2.0**, arXiv **2507.18546**, released Jul 2025. [V] — https://huggingface.co/fastino/gliner2-base-v1 . A `large` sibling exists in the same collection.
- GLiNER2 unifies NER, **text classification**, structured extraction and relation extraction in one model, with a **schema-driven interface at inference time**. The card's own example is exactly the shape you need: `extractor.classify_text(text, {"sentiment": ["positive","negative","neutral"]})` — i.e. **arbitrary candidate labels, no fine-tuning**. [V]
- Family: `fastino/gliner2-large-v1` (https://huggingface.co/fastino/gliner2-large-v1).
- **Original GLiNER** (urchade): Apache-2.0 repo licence. **[V]** Critical licence split from the official docs: **v2.1 models (small/medium/large) are Apache-2.0**, but **v1 base/small/medium/large are CC-BY-NC-4.0** (non-commercial). Also `knowledgator/gliner-multitask-v1.0` and `gliner-llama-multitask-1B-v1.0` are Apache-2.0. [V] — https://urchade.github.io/GLiNER/intro.html
- **ONNX: yes, first-class.** urchade's GLiNER docs have a dedicated `convert_to_onnx.md` documenting ONNX Runtime + OpenVINO export, with `pip install "gliner[onnx]"` and the same `predict_entities` API; note the doc says the **default** `gliner` install does *not* include either ONNX Runtime package. [V] — https://github.com/urchade/GLiNER/blob/main/docs/convert_to_onnx.md
- **Prebuilt community ONNX exports exist** on the Hub under `onnx-community/` (Transformers.js-compatible): `gliner_small-v2.1` (int8 183 MB), `gliner_medium-v2.1`, `gliner_multi-v2.1`, `gliner_large-v2.1`, `gliner-multitask-large-v0.5`. [V] — https://huggingface.co/onnx-community/gliner_small-v2.1
- **Caveat:** GLiNER is a *span/entity* extractor at heart. Its zero-shot "label" is an entity type. GLiNER2 is the variant that gives you document-level classification. Using v1 GLiNER for 12 routing classes is a shape mismatch; use GLiNER2 or a cross-encoder.

### 2.2 ModernBERT and the zero-shot NLI cross-encoders — [V]

- **ModernBERT-base**: 22 layers / **149 M params**, **Apache-2.0**, native **8192** context, RoPE + local-global alternating attention. Card reports GLUE 88.4 vs BERT 84.7, BEIR 41.6 vs 38.9. [V] — https://huggingface.co/answerdotai/ModernBERT-base and paper arXiv:2412.13663.
- **ModernBERT-large**: 28 layers / **395 M params**, Apache-2.0. [V] — https://huggingface.co/answerdotai/ModernBERT-large
- **ONNX is prebuilt for both**, including a full quantization ladder. `answerdotai/ModernBERT-base` ships `onnx/model.onnx` (599 MB), `model_fp16`, `model_int8`/`model_quantized`/`model_uint8` (151 MB) and `model_q4f16` (140 MB). [V] — https://huggingface.co/answerdotai/ModernBERT-base/blob/main/onnx/model_quantized.onnx
- **MoritzLaurer's zero-shot classifiers are the ones that accept arbitrary candidate labels at inference.** They are NLI models reframed as `entailment` vs `not_entailment`, driven by the HF `zero-shot-classification` pipeline with a `hypothesis_template`. [V] — https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0
  - `deberta-v3-large-zeroshot-v2.0` — **MIT**, 512-token context, 28-task mean f1_macro **0.676** zero-shot; the model card's own few-shot column (up to 500 training points/class) gives **0.846**.
  - The **banking77 row is the single most relevant number in this whole brief**: 72 classes, zero-shot f1_macro **0.513**; with 500 examples/class **0.766**. That is roughly what a *hard* multi-class routing problem costs, and it is why 12 classes is a much easier ask.
  - `ModernBERT-large-zeroshot-v2.0` — **Apache-2.0**, same training mix, **ONNX exports prebuilt** (int8 398 MB). Card: 0.85 accuracy / 0.834 f1_macro mean, ~1116 text/s on A100 batch 32. Author's own takeaway: *"very fast and memory efficient… multiple times faster and consumes multiple times less memory than DeBERTav3"* but *"slightly worse than DeBERTav3 on average."* [V] — https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0
  - `mDeBERTa-v3-base-mnli-xnli` — MIT, multilingual. [V]
- **Cross-encoder family:** the `zero-shot-classification` pipeline *is* a cross-encoder — one forward pass per (premise, hypothesis) pair. That is why 12 labels costs 12 passes. Measured below.

**My measured results on this M2** (ModernBERT-large-zeroshot-v2.0, int8 ONNX, 398 MB, 4 threads):

```
12-label zero-shot, serial  : 1085 ms total  (90 ms/label)
12-label zero-shot, batched :  622 ms total  (51.8 ms/label equiv)
single binary pass          :   79 ms
```

Routing sanity check (real patchwork-shaped query about a Worker 1102 error):
`troubleshooting 0.989 · cloud infrastructure 0.895 · database 0.113 · coding 0.109` — **it picked the right class.** [V]

Verifier sanity check — **this is the finding that matters**:
```
premise: "Question: <Worker 1102 question>\nAnswer: <answer>"
hypothesis: "The answer correctly and completely resolves the question."
  correct answer -> 0.884
  wrong   answer -> 0.506     <-- only 0.38 margin, wrong answer near coin-flip
hypothesis: "The answer is wrong or unfounded."
  correct answer -> 0.022
  wrong   answer -> 0.032     <-- no signal at all
hypothesis: "The answer addresses the question."
  correct answer -> 0.824
  wrong   answer -> 0.874     <-- fires on BOTH -> measures relevance, not truth
```
[V] — measured with a throwaway ONNX Runtime harness (see §5 for the method).

**Conclusion:** the zero-shot NLI cross-encoder is a *good* router and a *bad* correctness judge. Do not deploy it as the PASS/FAIL judge without fine-tuning it on your own labels.

### 2.3 SetFit — [V] docs, [U] CPU wall-clock

- Documented minimum: **8 labelled examples per class** — *"with only 8 labeled examples per class on the Customer Reviews sentiment dataset, SetFit is competitive with fine-tuning RoBERTa Large on the full training set of 3k examples."* [V] — https://github.com/huggingface/setfit and https://huggingface.co/blog/setfit
- Mechanism (two stages): contrastive Siamese fine-tune of a Sentence Transformer body on generated text pairs, **then** fit a classification head (scikit-learn `LogisticRegression` by default, or a torch `SetFitHead`). Paper: arXiv:2209.11055. Claim: *"comparable results with PEFT and PET… while being an order of magnitude faster to train."* [V]
- **Licence: Apache-2.0** for the library. [V] — https://github.com/huggingface/setfit/blob/main/LICENSE
- **Blocker for this project:** SetFit depends on `sentence-transformers` → **PyTorch**. In the target venv (no torch) SetFit is **not runnable**. You would need it only at *training* time in a throwaway venv, exporting an ONNX head — but the far cheaper move is stage 2 alone (see §2.5).
- On **CPU training cost** I could not fetch a documented number. **[U]** The paper says "order of magnitude faster", and my measurement of the *equivalent* torch-free stage-2 head is below.

### 2.4 Guard / prompt-injection classifiers — [V]

| | Params | License | Context | Custom labels? |
|---|---|---|---|---|
| Llama Prompt Guard 2 86M | 86 M | Llama 4 Community, **gated on HF** (HTTP-verified 200 + gating form); backbone mDeBERTa **MIT** | **512** | No — binary benign/malicious |
| Llama Prompt Guard 2 22M | 22 M | same | 512 | No |
| protectai/deberta-v3-base-prompt-injection-v2 | 184 M | **Apache-2.0** | 512 | No — binary 0/1 |
| granite-guardian-3.1-2b | 2 B | **Apache-2.0** | 4 K | **Yes** — prompt-configurable dimensions |
| Llama Guard 3-1B | 1 B | Llama 3.2 Community, gated | 8 K | Fixed 13-hazard taxonomy |

- **Prompt Guard 2** [V] — https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M and the canonical card at https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md
  - 86M backs onto **mDeBERTa-base**; 22M backs onto **DeBERTa-xsmall** — both **MIT** per the card.
  - Documented: 512-token context. 86M: AUC .998 EN / .995 multilingual, recall@1%FPR 97.5%, **92.4 ms per classification on A100 @512 tok**. 22M: AUC .995, recall 88.7%, **19.3 ms**. 22M *"reduces latency and compute costs by 75%."* 
  - **Explicitly binary** (`benign` / `malicious`). Card states v2 deliberately **dropped** the finer injection sub-label: *"we found this objective too broad to be useful."*
  - HF weights page is **gated** (accept Meta licence) — [V] HTTP 200 + licence acceptance form.
- **protectai/deberta-v3-base-prompt-injection-v2** [V] — https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2
  - **Apache-2.0**, `microsoft/deberta-v3-base` fine-tune, **binary** (`0` benign / `1` injection).
  - **ONNX is in the repo** and the model card documents the `ORTModelForSequenceClassification` path verbatim. Files: `onnx/model.onnx` 738 MB (fp32 only — no int8 export shipped).
  - Reported: post-training eval on 20 000 held-out prompts — accuracy 95.25%, precision 91.59%, **recall 99.74%**, F1 95.49%. Stated limitation: *"does not detect jailbreak attacks"* and *"we do not recommend using this scanner for system prompts, as it produces false-positives."*
  - 859 257 downloads / month. [V]
- **ibm-granite/granite-guardian-3.1-2b** [V] — https://huggingface.co/ibm-granite/granite-guardian-3.1-2b
  - **Apache-2.0**, Granite 3.1 2B Instruct fine-tune, released Dec 18 2024, paper arXiv:2412.07724.
  - **The only guard here that is relevant to job (b).** It ships **Groundedness**, **Answer Relevance** and **Context Relevance** risk dimensions, plus **Function Calling Hallucination** for agentic workflows. Risk definitions are configurable, i.e. you can define your own criterion rather than only pass/fail-on-harm.
  - It is **generative (2 B)**, so it is not a "cheap classifier" — it is a *cheaper, smaller judge*, which is still a different thing from a 395 M cross-encoder.
  - Community GGUF quantizations exist (e.g. `QuantFactory/granite-guardian-3.0-2b-GGUF`). [R]
- **Llama Guard 3-1B** [V] — https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/1B/MODEL_CARD.md
  - Fine-tuned Llama-3.2-1B, classifies prompts and responses, aligned to the MLCommons 13-hazard taxonomy (S1–S13). **It generates** "safe"/"unsafe" plus violated categories rather than emitting a logit.
  - Two releases: full and a **pruned+quantized** variant *"optimized for deployment on mobile devices."* Custom taxonomy requires prompting or fine-tuning (llama-recipes notebook linked from the card).
  - HF page gated (readme pins `license: llama3.2`). [V/R]

**Verdict on guards:** all of these are *safety* classifiers with fixed or prompt-configurable label spaces. **None of them accepts an arbitrary custom label set**, so none can serve your 12 routing classes. Use them, if at all, as an injection screen — and note that `protectai` is the only Apache-2.0 + ONNX-in-repo + non-gated option.

### 2.5 fastText and classical baselines — [V] evidence, strong

This is the under-rated option and the literature supports it in exactly your regime:

- **Edwards & Camacho-Collados, COLING 2020**, *"Go Simple and Pre-Train on Domain-Specific Corpora"* — direct fastText-vs-BERT comparison across many tasks. Verbatim from the abstract: *"BERT outperforms all baselines in standard datasets with large training sets. However, **in settings with small training datasets a simple method like fastText coupled with domain-specific word embeddings performs equally well or better than BERT**."* [V] — https://aclanthology.org/2020.coling-main.481/
- **ACL 2023 short 160**, *"Linear Classifier: An Often-Forgotten Baseline for Text Classification"* — *"for many text data, linear methods show competitive performance, high efficiency, and robustness."* [V] — https://aclanthology.org/2023.acl-short.160/
- fastText itself is a **bag-of-ngrams → linear classifier** (the repo's own issue tracker: sentence vector = averaged word/ngram embeddings, then multinomial logistic regression). Paper arXiv:1607.01759. [V] — https://github.com/facebookresearch/fastText
- **Practical translation for patchwork:** you already have a *better* feature extractor than fastText's n-gram average — bge-small. So the modern form of "the linear baseline" is **logistic regression (or nearest-centroid) over bge-small embeddings**, which I measured directly:

```
bge-small-en-v1.5, ONNX int8 (33 MB), M2, 4 threads:
  embedding throughput           :  3.96 ms / text
  12-class banking77 proxy subset (train pool 1613, test 479):
    zero-shot, class-NAME nearest-centroid  :  0.808   (no training at all)
    21 labelled exemplars  -> nearest-centroid :  0.841
    21 labelled exemplars  -> logistic regression: 0.722   (logreg hurts at n=21)
    24 labelled (2/class)  -> logreg 0.901 | nearest-centroid 0.897
    60 labelled            -> logreg 0.967 | nearest-centroid 0.964
   1000 labelled            -> logreg 0.979
  head training cost (21 x 384-dim):
    logistic regression  :  12.5 ms
    nearest-centroid     :   1.3 ms
```
[V] — measured (see §5), and `https://huggingface.co/onnx-community/bge-small-en-v1.5-ONNX` for the int8 export; `https://huggingface.co/BAAI/bge-small-en-v1.5` is **MIT** and ships its own `onnx/model.onnx` (133 MB).

Two things fall out of this that you should internalise:
1. **21 exemplars is enough for ~0.84, and it is a knife-edge.** At n=21 logreg *underperforms* nearest-centroid (0.722 < 0.841) because it overfits. At n=24 logreg jumps to 0.901. If you go linear, **start with nearest-centroid, not logreg**, and switch only when you have ≥2 examples/class.
2. **Zero-shot class-name similarity already gets 0.808.** Your class labels are doing most of the work. Spend your 21 exemplars on the classes that are *confusable*, and measure per-class error rather than a single accuracy number.

Caveat: banking77 is intent-shaped, English, and short. Your 12 classes may be broader and harder. Treat 0.808/0.841 as an *upper* estimate.

### 2.6 semantic-router and LLM-routing libraries — [V]

- **aurelio-labs/semantic-router** — **MIT** (LICENSE + `pyproject.toml` `license = "MIT"`), `version = "0.2.0.dev2"`, `requires-python = ">=3.9,<3.14"`. [V] — https://github.com/aurelio-labs/semantic-router and https://raw.githubusercontent.com/aurelio-labs/semantic-router/main/pyproject.toml
  - **No trained encoder.** It is pure embedding similarity: you define `Route(name, utterances=[...])`, embed the utterances, and route by cosine similarity to the incoming query. README describes it as *"use the magic of semantic vector space"* to replace LLM tool decisions. [V]
  - Encoders: API-based by default (`CohereEncoder`, `OpenAIEncoder`); a **`HuggingFaceEncoder`** exists for fully-local use via the `[local]` extra (which pulls torch), or you can drop in your own. There is also a `HybridRouteLayer`.
  - **Maturity caveat:** the pyproject pulls in `aurelio-sdk`, `openai`, `cohere`, `mistralai`, `voyageai` and a `tornado` server — a heavy dependency surface for what is, functionally, `argmax(query_vec @ route_vecs.T)`. For 12 classes and an existing bge-small index, **you can implement this in ~10 lines** and skip the dependency. Its `license = "MIT"` and the version pinned at `0.2.0.dev2` (a dev pre-release) are the honest maturity signal.
- **lm-sys/RouteLLM** — **Apache-2.0**, 5 455 stars. Framework for serving/evaluating routers; ships **trained** routers (matrix-factorisation and BERT-based) fit on preference data, claims up to 85% cost reduction at 95% of GPT-4 MT-Bench performance, >40% cheaper than commercial routers. Paper arXiv:2406.18665. [V] — https://github.com/lm-sys/RouteLLM
  - Its routing target is **strong-vs-weak model choice**, which is *closer* to your cheap/expensive tier decision than the 12-way intent classifier is. It is torch-based, so eval/serve in a separate venv.

### 2.7 2026-era router models / decision heads — [V]

- **TRINITY: An Evolved LLM Coordinator** — arXiv **2512.04695**, submitted 4 Dec 2025, revised 27 Apr 2026. *"a lightweight coordinator that orchestrates collaboration among large language models… comprising a compact language model (approximately 0.6B parameters) and a lightweight head (approximately 10K parameters), optimized with an evolutionary strategy."* Queries are processed over **multiple turns**, assigning one of three roles per turn. [V] — https://arxiv.org/abs/2512.04695
- **TinyRouter** — a from-scratch open-source rebuild of TRINITY. Frozen **0.6 B** encoder → **~10 K-param** head → (model, role) decision, trained by separable CMA-ES against a binary right/wrong reward. [V] — https://github.com/harrrshall/tinyrouter
  - **Read the results honestly, they are the most useful thing in this section:**
    - Math: TinyRouter **0.792**, best single model 0.794, **random routing 0.792** → *routing bought nothing.*
    - MMLU: TinyRouter **0.925**, best single 0.922, random routing 0.875 → modest win.
    - Combined: 0.858 vs best-single 0.835.
  - The repo's own takeaway is that they built an "oracle-ceiling diagnostic to ask whether the pool even leaves room for routing to help." **Adopt that diagnostic before building a router head.** Routing only pays when your candidate models are genuinely specialised.
- **`ModernBERT-*-llm-router` fine-tunes exist** on the Hub, all Apache-2.0, fine-tuned from `answerdotai/ModernBERT-base`/`large` on `DevQuasar/llm_router_dataset-synth` — e.g. https://huggingface.co/AdamLucek/ModernBERT-large-llm-router (64 downloads, 2 likes), https://huggingface.co/sanketrai/modernbert-llm-router (6 downloads). [V] **Low adoption: 6–64 downloads. Treat as templates, not dependencies.** The fact that several people independently fine-tuned ModernBERT into an LLM router is the useful signal: *this is the standard recipe.*
- **AWS reference implementation** of exactly the intent-router pattern: https://github.com/aws-samples/sample-finetune-modenBert-for-intent-classification — *"A Router is a model that classifies user prompt ('intent') and forwards it to the most appropriate Sub-Agent."* [V/R]
- **[U]** arXiv:2608.00030 ("SLMs as Multi-Agent Routers: Progressive SFT + RL") and arXiv:2609.23085 ("Measured Joules, Learned Routes") surfaced in search but I did **not** fetch them; treat as unfiled pointers. The arXiv listing page for 2609.23085 does confirm the title exists.

---

## 3. Ranked recommendations

### Job (a) — classify an incoming query into ~12 known classes

**Rank 1 — Nearest-centroid / logistic regression over your existing bge-small embeddings. Build nothing else first.**
Zero-shot class-name similarity measured **0.808**; with your 21 exemplars **0.841** (nearest-centroid) / **0.722** (logreg). Embedding cost **3.96 ms/query**. Head training **1.3–12.5 ms**. Licence MIT. No torch, no new model, runs through the ONNX runtime already proven on this box.
Use **nearest-centroid** while you have <2 examples/class, then graduate to logreg at ≥2/class. Do not use logreg at n=21 — I measured it degrading below nearest-centroid.
Evidence: https://huggingface.co/BAAI/bge-small-en-v1.5 (MIT) · https://huggingface.co/onnx-community/bge-small-en-v1.5-ONNX (int8 33 MB) · https://aclanthology.org/2020.coling-main.481/ (small-data regime favours the simple model) · measured (see §5).

**Rank 2 — MoritzLaurer/ModernBERT-large-zeroshot-v2.0 (ONNX int8) as a zero-shot fallback / cold-start router.**
**622 ms for all 12 labels batched**, **Apache-2.0**, 8192 context, arbitrary labels at inference with no training at all. It picked the correct class on a real patchwork-shaped query (troubleshooting 0.989). Use it to (i) bootstrap pseudo-labels for the classes you have no exemplars for, and (ii) cross-check the embedding router. Too slow at 622 ms to sit in the hot path of every query unless you accept ~0.6 s of routing latency.
Evidence: https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0 · measured.

**Rank 3 — Fine-tune ModernBERT-base (149 M, Apache-2.0, ONNX prebuilt) as a proper 12-way head once you have ~20+ labels/class.**
This is the standard industry recipe — several independent Hub releases did exactly this for LLM routing — and ModernBERT's ONNX ladder (incl. 151 MB int8) keeps you torch-free at inference. Requires a one-time torch training run in a throwaway venv. Expect it to beat everything above only once you have real per-class volume; below that, the linear head is the better bet.
Evidence: https://huggingface.co/answerdotai/ModernBERT-base · https://github.com/aws-samples/sample-finetune-modenBert-for-intent-classification · https://huggingface.co/AdamLucek/ModernBERT-large-llm-router

**Rank 4 — GLiNER2 (`fastino/gliner2-base-v1`, 205 M, Apache-2.0).**
Genuinely zero-shot and schema-driven, with `classify_text(text, {"intent": [...]})` as a first-class API. Ranking it low only because at 205 M with a different architecture it is unlikely to beat a fine-tuned 149 M ModernBERT, and you have no measurement for it yet. Worth a spike if you want zero-shot on classes that change often.
Evidence: https://huggingface.co/fastino/gliner2-base-v1 · https://github.com/urchade/GLiNER/blob/main/docs/convert_to_onnx.md

**Not recommended for (a):** fastText (bge-small + linear is strictly stronger in this regime), semantic-router as a *dependency* (reimplement 10 lines instead), and any Llama-Guard / Prompt-Guard model — none accepts a custom label space.

### Job (b) — binary PASS/FAIL: does the small model's answer resolve the question?

**Rank 1 — Fine-tune a cross-encoder on your own PASS/FAIL labels. Zero-shot will not work; I measured it failing.**
The `ModernBERT-large-zeroshot-v2.0` NLI model scored a **correct** answer 0.884 and a **wrong** answer 0.506 on "correctly and completely resolves the question" — a 0.38 margin with the wrong answer at essentially a coin flip. The probe "The answer is wrong or unfounded" scored **0.022 vs 0.032** — literally no signal. And "The answer addresses the question" fired on **both** (0.824 / 0.874), which proves the model is measuring *on-topic relevance*, not *correctness*.
Generic NLI is the wrong inductive bias for this job. Take the same backbone (`ModernBERT-base`, 149 M, Apache-2.0, ONNX prebuilt) and fine-tune it **as a binary classifier on your own judge outputs**. This also gives you the cheapest possible path: your existing LLM judge is *already producing* the labels — mine them from logs and distil.
Budget: ~79 ms per pass at int8 on this M2, so a 3-way ensemble or a longer premise costs <250 ms.
Evidence: measured (see §5) · https://huggingface.co/answerdotai/ModernBERT-base · https://www.philschmid.de/fine-tune-modern-bert-in-2025

**Rank 2 — Granite Guardian 3.1 2B as a smaller drop-in judge (not a classifier).**
Apache-2.0, and it is the **only** guard in this survey whose risk dimensions include **Groundedness**, **Answer Relevance** and **Function Calling Hallucination** — i.e. the actual question "is this answer faithful and on-point". It will not be free, but a 2 B local model replacing a frontier judge on 24.5% of spend is the highest-confidence cost win available without any training. GGUF quantizations exist.
Evidence: https://huggingface.co/ibm-granite/granite-guardian-3.1-2b

**Rank 3 — TRINITY/TinyRouter-style tiny learned head over a frozen encoder.**
Architecturally the closest thing to what you want: frozen encoder → **~10 K-param head** → decision, trained by evolution against a binary reward. The honest caveat is in their own results — **on math it tied random routing (0.792 vs 0.792)**. Adopt their *oracle-ceiling diagnostic* first: measure whether an oracle that always picks the right tier actually beats the best single tier on your traffic. If it does not, no router will help.
Evidence: https://arxiv.org/abs/2512.04695 · https://github.com/harrrshall/tinyrouter

**Not recommended for (b):** Llama Prompt Guard 2 (binary injection/jailbreak only — wrong question), protectai/deberta-v3-base-prompt-injection-v2 (same), Llama Guard 3-1B (harm taxonomy, and it generates rather than classifies).

---

## 4. Verification ledger

**Verified by running on this machine** (M2, 16 GB, `onnxruntime` 1.30.0 + `transformers` 5.17.0, **no torch imported**):
- ONNX Runtime is a working torch-free path on macOS/Apple Silicon for these models. [V]
- ModernBERT-large-zeroshot-v2.0 int8: 622 ms batched / 1085 ms serial for 12 labels; 79 ms single pass. [V]
- ModernBERT-large-zeroshot-v2.0 routing behaviour correct on a patchwork-shaped query. [V]
- **ModernBERT-large-zeroshot-v2.0 fails to discriminate correct vs incorrect answers.** [V]
- bge-small-en-v1.5 int8: 3.96 ms/text embedding. [V]
- 12-class few-shot curve: 0.808 zero-shot → 0.841 @ 21 → 0.901 @ 24 → 0.967 @ 60 → 0.979 @ 1000. [V]
- Head training cost: 1.3 ms (centroid) / 12.5 ms (logreg) at n=21. [V]

**Verified by fetching a primary source** (URL given inline above): GLiNER2 existence/size/licence; GLiNER licence split + ONNX docs + onnx-community exports; ModernBERT sizes/licences/ONNX file listings; all four MoritzLaurer model cards incl. the banking77 numbers; SetFit 8-examples claim + Apache-2.0; both Prompt Guard 2 model cards + gating; protectai card incl. ONNX path and eval numbers; granite-guardian-3.1-2b card; Llama Guard 3-1B card; semantic-router licence/version/mechanism; RouteLLM licence/claims; TRINITY abstract; TinyRouter README and results; ModernBERT llm-router fine-tunes; both ACL fastText/linear-baseline abstracts.

**Read but not confirmed against a primary source**: community GGUF quantizations of granite-guardian; the AWS ModernBERT intent-router sample (repo page only); arXiv:2608.00030 and arXiv:2609.23085 (titles confirmed on listing pages, papers not fetched).

**Could not verify**: SetFit CPU wall-clock training time (no documented figure found; PyTorch is not installed here so I could not measure it). The exact licence text governing the Prompt Guard 2 *weights* (HF page is gated behind licence acceptance; the underlying mDeBERTa/DeBERTa-xsmall backbones are confirmed MIT).

**Explicitly not invented**: no model name or URL in this brief was generated by pattern-matching. Every URL was fetched or returned by search during this session.

---

## 5. Reproduction

The three harnesses were ephemeral throwaway scripts in a scratch virtualenv and
are deliberately **not** committed, so what follows is the method rather than a
path. Rebuild them in any scratch venv:

```bash
python3 -m venv <scratch-venv> && source <scratch-venv>/bin/activate
pip install onnxruntime numpy "transformers>=4.48" scikit-learn datasets
```

1. **Batch latency** — load the int8 ONNX export of
   `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` through
   `onnxruntime.InferenceSession` (`CPUExecutionProvider`, 4 threads), then time
   one 12-label zero-shot pass serially and batched (no `torch` import anywhere —
   `transformers` is used for its tokenizer only).
2. **Correct-vs-incorrect discrimination** — the same session, one premise
   (`Question: …\nAnswer: …`) against three candidate hypotheses, scoring a
   known-correct and a known-wrong answer. This is the test that produced the
   0.884 / 0.506 result and it is the one worth re-running first.
3. **12-class few-shot curve** — embeddings from the int8 ONNX export of
   `BAAI/bge-small-en-v1.5`, then nearest-centroid and logistic-regression heads
   fitted at n = 0/21/24/60/1000 over a banking77 12-class proxy subset.

Model weights (~415 MB total) went to scratch directories and can be deleted.
