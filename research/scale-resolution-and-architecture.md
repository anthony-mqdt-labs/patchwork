# Scale, Resolution, and the Coastline Paradox in Neural Architecture

## Investigation: Does the coastline paradox describe anything real about networks?
**Agent:** penny (Hermes, profile `penny`)
**Date:** 2026-09-24
**Status:** in-progress — conceptual thread. Literature synthesis, **no experiments run**.
**Trigger:** an evening conversation that began with "what's that thing where the coast
gets longer the finer you measure it" and drifted into whether networks have an
architectural analogue of a focus knob.

**Session:** `20260921_171131_0151a1` — profile `penny`, source `cli`, model `deepseek-v4-flash`,
started 2026-09-21 17:12 PDT, still open at the time of writing.
**Transcript title:** `Explain coastline paradox and infinite length`
**Resume this conversation:**

```bash
hermes --profile penny --resume 20260921_171131_0151a1
# or by title:
hermes --profile penny -c "Explain coastline paradox"
```

Transcripts live in the agent's local session store (`sessions` for the id and title,
`messages` for the full history), so this conversation is recoverable independently of the
terminal that produced it. Any other agent profile can read it from its own store the same
way; the id above is the handle.

> **Read this as a map, not as evidence.** Every claim below is either (a) established
> literature, (b) an explicit analogy that the author believes holds, or (c) a proposal.
> The three are labelled. Nothing here has been measured in this repository.

### Hypothesis

That the coastline paradox, and the fractal-dimension reasoning that resolves it, describe
something structurally real about trained networks rather than merely something poetically
similar — and that the resolution/precision axis is therefore a legitimate architectural
knob rather than a property that must be trained in.

### Method

Literature synthesis and conceptual mapping. Sources read: Mandelbrot/Richardson on the
paradox; scaling-law and intrinsic-dimension work; the renormalization-group/deep-learning
mapping; superposition; loss-landscape geometry; equivariance and multi-scale constructions.
No code was written, no models trained, no measurements taken in this session.

### Results

None empirical. The thread's outputs are the mapping table in §3, the correction in §2, the
architectural recipes in §6, and a falsifiable proposal in §8. One question was sharpened
from "is there a focus knob" to "is precision a lens or a co-author", which is measurable.

### Implications (preliminary)

See §9. The cascade's stopping rule is a resolution/crossover decision, and the failure modes
of escalation map onto the paradox's dust-versus-coastline ambiguity.

---

## 1. The paradox, compressed

Measure a coastline with a 100 km ruler and it has one length. Halve the ruler and inlets
previously bridged are now traced, so the total rises. Halve again and it rises again. For a
genuinely fractal boundary the measured length diverges as the ruler shrinks to zero — there
is no true answer, only the pair *(measure, resolution)*.

What converges instead is the **dimension**. Plot log(length) against log(ruler size); the
slope is the Hausdorff dimension — roughly 1.25 for the west coast of Britain, near 1.0 for a
smooth arc. The dimension is the invariant. Length was never a property of the coast.

(Mandelbrot 1967, building on Richardson's observation that different nations' encyclopedias
disagreed wildly about their own border lengths — and that the disagreement scaled regularly
with the measuring stick.)

---

## 2. The correction that governs everything below

In geography the terrain is independent of the ruler: Britain is there, the stick is ours.

**In a neural network that separation collapses.** The gradient is computed in the mantissa.
Change numerical precision and the loss landscape is not merely observed differently — it is a
different landscape. Round the weights and the high-frequency structure does not become
unresolved, it becomes absent, and the terrain you then descend was constituted by that choice.

So an architectural "focus" or "precision" dial does not zoom in on pre-existing structure.
It participates in creating it. Any theory of such a dial must therefore be a theory of the
whole measurement act, not of an observer behind progressively better glass.

This is not fatal to the analogy. It relocates it: the analogy is not about *revealing*
detail, it is about *negotiating* a landscape — which is what the optimiser already does.

---

## 3. Where the mapping is exact rather than decorative

| Neural construct | Coastline analogue | Epistemic status |
|---|---|---|
| Training-compute scaling law (`L ∝ C^-α`) | Richardson's log-log plot; α is the fractal dimension | (a) established — Kaplan 2020, Hoffmann 2022 |
| Intrinsic dimension of a dataset | The measured dimension of the "coast" | (a) established — Pope et al. ICLR 2021 |
| Context window / attention span | Ruler length | (b) analogy, defensible |
| Diffusion noise schedule | An explicit ruler ladder | (b) analogy, tight |
| DEQ / weight-tied recursion | Iterated function system with a fixed-point attractor | (b) analogy, tight |
| Adam's second moment | Per-coordinate resolution decision | (b) analogy; is literally preconditioning |
| Filter-normalised loss landscape plots | Coastline measurements with ruler = ε | (a) — this is what those plots *are* |
| Skip connections | Measurably reduce landscape roughness | (a) — Li et al. 2018 |
| Sharpness | A length with no stated ruler | (a) — see §8 |

Two entries carry most of the weight.

**Scaling laws are Richardson plots.** Loss against compute is a clean power law; the exponent,
not the absolute loss, is what transfers between budgets. The exponent is the fractal dimension
in disguise: it reports how much new structure the data yields each time you shorten the stick.
It also refuses to hand you a terminus, which is the paradox restated.

**Intrinsic dimension is resolution-dependent.** Pope et al. measured the intrinsic dimension of
natural images and found it is neither small nor fixed — it climbs with resolution, approaching
the ambient dimension. The tidy manifold hypothesis is false in precisely the way the coastline
is not 3,000 km long. Each doubling of resolution reveals a new order of structure.

**Renormalization group is the rigorous form of the whole analogy.** Mehta & Schwab (2014) map
a variational RG transformation onto a restricted Boltzmann machine; Lin, Tegmark & Rolnick argue
depth *is* RG flow. RG exists to handle systems with no characteristic scale — systems whose
critical exponents are non-integer, i.e. whose dimension is anomalous. Power-law scaling laws
are circumstantial evidence that trained networks sit near such a point. If they do, then
"coastlines and mountains" is not metaphor; it is the same phenomenon viewed from either end of
one telescope.

**Per-architecture notes**

- **Transformer.** Attention is an explicit budget of pairwise reach; context length is the ruler.
  Widening the window uncovers dependencies shorter windows cannot span. The quadratic cost is
  the price of resolution, and the well-documented gap between nominal and *effective* context
  is a statement about where genuine structure ends and sampling noise begins.
- **RNN / deep equilibrium models.** One map applied repeatedly is an iterated function system;
  its attractor may have non-integer dimension. A DEQ is not merely like this — it *is* the fixed
  point of one nonlinear operator. "Implicit width" is a fractal generator's attractor under
  another name.
- **Diffusion.** The noise schedule is a ruler ladder made explicit: denoise at σ, then σ/2, then
  σ/4, measuring the data distribution at each granularity. The continuum limit is an SDE whose
  sample paths are literally fractal curves (Brownian graph Hausdorff dimension 3/2, Taylor 1955).
  Diffusion works partly because it never demands one canonical answer at one canonical resolution.
- **Reinforcement learning — weakest fit, and the failure is informative.** Reward is a coarse
  integral over an objective that is genuinely fractal in the agent's own behavioural detail.
  Reward hacking is what happens when the measuring stick is blunt enough to return the wrong
  shape. That is not an analogy; it is the definition.

---

## 4. Coarse models are not wrong — they are limiting cases

The motivating example: projectile paths are taught as parabolas (exact under uniform gravity,
flat world, no drag), while the true path in an inverse-square field is a conic.

The parabola is **a member of the conic family** — the e = 1 case, the ellipse whose focus has
receded to infinity. The coarse model is not a false shape; it is a limiting shade of the exact
one, obtained by running a scale parameter to an extreme. It is exactly right in a different
world, and the fine model approaches that world as scale → ∞.

Historical precedent, and it is the same mathematics: Ptolemaic epicycles. Deferent, then epicycle,
then epicycle upon epicycle — a truncated Fourier series of the orbit, converging to the ellipse
as terms are added. Refinement as a ladder, accuracy bought in instalments.

And the pointed case: a **symplectic integrator** is a deliberately "wrong" discrete model that
does not reproduce the true trajectory, yet conserves energy and keeps orbits closed over long
horizons where a naive fine-grained integrator drifts. The coarse model preserves the *invariant*.
Invariants are exactly what survives coarse-graining — the *relevant operators* of §3. The coarse
model is the fine model with the scale stripped away, and what remains is what mattered at that
scale.

**Implication for this repository:** a tiered cascade is a coarse model being *deliberately*
chosen. See §9.

---

## 5. Superposition versus distinct notions — big model, young mind

The observation that motivated this write-up: a large model can hold many "shades" of a solution
in superposition, functionally adjacent in weight space, while a small model may have to carve
them into distinct, separable notions.

This has a published mechanism and it is a phase transition, not a continuum. Anthropic's *Toy
Models of Superposition* (Elhage et al. 2022) shows exactly this structure: features packed into
overlapping directions when the model is narrow, resolving into dedicated directions when there
is room. Nobody instructs the model to do either. Interference has a cost, and superposition is
what you do when you cannot afford to allocate.

So the large/small distinction is a capacity-relative phase behaviour, and the "distinct notions"
of a small model are the price of having no spare dimensions to braid them into.

---

## 6. Architecture versus training: the useful reframe

**Training does not create permutations. Architecture chooses which permutations are free.**

That is what a symmetry is — a transformation that holds exactly, at zero cost, without ever
having been learned. The architecture fixes the invariance structure and the orbit; training
merely selects an element within it. Corollary: architectural priors determine which "shades"
exist at all, and in the lazy (NTK) regime a network only interpolates within the span of its
initial features — it cannot invent a shade it was not built to hold.

### Three recipes for shades you do not have to train

1. **Bake the symmetry in (equivariance).** Group-equivariant convolutions return a whole orbit
   of a filter for the price of one (Cohen & Welling 2016). Scale-equivariant steerable networks
   do this for scale specifically (Sosnovik et al., ICLR 2020). The shades are then free by
   construction.

2. **Change coordinates until the permutation becomes a symmetry you already have.** This is the
   most elegant option and the closest thing to the "focus dial" idea. A log-polar transform turns
   uniform scaling into pure translation — which is why the Fourier-Mellin transform reads off both
   scale and rotation as ordinary shifts. The task is therefore not to teach a network about
   resolution; it is to warp the space so that resolution *is* position, and let convolution's
   weight sharing walk along it for free. Scale becomes a direction in the input.

3. **Do not learn the hierarchy at all.** Mallat's scattering transform — cascaded wavelets with
   modulus nonlinearities, fixed, provably stable, never trained — hands you a multi-resolution
   feature hierarchy from pure mathematics (Bruna & Mallat 2013). Build the scale ladder out of
   wavelets and let learning concern itself only with the mixing at the top.

4. **(Bonus) Make it a continuum instead of a ladder.** A hypernetwork emits the weights of
   another network as a function of a scale parameter (Ha et al. 2016). Train at a handful of
   scales and the manifold between them becomes interpolation rather than separate solutions;
   the permutation becomes a coordinate.

### Prior art for a "focus" knob — mostly built, under other names

| Knob | Mechanism | Reference |
|---|---|---|
| Capacity focus | One model that runs at many widths/depths | Slimmable Nets; Once-For-All |
| Representation focus | Nested embeddings, truncatable to any dimension | Matryoshka Representation Learning |
| Compute focus | Extra depth spent only on hard tokens | ACT; PonderNet; Mixture-of-Depths (2404.02258) |
| Diffusion focus | Explicit noise/resolution schedule | Noise-conditioned score networks |
| Numeric precision | Per-**block** shared exponent, per-value small mantissa | OCP microscaling (MX) formats, FP8/FP4 block scaling |

The numeric row is where the remaining territory is. Per-weight precision was never viable —
SIMT hardware punishes heterogeneity, since a warp wants uniform bit-width or it serialises —
but blockwise microscaling is a spatially local, partly trainable precision dial that keeps the
coast rough where it needs to be and flat where it doesn't. This is where hardware roadmaps are
already pointing, and it sits at the intersection the original idea reached for.

---

## 7. The sharpest consequence: sharpness has no stated ruler

Filter-normalised loss landscape plots — smooth valleys for skip-connected networks, jagged
ravines for plain deep ones — are, mathematically, coastline measurements taken with a ruler of
length ε.

Two consequences follow.

1. **A residual connection is a fractal-dimension reduction device.** Li et al. (2018) show skip
   connections demonstrably flatten those plots. Part of why they train is that they make the
   coast measurably less ragged.
2. **Sharpness is scale-dependent and therefore ill-posed absent a stated ε.** It is not invariant
   under reparametrisation (Dinh et al. 2017); rescaling weights moves it. Andriushchenko et al.
   (2023) give the modern treatment. This is the legal-fiction problem exactly: two papers report
   incompatible sharpness values and neither is wrong, because neither named its ruler.

The coastline paradox did not stay in geography. It followed the field into the optimiser.

---

## 8. A cheap test that would settle the most interesting question

**Question.** Is precision a lens (it changes how much coast you see) or a co-author (it changes
what the coast is)? In fractal terms: does the *exponent* hold steady while the measured lengths
change — the fractal outcome — or does the exponent itself shift?

**Method (small, local, single machine).**

1. Train a modest network to convergence; small enough for 16 GB, i.e. well below the local ceiling.
2. Sweep loss at perturbation radius ε and record the roughness curve. This is Richardson's plot
   in loss-space; the machinery already exists in the loss-landscape visualisation literature.
3. Repeat the sweep across a ladder of training/quantisation mantissa widths.
4. Fit the exponent at each rung and compare.

**Prediction, stated in advance so the test can fail.** The exponent should hold roughly constant
for a while, then collapse abruptly below some threshold — locating the cliff where the data
manifold ends and the arithmetic begins. If the exponent drifts smoothly instead, the "lens"
framing is wrong and precision is a full co-author, which is a more interesting result.

**Negative result is publishable-shaped here.** A clean threshold would be an observation about
where measured structure becomes arithmetic artefact, and it would bear directly on how far
quantisation can be pushed before the landscape you are optimising stops being the landscape you
meant to optimise — which is an open question in this repository already (`research/quant-impact.md`,
`docs/references.md` quantization section).

---

## 9. Relevance to patchwork (hypothesis, not finding)

The core insight of this repository — a lightweight router dispatching to specialised modules, with
tiered cascades — is the engineering form of §4: **choose a coarse model deliberately, and escalate
only when the coarse answer stops being close enough.**

In the terms of this document, a cascade stopping rule is a resolution/crossover decision:
measure at a coarse rung, and stop when the next rung stops changing the answer. The failure mode
has a precise analogue too — escalate too little and you are measuring dust while believing you are
measuring Britain; escalate always and you pay maximal resolution for structure that a coarse ruler
already captured.

This is offered as a framing, not as a claim about the current implementation. If it is useful, the
natural place to test it is the tier threshold behaviour already instrumented in
`experiments/router/BENCH-REPORT.md`.

---

## 10. Open questions

1. Does the roughness exponent of the loss landscape correlate with anything we care about
   (generalisation, quantisation robustness, router threshold stability) — or is it only descriptive?
2. Does blockwise precision earn its keep for the router/embedder tiers specifically, where signal
   is cheap and latency dominates?
3. Is there a log-polar-style coordinate change that makes *token-routing depth* a symmetry rather
   than a learned decision? (Speculative. Likely the most interesting direction here.)
4. Do the superposition/allocation phase boundaries predict when a small model should be given a
   dedicated notion (a separate tier) versus asked to braid two into one set of weights?

---

## 11. References

Verified during the writing of this document (arXiv IDs confirmed):

- **The Intrinsic Dimension of Images and Its Impact on Learning** — Pope, Zhu, Abdelkader,
  Goldblum, Goldstein. arXiv:2104.08894 (ICLR 2021).
- **Toy Models of Superposition** — Elhage et al., Transformer Circuits, 2022.
  transformer-circuits.pub/2022/toy_model
- **Mixture-of-Depths: Dynamically allocating compute in transformer-based language models** —
  arXiv:2404.02258.
- **Scale-Equivariant Steerable Networks** — Sosnovik, Szmaja, Smeulders. arXiv:1910.11093
  (ICLR 2020).
- **An exact mapping between the Variational Renormalization Group and Deep Learning** —
  Mehta & Schwab. arXiv:1410.3831.
- **Invariant Scattering Convolution Networks** — Bruna & Mallat. arXiv:1203.1513.
- **A Modern Look at the Relationship between Sharpness and Generalization** — Andriushchenko
  et al. arXiv:2302.07011.

Cited from established knowledge; not re-verified in this session:

- **How Long Is the Coast of Britain? Statistical Self-Similarity and Fractional Dimension** —
  Mandelbrot, *Science*, 1967.
- **The α-dimensional measure of the graph and set of zeros of a Brownian path** — Taylor, 1955
  (graph Hausdorff dimension 3/2).
- **Sharp Minima Can Generalize For Deep Nets** — Dinh, Pascanu, Bengio, Vinyals, 2017.
- **Visualizing the Loss Landscape of Neural Nets** — Li, Xu, Taylor, Studer, Goldstein.
  arXiv:1712.09913 (NeurIPS 2018).
- **Group Equivariant Convolutional Networks** — Cohen & Welling, 2016.
- **Matryoshka Representation Learning** — Kusupati et al., 2022.
- **Hypernetworks** — Ha, Dai, Le, 2016.
- **Scaling Laws for Neural Language Models** — Kaplan et al. arXiv:2001.08361.
- **Training Compute-Optimal Large Language Models** (Chinchilla) — Hoffmann et al., 2022.
- **Adaptive Computation Time for Recurrent Neural Networks** — Graves, 2016; **PonderNet** —
  Banino et al., 2021; **Slimmable Neural Networks** — Yu et al., 2019; **Once-for-All** —
  Cai et al., 2020.
- **OCP Microscaling (MX) Formats Specification** — Open Compute Project, 2023.
- **The Fourier-Mellin transform** / log-polar registration — classical image registration
  literature, background only.
