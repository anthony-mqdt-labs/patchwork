"""Live fire for the rung-3 decision tier (darkcore/decisions.py).

Three questions, answered with numbers from this machine:

 1. SAFETY (the gate). `docs/routing-architecture.md` §6 calls rung 3 "the
    danger rung" — cheap plus unreliable buys false security. So the question
    is not "is it fast" but "how many answers that do NOT resolve the query
    would it pass?" A decision judge that passes a bad answer is worse than no
    judge at all, because the cascade would stop climbing on a wrong answer.
 2. CALIBRATION. Raw next-token probability is not a trained decision head.
    This script fits a Platt (temperature+bias) calibration by leave-one-out
    and reports what the head is worth before and after — which is what
    decides whether fine-tuning is required, or whether calibration on a
    handful of labelled pairs is enough.
 3. COST. Forward passes and tokens, against the rung-4 judge it would
    replace (the judge tax measured 24.5% of total spend in bench v0.2).

Run:
    .venv/bin/python decision_bench.py                 # micro tier only
    .venv/bin/python decision_bench.py --judge         # + LLM-judge comparison
    .venv/bin/python decision_bench.py --knn           # + embedder kNN baseline

Artifact: DECISION-BENCH.json (numbers only — the PII rule applies to bench
output too, so query/answer text is never written to it).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS_PATH = os.path.join(HERE, "fixtures", "decision-pairs.json")
BATTERY_PATH = os.path.join(HERE, "battery.jsonl")
OUT_PATH = os.path.join(HERE, "DECISION-BENCH.json")

# The one tier whose weights are resident on this box (2.2 GB, ternary 2-bit).
DEFAULT_TIER = "T1"
DEFAULT_MODEL = "prism-ml/Ternary-Bonsai-8B-mlx-2bit"

CLASS_OPTIONS = ["agentic", "reasoning", "emotional"]


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return None


# ------------------------------------------------------------------ calibration
def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _logloss(rows, a, b):
    tot = 0.0
    for r in rows:
        q = min(max(_sigmoid(a * _logit(r["p_yes"]) + b), 1e-9), 1 - 1e-9)
        tot -= r["label"] * math.log(q) + (1 - r["label"]) * math.log(1 - q)
    return tot / max(len(rows), 1)


def fit_platt(rows):
    """Grid-search (temperature, bias) minimising log loss. No scipy needed."""
    best, best_ll = (1.0, 0.0), float("inf")
    a = 0.05
    while a <= 4.0:
        b = -3.0
        while b <= 3.0:
            ll = _logloss(rows, a, b)
            if ll < best_ll:
                best_ll, best = ll, (round(a, 3), round(b, 3))
            b += 0.05
        a += 0.05
    return {"a": best[0], "b": best[1], "logloss": round(best_ll, 4)}


def platt_loo(rows):
    """Leave-one-out calibrated probabilities — honest, no self-scoring."""
    out = []
    for i, r in enumerate(rows):
        fit = fit_platt([x for j, x in enumerate(rows) if j != i])
        p = _sigmoid(fit["a"] * _logit(r["p_yes"]) + fit["b"])
        out.append({**r, "p_cal": round(p, 4), "fit": fit})
    return out


def auc(rows):
    """Rank-based AUC over p (ties count half)."""
    pos = [r["p"] for r in rows if r["label"] == 1]
    neg = [r["p"] for r in rows if r["label"] == 0]
    if not pos or not neg:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def metrics(rows, key, t):
    neg = [r for r in rows if r["label"] == 0]
    pos = [r for r in rows if r["label"] == 1]
    fp = [r["id"] for r in neg if r[key] >= t]
    fn = [r["id"] for r in pos if r[key] < t]
    return {
        "threshold": round(t, 2),
        "false_pass_rate": round(len(fp) / max(len(neg), 1), 3),
        "false_fail_rate": round(len(fn) / max(len(pos), 1), 3),
        "passed_unsafe_ids": fp,
        "failed_good_ids": fn,
    }


def sweep(rows, key, lo=0.05, hi=0.96, step=0.05):
    out, t = [], lo
    while t <= hi:
        out.append(metrics(rows, key, t))
        t += step
    return out


def safe_threshold(sw):
    """Smallest threshold with zero false passes (the conservative choice)."""
    for row in sw:
        if row["false_pass_rate"] == 0.0:
            return row
    return None


def triage_economics(rows, bypasses=(0.95, 0.9, 0.85, 0.8), fails=(0.05, 0.1, 0.15, 0.2)):
    """What the three-way triage mode buys: pairs where the rung-3 head is
    confident enough to short-circuit the judge, in either direction, plus the
    errors it would make by doing so. Offline — reuses recorded p_yes, so a
    threshold sweep costs nothing."""
    out = []
    for b in bypasses:
        for f in fails:
            skipped = fp = fn = 0
            for r in rows:
                p, lab = r["p_yes"], r["label"]
                if p >= b:
                    skipped += 1
                    fp += int(lab == 0)
                elif p <= f:
                    skipped += 1
                    fn += int(lab == 1)
            out.append({"bypass": b, "fail": f, "judge_calls_skipped": skipped,
                        "skip_share": round(skipped / len(rows), 3),
                        "false_pass_by_rung3": fp, "false_fail_by_rung3": fn,
                        "to_judge": len(rows) - skipped})
    return out


def triage_loo(rows, bypasses=(0.95, 0.9, 0.85, 0.8, 0.75),
               fails=(0.05, 0.1, 0.15, 0.2, 0.25)):
    """Leave-one-out triage: choose the threshold pair on the other n-1 pairs,
    apply it to the held-out one. The in-sample sweep below will always look
    better than this — picking thresholds on the same pairs you score them on
    is selection on the test set, and this is the number to quote."""
    def pick(train):
        best = None
        for b in bypasses:
            for f in fails:
                fp = sum(1 for r in train if r["p_yes"] >= b and r["label"] == 0)
                fn = sum(1 for r in train if r["p_yes"] <= f and r["label"] == 1)
                sk = sum(1 for r in train if r["p_yes"] >= b or r["p_yes"] <= f)
                if fp == 0 and fn == 0 and (best is None or sk > best[2]):
                    best = (b, f, sk)
        return best

    skipped = fp = fn = 0
    picks = []
    for i, r in enumerate(rows):
        train = [x for j, x in enumerate(rows) if j != i]
        b, f, _ = pick(train) or (None, None, 0)
        picks.append({"id": r["id"], "bypass": b, "fail": f})
        if b is None:
            continue
        if r["p_yes"] >= b:
            skipped += 1
            fp += int(r["label"] == 0)
        elif r["p_yes"] <= f:
            skipped += 1
            fn += int(r["label"] == 1)
    return {"judge_calls_skipped": skipped, "skip_share": round(skipped / len(rows), 3),
            "false_pass_by_rung3": fp, "false_fail_by_rung3": fn, "picks": picks}


# ---------------------------------------------------------------------- output
class SelfJudgingPool:
    """Deviation, deliberate and documented: with only one tier resident, the
    rung-4 judge baseline has the answering tier judge itself. The comparison
    is about COST and cache shape (a 256-token generation versus one forward
    pass), not about judge quality — a real deployment would use the next tier
    up, which is not downloaded on this box."""

    def __init__(self, pool, tier):
        self.pool, self.tier = pool, tier

    def next_tier(self, _tier):
        return self.tier

    def generate(self, *a, **k):
        return self.pool.generate(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=DEFAULT_TIER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="micro", choices=["micro", "http"],
                    help="micro = in-process head on a resident tier; "
                         "http = a System One service (Laya/OpenJev/Tiny-Jev) "
                         "at --url, measured on the SAME pairs")
    ap.add_argument("--url", default=None,
                    help="base URL of a running decision service (or set "
                         "DARKCORE_DECISION_URL)")
    ap.add_argument("--decision-model", default=None,
                    help="value for the request's `model` field. Omitted by "
                         "default: a local `laya serve` rejects unknown model "
                         "names, while the hosted relay requires one")
    ap.add_argument("--judge", action="store_true",
                    help="also run the rung-4 LLM judge on the same pairs")
    ap.add_argument("--knn", action="store_true",
                    help="also run the embedder kNN class prior for comparison")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    from darkcore.decisions import MicroScorer, judge_state
    from darkcore.models import ModelPool

    spec = json.load(open(PAIRS_PATH))
    pairs = spec["pairs"]
    statement = spec["statement"]
    battery = [json.loads(l) for l in open(BATTERY_PATH) if l.strip()]

    print(f"tier={args.tier} model={args.model} backend={args.backend}")
    pool = ModelPool([{"id": args.tier, "model": args.model, "max_tokens": 768}])
    if args.backend == "http":
        from darkcore.decisions import resolve_backend
        url = args.url or os.environ.get("DARKCORE_DECISION_URL")
        if not url:
            print("--backend http needs --url or DARKCORE_DECISION_URL")
            return 2
        scorer = resolve_backend("http", pool, args.tier,
                                 {"url": url, "model": args.decision_model})
        load_ms = 0.0
        print(f"url={url} (no local weights loaded; dialects are normalised)")
    else:
        t0 = time.perf_counter()
        model, tok, load_ms = pool.acquire(args.tier)
        scorer = MicroScorer(model, tok, model_id=args.tier)
        print(f"load_ms={load_ms} (wall {round((time.perf_counter()-t0)*1000,1)} ms) "
              f"yes_ids={scorer.yes_ids} no_ids={scorer.no_ids}")
    print()

    # ---- 1. rung-3 safety on labelled pairs --------------------------------
    rows = []
    for p in pairs:
        r = scorer.noul(judge_state(p["query"], p["answer"]), statement)
        rows.append({"id": p["id"], "class": p["class"], "label": p["label"],
                     "p_yes": r["noul"], "p_raw": r["p_yes"],
                     "p_either": r["p_either"], "forward_ms": r["forward_ms"]})
    rows_raw = [{**r, "p": r["p_yes"]} for r in rows]
    rows_cal = platt_loo(rows_raw)

    print(f"=== 1. rung-3 verifier on {len(rows)} labelled pairs ===")
    print(f"{'id':<10}{'lbl':<5}{'p_yes':<9}{'p_either':<10}{'fwd_ms':<8}")
    for r in rows:
        print(f"{r['id']:<10}{r['label']:<5}{r['p_yes']:<9}{r['p_either']:<10}{r['forward_ms']:<8}")

    auc_raw = auc(rows_raw)
    sw_raw = sweep(rows_raw, "p")

    print(f"\nAUC (raw p_yes) = {auc_raw}")
    print(f"raw @0.50 -> {metrics(rows_raw, 'p', 0.5)}")
    safe_raw = safe_threshold(sw_raw)
    print(f"raw, smallest safe threshold -> {safe_raw}")

    # ---- 2. calibration (LOO Platt) ---------------------------------------
    rows_calp = [{**r, "p": r["p_cal"]} for r in rows_cal]
    sw_cal = sweep(rows_calp, "p")
    auc_cal = auc(rows_calp)
    safe_cal = safe_threshold(sw_cal)
    fits = {r["id"]: r["fit"] for r in rows_cal}
    a_vals = [f["a"] for f in fits.values()]

    print(f"\n=== 2. post-hoc calibration (LOO Platt) ===")
    print(f"AUC (calibrated) = {auc_cal}")
    print(f"calibrated @0.50 -> {metrics(rows_calp, 'p', 0.5)}")
    print(f"calibrated, smallest safe threshold -> {safe_cal}")
    print(f"fitted temperature a: min={min(a_vals):.2f} max={max(a_vals):.2f} "
          f"median={sorted(a_vals)[len(a_vals)//2]:.2f}  (a=1 means no rescaling)")

    # ---- 2b. rung-3 triage economics (offline: reuses recorded p_yes) -----
    tri = triage_economics(rows)
    safe_tri = [t for t in tri if t["false_pass_by_rung3"] == 0
                and t["false_fail_by_rung3"] == 0]
    best_tri = max(safe_tri, key=lambda t: t["judge_calls_skipped"]) if safe_tri else None
    print(f"\n=== 2b. rung-3 triage mode (offline) ===")
    print(f"{'bypass':<8}{'fail':<7}{'judge skipped':<15}{'false-pass':<12}"
          f"{'false-fail':<12}{'to judge':<9}")
    for t in tri:
        print(f"{t['bypass']:<8}{t['fail']:<7}{str(t['judge_calls_skipped']) + ' (' + str(int(t['skip_share']*100)) + '%)':<15}"
              f"{t['false_pass_by_rung3']:<12}{t['false_fail_by_rung3']:<12}{t['to_judge']:<9}")
    print(f"best triage with zero measured error: {best_tri}")
    tri_loo = triage_loo(rows)
    print(f"OUT-OF-SAMPLE (LOO) triage: {tri_loo['judge_calls_skipped']}/{len(rows)} "
          f"judge calls skipped ({int(tri_loo['skip_share']*100)}%), "
          f"false-pass {tri_loo['false_pass_by_rung3']}, "
          f"false-fail {tri_loo['false_fail_by_rung3']}")
    print("  (quote the LOO row, not the in-sample sweep above; and note that "
          "0 errors on n pairs only bounds the true rate by the rule of three)")

    # ---- 3. choice over the battery (the predictor job) -------------------
    print(f"\n=== 3. choice over battery.jsonl ({len(battery)} queries) ===")
    modes = ["service"] if args.backend == "http" else ["first_token", "binary"]
    choice_rows = []
    for item in battery:
        q = item["query"]
        row = {"id": item["id"], "truth": item["class"], "passes": 0, "forward_ms": 0.0}
        for m in modes:
            r = scorer.choice(q, "Which class is this query?", CLASS_OPTIONS, mode=m)
            row[f"{m}_choice"] = r["choice"]
            row[f"{m}_conf"] = r["confidence"]
            row["probabilities"] = r["probabilities"]
            row["passes"] += r["passes"]
            row["forward_ms"] = round(row["forward_ms"] + r["forward_ms"], 2)
        choice_rows.append(row)
    for mode in modes:
        hit = sum(1 for r in choice_rows if r.get(f"{mode}_choice") == r["truth"])
        print(f"  {mode:<12} accuracy {hit}/{len(choice_rows)} = {hit/len(choice_rows):.3f}")
    for r in choice_rows:
        got = r[f"{modes[-1]}_choice"]
        flag = "" if got == r["truth"] else "  <-- miss"
        print(f"  {r['id']:<4} truth={r['truth']:<10} {modes[-1]}={got:<10}"
              f" (conf {r[f'{modes[-1]}_conf']}){flag}")

    # ---- 4. kNN baseline (same queries, same protocol) -------------------
    knn = None
    if args.knn:
        try:
            import copy
            from darkcore.predictor import ClassPrior
            from darkcore import surface
            cfg = surface.get_config()
            uri = surface.resolve_exemplar_path(cfg["params"]["exemplar_store_ref"])
            prior = ClassPrior(uri).warm()
            verdicts = [{"id": b["id"], "truth": b["class"],
                         **prior.classify(b["query"])} for b in battery]
            hit = sum(1 for v in verdicts if v["class"] == v["truth"])
            knn = {"n_exemplars": len(prior.ids), "accuracy": hit / len(verdicts),
                   "verdicts": [{k: v[k] for k in ("id", "truth", "class", "confidence")}
                                for v in verdicts]}
            print(f"\n=== 4. embedder kNN baseline (n={len(prior.ids)} exemplars) ===")
            print(f"  accuracy {hit}/{len(verdicts)} = {hit/len(verdicts):.3f}")
        except Exception as e:  # noqa: BLE001 — comparison is optional
            print(f"\n=== 4. kNN baseline unavailable: {type(e).__name__}: {e}")

    # ---- 5. cost versus the rung-4 judge it would replace ----------------
    judge = None
    if args.judge:
        from darkcore import verifiers
        sjp = SelfJudgingPool(pool, args.tier)
        jrows, agree = [], 0
        for p in pairs:
            p_yes = next(r["p_yes"] for r in rows if r["id"] == p["id"])
            d_ok = p_yes >= 0.5
            t = time.perf_counter()
            j_ok, detail = verifiers.next_tier_judge(
                p["query"], p["answer"], sjp, args.tier, {"expected": [], "thresholds": {}})
            jrows.append({"id": p["id"], "label": p["label"], "judge_pass": j_ok,
                          "judge_tokens": detail.get("judge_tokens", 0),
                          "judge_ms": round((time.perf_counter() - t) * 1000, 1)})
            agree += int(j_ok == d_ok)
        tok_total = sum(r["judge_tokens"] for r in jrows)
        ms_total = sum(r["judge_ms"] for r in jrows)
        dec_ms = sum(r["forward_ms"] for r in rows)
        print(f"\n=== 5. rung-4 judge comparison ({len(pairs)} pairs) ===")
        print(f"  judge:   {tok_total} tokens, {ms_total:.0f} ms total "
              f"({ms_total/len(jrows):.0f} ms/pair)")
        print(f"  rung-3:  0 tokens, {dec_ms:.0f} ms total ({dec_ms/len(rows):.1f} ms/pair)")
        print(f"  verdict agreement (judge vs rung-3 @0.5): {agree}/{len(pairs)}")
        print("  NOTE: baseline judge = the same tier judging itself (documented deviation)")
        judge = {"tokens_total": tok_total, "ms_total": round(ms_total, 1),
                 "ms_per_pair": round(ms_total / len(jrows), 1),
                 "decision_ms_total": round(dec_ms, 2),
                 "decision_ms_per_pair": round(dec_ms / len(rows), 2),
                 "agreement": agree / len(pairs), "rows": jrows}

    # ---- artifact ---------------------------------------------------------
    artifact = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),  # repo convention: UTC
        "git_sha": git_sha(), "tier": args.tier, "model": args.model,
        "n_pairs": len(pairs), "n_battery": len(battery),
        "statement": statement,
        "backend": args.backend, "url": args.url,
        "yes_ids": getattr(scorer, "yes_ids", None),
        "no_ids": getattr(scorer, "no_ids", None), "load_ms": load_ms,
        "pairs": rows,
        "raw": {"auc": auc_raw, "at_0.50": metrics(rows_raw, "p", 0.5),
                "smallest_safe": safe_raw, "sweep": sw_raw},
        "calibrated_loo": {"auc": auc_cal, "at_0.50": metrics(rows_calp, "p", 0.5),
                           "smallest_safe": safe_cal, "sweep": sw_cal,
                           "fits": fits},
        "triage": {"all": tri, "best_safe": best_tri, "loo": tri_loo},
        "choice": choice_rows,
        "choice_accuracy": {m: sum(1 for r in choice_rows if r.get(f"{m}_choice") == r["truth"])
                            / len(choice_rows) for m in modes},
        "knn": knn, "judge": judge,
        "cost": {
            "decision_forward_ms_total": round(sum(r["forward_ms"] for r in rows), 2),
            "decision_generated_tokens": 0,
            "choice_forward_ms_total": round(sum(r["forward_ms"] for r in choice_rows), 2),
        },
    }
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
