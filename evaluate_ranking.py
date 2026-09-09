#!/usr/bin/env python3
"""Stage 4: rankings + metrics from proxy score files (EXPERIMENT_SPEC.md §15.5).

Pure function of runs/<run_id>/proxy_scores/<student>/*.jsonl — no GPU, no
network, no model loading. Ground truth comes from the manifest header's
`published_rankings` (ordered lists of tie-groups), never a hard-coded copy.

Implements PROXY_SPEC.md §1:
  §1.1 tie-groups; Kendall tau-b primary, Spearman (mid-ranks) secondary,
       pairwise accuracy excluding GT-tied pairs
  §1.2 top-1 / worst-teacher accuracy; NDCG@1/2 + top-1 regret only when
       published_scores exist (never invented)
  §1.3 seeded task-level bootstrap (default 10000) with CIs, P(top-1),
       ordering distribution; analytic null baselines always printed
  §1.4 student_specificity: the GLM-5 vs Qwen3.5-Plus flip
  §1.5 sample-efficiency ablation over nested seeded subsets (--ablation-grid)

Statistical power caveat (§1.0): m=4 teachers -> exploratory benchmark; no
significance claims below p=1/24.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "artifacts" / "task_manifest.jsonl"

FLIP_PAIR = ("GLM-5", "Qwen3.5-Plus")  # §1.4: the only label difference


# ---------------------------------------------------------------------------
# Ground truth helpers (tie-group representation, §1.1)
# ---------------------------------------------------------------------------

def gt_quality(published: list[list[str]]) -> dict:
    """teacher -> quality value (higher = better; tied teachers equal)."""
    q = {}
    for gi, group in enumerate(published):
        for t in group:
            q[t] = -float(gi)
    return q


def comparable_pairs(published: list[list[str]]) -> list[tuple]:
    """GT-comparable (better, worse) teacher pairs — tied pairs excluded."""
    pairs = []
    for gi, group in enumerate(published):
        for gj in range(gi + 1, len(published)):
            for a in group:
                for b in published[gj]:
                    pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# Rank metrics (manual implementations; no scipy in the runtime env)
# ---------------------------------------------------------------------------

def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    conc = disc = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - ties_x) * (n0 - ties_y))
    if denom == 0:
        return None
    return (conc - disc) / denom


def midranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_midrank(x: list[float], y: list[float]) -> float | None:
    rx, ry = midranks(x), midranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def ranking_metrics(pred: dict, published: list[list[str]]) -> dict:
    """pred: teacher -> aggregate proxy score (higher = better)."""
    teachers = [t for g in published for t in g]
    gq = gt_quality(published)
    xs = [gq[t] for t in teachers]
    ys = [pred[t] for t in teachers]
    # pairwise accuracy over GT-comparable pairs; exact pred tie = 0.5 credit
    correct = 0.0
    pairs = comparable_pairs(published)
    for better, worse in pairs:
        if pred[better] > pred[worse]:
            correct += 1
        elif pred[better] == pred[worse]:
            correct += 0.5
    pred_order = sorted(teachers, key=lambda t: -pred[t])
    if len({pred[t] for t in teachers}) == 1:
        # All-tied prediction carries NO ranking information; the sorted
        # order is arbitrary. Refuse to score it as a ranking.
        return {"degenerate_all_tied": True, "kendall_tau_b": None,
                "spearman_midrank": None, "pairwise_accuracy": None,
                "n_comparable_pairs": len(pairs), "top1_correct": False,
                "worst_correct": False, "predicted_order": pred_order}
    return {
        "kendall_tau_b": kendall_tau_b(xs, ys),
        "spearman_midrank": spearman_midrank(xs, ys),
        "pairwise_accuracy": correct / len(pairs) if pairs else None,
        "n_comparable_pairs": len(pairs),
        "top1_correct": pred_order[0] in published[0],
        "worst_correct": pred_order[-1] in published[-1],
        "predicted_order": pred_order,
    }


def ndcg_and_regret(pred: dict, published_scores) -> dict:
    if not published_scores:
        return {"ndcg@1": "unavailable (no published_scores)",
                "ndcg@2": "unavailable (no published_scores)",
                "top1_regret": "unavailable (no published_scores)"}
    gains = published_scores
    order = sorted(pred, key=lambda t: -pred[t])
    ideal = sorted(gains.values(), reverse=True)
    out = {}
    for k in (1, 2):
        dcg = sum(gains[t] / math.log2(i + 2) for i, t in enumerate(order[:k]))
        idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal[:k]))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else None
    best = max(gains.values())
    out["top1_regret"] = best - gains[order[0]]
    return out


# ---------------------------------------------------------------------------
# Score loading / aggregation
# ---------------------------------------------------------------------------

class PerTask(defaultdict):
    """task_id -> teacher -> value, plus the aggregation rule for the file.

    ``aggregation`` is ``"sum_ratio"`` (values are (num, den); teacher score
    = sum(num)/sum(den)) or ``"grace_teacher_level"`` (values are projected
    gradient vectors; teacher score = -official grace() over the selected
    tasks, duplicates included so the bootstrap stays a task resample)."""

    def __init__(self, aggregation: str = "sum_ratio"):
        super().__init__(dict)
        self.aggregation = aggregation


def _grace_score_fn():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "teacher_proxy_compute_for_grace", SCRIPT_DIR / "compute_proxies.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.grace_teacher_score


def load_score_sets(run_dir: Path, student: str, proxy: str) -> dict:
    """Load every predeclared score view from one physical proxy file.

    Returns ``result_name -> (per_task, meta)``.  Ordinary files have one
    result named after the proxy.  A file containing ``score_views`` produces
    one result per declared view (for example ``cmd_error/fewer_errors``), so
    opposite directions are reported independently rather than selected after
    looking at benchmark agreement.

    Each ``per_task`` maps task_id -> teacher -> (num, den).

    Aggregation is unified as sum(num)/sum(den) over tasks:
      * default (mean of scores): num=score, den=1
      * meta.aggregation == "ratio_of_means" (e.g. RSR, official repo
        semantics): num=ratio_num, den=ratio_den
    """
    slug = student.replace("/", "__")
    path = run_dir / "proxy_scores" / slug / f"{proxy}.jsonl"
    if not path.exists():
        return {}
    rows = []
    meta = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(r)
        meta = r.get("meta") or meta
    view_names = sorted({
        view for row in rows for view in (row.get("score_views") or {})
    })
    if view_names:
        loaded = {}
        for view in view_names:
            per_task = PerTask()
            for row in rows:
                value = (row.get("score_views") or {}).get(view)
                if value is not None:
                    per_task[row["task_id"]][row["teacher"]] = (
                        float(value), 1.0)
            view_meta = dict(meta)
            view_meta["score_view"] = view
            view_meta["physical_proxy"] = proxy
            loaded[f"{proxy}/{view}"] = (per_task, view_meta)
        return loaded

    if meta.get("aggregation") == "grace_teacher_level":
        per_task = PerTask("grace_teacher_level")
        for r in rows:
            if r.get("grad_proj"):
                per_task[r["task_id"]][r["teacher"]] = r["grad_proj"]
        return {proxy: (per_task, meta)}

    per_task = PerTask()
    for r in rows:
        if (r.get("meta") or {}).get("aggregation") == "ratio_of_means":
            if r.get("ratio_num") is not None and r.get("ratio_den"):
                per_task[r["task_id"]][r["teacher"]] = (
                    float(r["ratio_num"]), float(r["ratio_den"]))
        elif r.get("score") is not None:
            per_task[r["task_id"]][r["teacher"]] = (float(r["score"]), 1.0)
    return {proxy: (per_task, meta)}


def load_scores(run_dir: Path, student: str, proxy: str) -> tuple:
    """Backward-compatible single-score loader used by external callers."""
    score_sets = load_score_sets(run_dir, student, proxy)
    if not score_sets:
        return None, {}
    if proxy in score_sets:
        return score_sets[proxy]
    raise ValueError(
        f"{proxy} contains multiple score_views; use load_score_sets()")


_GRACE_FN = []
_GRACE_MEMO = {}  # (id(per_task), teacher, sorted task draw) -> score


def _team_score(per_task, teachers, tasks) -> dict:
    if getattr(per_task, "aggregation", "sum_ratio") == "grace_teacher_level":
        if not _GRACE_FN:
            _GRACE_FN.append(_grace_score_fn())
        out = {}
        # The bootstrap / ablation / paired-difference draws share one seed,
        # so identical task multisets recur across proxies: memoize the
        # (expensive: 10 eigendecompositions) official grace() per draw.
        draw_key = tuple(sorted(tasks))
        for te in teachers:
            key = (id(per_task), te, draw_key)
            if key not in _GRACE_MEMO:
                g = _GRACE_FN[0]([per_task[t][te] for t in tasks])
                _GRACE_MEMO[key] = -g if g is not None else float("nan")
            out[te] = _GRACE_MEMO[key]
        return out
    return {te: (sum(per_task[t][te][0] for t in tasks)
                 / sum(per_task[t][te][1] for t in tasks))
            for te in teachers}


def _task_value(per_task, task, teacher) -> float | None:
    """Per-trajectory score where one exists (None for teacher-level-only
    proxies such as GRACE)."""
    if getattr(per_task, "aggregation", "sum_ratio") != "sum_ratio":
        return None
    num, den = per_task[task][teacher]
    return num / den


def aggregate(per_task: dict, teachers: list[str], task_ids=None) -> tuple | None:
    """sum(num)/sum(den) per teacher over the common-support task set."""
    use = [t for t in (task_ids if task_ids is not None else per_task)
           if t in per_task and all(te in per_task[t] for te in teachers)]
    if not use:
        return None
    return _team_score(per_task, teachers, use), len(use)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def trajectory_diagnostics(per_task: dict, teachers: list[str]) -> dict:
    """Within-teacher dispersion and matched-task winners (PROXY_SPEC §10)."""
    tasks = [task for task in per_task
             if all(teacher in per_task[task] for teacher in teachers)]
    if getattr(per_task, "aggregation", "sum_ratio") != "sum_ratio":
        return {"note": "teacher-level proxy: no per-trajectory scores "
                        "(PROXY_SPEC.md §7.6), §10 diagnostics not defined",
                "n_matched_tasks": len(tasks)}
    by_teacher = {}
    for teacher in teachers:
        values = [_task_value(per_task, task, teacher) for task in tasks]
        q1, q3 = _quantile(values, 0.25), _quantile(values, 0.75)
        by_teacher[teacher] = {
            "n": len(values),
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "variance": statistics.variance(values) if len(values) > 1 else 0.0,
            "q1": q1, "q3": q3,
            "iqr": q3 - q1 if q1 is not None and q3 is not None else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    wins = Counter()
    tie_sizes = Counter()
    for task in tasks:
        values = {teacher: _task_value(per_task, task, teacher)
                  for teacher in teachers}
        best = max(values.values())
        winners = [teacher for teacher, value in values.items() if value == best]
        tie_sizes[len(winners)] += 1
        for teacher in winners:
            wins[teacher] += 1.0 / len(winners)
    return {
        "within_teacher": by_teacher,
        "task_winner_fractional_share": {
            teacher: wins[teacher] / len(tasks) if tasks else None
            for teacher in teachers
        },
        "task_winner_tie_size_counts": dict(sorted(tie_sizes.items())),
        "n_matched_tasks": len(tasks),
    }


# ---------------------------------------------------------------------------
# Bootstrap (§1.3)
# ---------------------------------------------------------------------------

def bootstrap(per_task, teachers, published, n_boot, seed, task_ids=None) -> dict:
    rng = random.Random(seed)
    candidates = task_ids if task_ids is not None else per_task
    tasks = [t for t in candidates
             if t in per_task and all(te in per_task[t] for te in teachers)]
    if not tasks:
        return {"error": "no tasks with complete teacher coverage"}
    taus, pw, top1 = [], [], 0
    orders = Counter()
    rank_counts = {teacher: [0] * len(teachers) for teacher in teachers}
    for _ in range(n_boot):
        draw = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        pred = _team_score(per_task, teachers, draw)
        m = ranking_metrics(pred, published)
        if m["kendall_tau_b"] is not None:
            taus.append(m["kendall_tau_b"])
        if m["pairwise_accuracy"] is not None:
            pw.append(m["pairwise_accuracy"])
        top1 += m["top1_correct"]
        order = tuple(m["predicted_order"])
        orders[order] += 1
        for rank, teacher in enumerate(order):
            rank_counts[teacher][rank] += 1

    def ci(v):
        if not v:
            return None
        s = sorted(v)
        return [s[int(0.025 * len(s))], s[min(int(0.975 * len(s)), len(s) - 1)]]

    return {
        "n_boot": n_boot, "n_tasks_resampled": len(tasks),
        "p_top1_correct": top1 / n_boot,
        "tau_b_ci95": ci(taus), "pairwise_accuracy_ci95": ci(pw),
        "ordering_distribution": [
            {"order": list(o), "p": c / n_boot}
            for o, c in orders.most_common()],
        "rank_probability": {
            teacher: [count / n_boot for count in counts]
            for teacher, counts in rank_counts.items()
        },
    }


# ---------------------------------------------------------------------------
# Sample-efficiency ablation (§1.5) — nested seeded subsets
# ---------------------------------------------------------------------------

def ablation(per_task, teachers, published, grid, seed, n_boot=1000,
             stability_threshold=0.8) -> dict:
    tasks = sorted(t for t in per_task
                   if all(te in per_task[t] for te in teachers))
    rng = random.Random(seed)
    shuffled = tasks[:]
    rng.shuffle(shuffled)  # one shuffle; prefixes give nested subsets
    out = []
    full = _team_score(per_task, teachers, shuffled)
    full_order = ranking_metrics(full, published)["predicted_order"]
    seen = set()
    for n in grid:
        # a grid point above the matched pool means "all matched tasks"
        # (a few tasks can lack a score, e.g. no scorable action); do it once
        n_eff = min(n, len(shuffled))
        if n_eff in seen:
            continue
        seen.add(n_eff)
        subset = shuffled[:n_eff]
        n = n_eff
        agg = _team_score(per_task, teachers, subset)
        m = ranking_metrics(agg, published)
        boot = bootstrap(per_task, teachers, published, n_boot,
                         seed + n, task_ids=subset)
        p_full_order = next((row["p"] for row in
                             boot.get("ordering_distribution", [])
                             if row["order"] == full_order), 0.0)
        out.append({"n": n, "kendall_tau_b": m["kendall_tau_b"],
                    "pairwise_accuracy": m["pairwise_accuracy"],
                    "top1_correct": m["top1_correct"],
                    "predicted_order": m["predicted_order"],
                    "matches_full_order": m["predicted_order"] == full_order,
                    "bootstrap": boot,
                    "p_full_order": p_full_order})
    eligible = [row for row in out if "skipped" not in row]
    stable_n = None
    for i, row in enumerate(eligible):
        if all(later["matches_full_order"] and
               later["p_full_order"] >= stability_threshold
               for later in eligible[i:]):
            stable_n = row["n"]
            break
    return {
        "nested_subset_seed": seed,
        "full_order": full_order,
        "stability_definition": (
            "point order equals full-pool order and bootstrap probability of "
            f"that full order >= {stability_threshold:.2f} at this and all "
            "larger evaluated n"
        ),
        "smallest_stable_n": stable_n,
        "rows": out,
    }


def bootstrap_difference(per_a, per_b, teachers, published, n_boot, seed):
    """Paired task bootstrap of tau-b(A)-tau-b(B)."""
    tasks = [task for task in per_a
             if task in per_b and
             all(teacher in per_a[task] and teacher in per_b[task]
                 for teacher in teachers)]
    if not tasks:
        return {"error": "no common matched tasks"}
    rng = random.Random(seed)
    differences = []
    for _ in range(n_boot):
        draw = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        ma = ranking_metrics(_team_score(per_a, teachers, draw), published)
        mb = ranking_metrics(_team_score(per_b, teachers, draw), published)
        if ma["kendall_tau_b"] is not None and mb["kendall_tau_b"] is not None:
            differences.append(ma["kendall_tau_b"] - mb["kendall_tau_b"])
    ordered = sorted(differences)
    ci = None if not ordered else [
        ordered[int(0.025 * len(ordered))],
        ordered[min(int(0.975 * len(ordered)), len(ordered) - 1)],
    ]
    return {
        "definition": "kendall_tau_b(first)-kendall_tau_b(second)",
        "n_common_tasks": len(tasks), "n_boot": n_boot, "ci95": ci,
        "mean_difference": statistics.fmean(differences) if differences else None,
        "p_difference_gt_0": (sum(x > 0 for x in differences) /
                              len(differences)) if differences else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="terminal_lego")
    p.add_argument("--run-id", required=True)
    p.add_argument("--students", nargs="+", default=["Qwen/Qwen3-8B"])
    p.add_argument("--proxies", nargs="+", default=None)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--ablation-bootstrap", type=int, default=1000,
                   help="bootstrap replicates per nested task-count subset")
    p.add_argument("--stability-threshold", type=float, default=0.8,
                   help="minimum bootstrap P(full order) for stable task count")
    p.add_argument("--expensive-bootstrap", type=int, default=500,
                   help="bootstrap cap for teacher-level aggregators that "
                        "refit per replicate (GRACE: 10 eigendecompositions "
                        "per teacher per replicate)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--ablation-grid", default="10,25,50,100,200,500,1000",
                   help="comma list; PROXY_SPEC.md §1.5 (500 validates the "
                        "200-task default budget)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    header = json.loads(open(args.manifest).readline())
    published_all = header["published_rankings"]
    published_scores = header.get("published_scores")
    teachers = header["teachers"]
    runs_link = SCRIPT_DIR / "runs"
    run_dir = (runs_link / args.run_id).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run dir {run_dir} not found")
    grid = [int(x) for x in args.ablation_grid.split(",") if x.strip()]

    report = {"run_id": args.run_id, "dataset": args.dataset,
              "seed": args.seed,
              "null_baselines": {"expected_tau_b": 0.0, "top1": 0.25,
                                 "pairwise": 0.5},
              "power_caveat": "m=4 teachers; permutation floor p=1/24; "
                              "exploratory (PROXY_SPEC.md §1.0)",
              "students": {}}

    for student in args.students:
        if student not in published_all:
            raise SystemExit(f"no published ranking for student {student!r}")
        published = published_all[student]
        slug_dir = run_dir / "proxy_scores" / student.replace("/", "__")
        proxies = args.proxies or sorted(
            p.stem for p in slug_dir.glob("*.jsonl")) if slug_dir.exists() else []
        if not proxies:
            print(f"[warn] no score files for {student} under {slug_dir}")
            continue
        sres = {}
        loaded_per_task = {}
        for proxy in proxies:
            score_sets = load_score_sets(run_dir, student, proxy)
            if not score_sets:
                print(f"[warn] missing scores: {student}/{proxy}")
                continue
            for result_name, (per_task, meta) in score_sets.items():
                agg = aggregate(per_task, teachers)
                if agg is None:
                    sres[result_name] = {
                        "error": "no tasks with full teacher coverage"}
                    continue
                pred, n_used = agg
                loaded_per_task[result_name] = per_task
                expensive = getattr(per_task, "aggregation", "sum_ratio") != "sum_ratio"
                n_boot = (min(args.bootstrap, args.expensive_bootstrap)
                          if expensive else args.bootstrap)
                m = ranking_metrics(pred, published)
                # published_scores is keyed per student in the registry
                ps = (published_scores or {}).get(student) \
                    if isinstance(published_scores, dict) and student in (
                        published_scores or {}) else None
                m.update(ndcg_and_regret(pred, ps))
                fp = sorted(FLIP_PAIR, key=lambda t: -pred[t])
                sres[result_name] = {
                    "teacher_scores": pred, "n_tasks_used": n_used,
                    "metrics": m,
                    "student_specificity": {
                        "pair": list(FLIP_PAIR),
                        "predicted_order": fp,
                        "student_dependent_proxy": bool(
                            meta.get("student_dependent", True)),
                    },
                    "bootstrap": bootstrap(per_task, teachers, published,
                                           n_boot, args.seed),
                    "sample_efficiency": ablation(
                        per_task, teachers, published, grid, args.seed,
                        min(args.ablation_bootstrap, n_boot),
                        args.stability_threshold),
                    "trajectory_diagnostics": trajectory_diagnostics(
                        per_task, teachers),
                    "meta": meta,
                }
        differences = {}
        for first, second in itertools.combinations(sorted(loaded_per_task), 2):
            pair_expensive = any(
                getattr(loaded_per_task[x], "aggregation", "sum_ratio")
                != "sum_ratio" for x in (first, second))
            differences[f"{first} - {second}"] = bootstrap_difference(
                loaded_per_task[first], loaded_per_task[second], teachers,
                published,
                min(args.bootstrap, args.expensive_bootstrap)
                if pair_expensive else args.bootstrap, args.seed)
        if differences:
            sres["_paired_proxy_differences"] = differences
        report["students"][student] = sres

    # §1.4: does any proxy reproduce the flip across students?
    if len(args.students) >= 2:
        flips = {}
        for proxy in {p for s in report["students"].values() for p in s}:
            per_student = {}
            for student in args.students:
                r = report["students"].get(student, {}).get(proxy)
                if r and "student_specificity" in r:
                    per_student[student] = r["student_specificity"][
                        "predicted_order"]
            if len(per_student) == len(args.students):
                unique = {tuple(v) for v in per_student.values()}
                gt_orders = {s: [t for g in published_all[s] for t in g
                                 if t in FLIP_PAIR] for s in args.students}
                matches = all(per_student[s] == gt_orders[s]
                              for s in args.students)
                flips[proxy] = {
                    "per_student": per_student,
                    "reproduces_flip": ("student-invariant"
                                        if len(unique) == 1 else
                                        ("yes" if matches else "no")),
                }
        report["flip_analysis"] = flips

    out_base = Path(args.out) if args.out else run_dir / "ranking_report"
    json_path = out_base.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    md = [f"# Ranking report — {args.run_id}", "",
          f"Null baselines: tau-b 0.0 · top-1 0.25 · pairwise 0.5 "
          f"(chance). {report['power_caveat']}", ""]
    for student, sres in report["students"].items():
        md.append(f"## {student}\n")
        md.append("| proxy | tau-b | pairwise | top-1 | worst | "
                  "P(top-1) boot | predicted order |")
        md.append("|---|---|---|---|---|---|---|")
        for proxy, r in sorted(sres.items()):
            if proxy.startswith("_"):
                continue
            if "metrics" not in r:
                md.append(f"| {proxy} | — error: {r.get('error')} | | | | | |")
                continue
            m = r["metrics"]
            b = r["bootstrap"]
            def _f(v, spec):
                return format(v, spec) if isinstance(v, (int, float)) else "n/a"
            if m.get("degenerate_all_tied"):
                md.append(f"| {proxy} | — all teacher scores tied "
                          f"(no signal at this n) | | | | | |")
                continue
            md.append(
                f"| {proxy} | {_f(m['kendall_tau_b'], '+.3f')} | "
                f"{_f(m['pairwise_accuracy'], '.3f')} | "
                f"{'✓' if m['top1_correct'] else '✗'} | "
                f"{'✓' if m['worst_correct'] else '✗'} | "
                f"{_f(b.get('p_top1_correct'), '.3f')} | "
                f"{' > '.join(m['predicted_order'])} |")
        md.append("")
    md_path = out_base.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n")
    print(f"[report] {json_path}")
    print(f"[report] {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
