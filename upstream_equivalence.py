#!/usr/bin/env python3
"""Numerical equivalence of our proxy code against the pinned upstream code.

GPU required (loads the student once). For a few teacher trajectories:
  rsr    : official rsr_cal.infer_dataset + compute_sample_metrics
           vs our _assistant_rank_surprisal            -> exact match expected
  aslec  : official output_drop_score / output_causal_score
           vs our _aslec_components + CASL regression  -> exact match expected
  scas   : official calculate_scas_metrics_on_answer on a SINGLE-TURN chat
           vs our official_final score view             -> match expected
           (agent_all_assistant remains a separate agentic adaptation).
Result of the 2026-08-25 run: artifacts/upstream_equivalence_n8.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compute_proxies as cp  # noqa: E402

UP = cp.UPSTREAM_DIR


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_rsr(tok, model, chats):
    sys.path.insert(0, str(UP / "RankSurprisalRatio"))  # rsr_cal imports rsr_utils
    rsr_cal = _load("rsr_cal", UP / "RankSurprisalRatio" / "rsr_cal.py")
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for i, chat in enumerate(chats):
            f.write(json.dumps({"id": i, "messages": chat}) + "\n")
    inferred = rsr_cal.infer_dataset(model, tok, Path(f.name), batch_size=1,
                                     max_model_len=32768, chat_template="qwen")
    _, per_sample = rsr_cal.compute_sample_metrics(inferred, rank_clip_r=100)
    out = []
    for chat, off in zip(chats, per_sample):
        s_rank, s_nll, n, _ = cp._assistant_rank_surprisal(tok, model, chat, 32768)
        out.append({"official": {"avg_rank": off["avg_rank_clip"],
                                 "avg_surprisal": off["avg_surprisal"],
                                 "n": off["resp_token_length"]},
                    "ours": {"avg_rank": s_rank / n, "avg_surprisal": s_nll / n,
                             "n": n}})
    ok = all(o["official"]["n"] == o["ours"]["n"]
             and abs(o["official"]["avg_rank"] - o["ours"]["avg_rank"]) < 1e-6
             and abs(o["official"]["avg_surprisal"] - o["ours"]["avg_surprisal"]) < 1e-3
             for o in out)
    return {"pass": ok, "samples": out}


def check_aslec(tok, model, chats):
    import numpy as np
    merge = _load("aslec_merge", UP / "ASLEC" / "merge_cal_limo_ours.py")
    steps_all = [cp._assistant_step_logprobs(tok, model, chat, 32768)[0]
                 for chat in chats]
    part_lengths = [[len(s) for s in steps] for steps in steps_all]
    logprobs = [[v for s in steps for v in s] for steps in steps_all]
    results = {}
    for skip in (1, 2):
        official_drop = [merge.output_drop_score(pl, lp, skip_tokens=skip)
                         for pl, lp in zip(part_lengths, logprobs)]
        ours = [cp._aslec_components(steps, skip_tokens=skip) for steps in steps_all]
        m_adj, b1, b2, g, ic = merge.output_causal_score(
            part_lengths, logprobs, skip_tokens=skip)
        x = np.asarray([[c["mean_nonfirst"], c["mean_first"],
                         c["first_token_ratio"], 1.0] for c in ours])
        y = np.asarray([c["mean_logprob"] for c in ours])
        ob1, ob2, og, oic = np.linalg.lstsq(x, y, rcond=None)[0].tolist()
        ours_adj = [c["mean_logprob"] - og * c["first_token_ratio"] for c in ours]
        results[f"skip{skip}"] = {
            "drop_official": official_drop,
            "drop_ours": [c["drop_score"] for c in ours],
            "casl_official": [float(v) for v in m_adj],
            "casl_ours": ours_adj,
            "coef_official": [float(b1), float(b2), float(g), float(ic)],
            "coef_ours": [ob1, ob2, og, oic],
        }
        results[f"skip{skip}"]["pass"] = (
            np.allclose(official_drop, results[f"skip{skip}"]["drop_ours"], atol=1e-9)
            and np.allclose(m_adj, ours_adj, atol=1e-6))
    return {"pass": all(r["pass"] for r in results.values()), **results}


def check_scas(tok, model, chats):
    sys.path.insert(0, str(UP / "Student-Centric-Answer-Selection"))
    from scas.scoring.metric_utils import calculate_scas_metrics_on_answer
    layer = cp._scas_target_layer(model)
    out = []
    for chat in chats:
        # single-turn reduction: first user turn + first assistant turn
        single = [chat[0], next(m for m in chat if m["role"] == "assistant")]
        off = calculate_scas_metrics_on_answer(
            model, tok, [single], layer, lambda_scas=cp.SCAS_LAMBDA)[0]
        ours = cp._trajectory_scas_score_views(
            tok, model, single, 32768, cp.SCAS_LAMBDA)["official_final"]
        keys = ["scas_score", "answer_answer_similarity",
                "answer_question_similarity", "answer_mean_nll",
                "question_mean_nll"]
        out.append({"official": {k: off[k] for k in keys},
                    "ours": {k: ours.get(k) for k in keys},
                    "n_answer_ours": ours.get("n_answer_tokens")})
    rel = [abs(o["official"]["scas_score"] - o["ours"]["scas_score"])
           / max(abs(o["official"]["scas_score"]), 1e-9) for o in out]
    numeric_ok = all(abs(o["official"][k] - o["ours"][k]) < 1e-4
                     for o in out for k in keys)
    return {"pass": numeric_ok, "max_rel_diff_scas_score": max(rel),
            "note": "official_final reproduces the released final-answer "
                    "boundary; agent_all_assistant is a separate adaptation",
            "samples": out}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-file", required=True)
    p.add_argument("--student", default="Qwen/Qwen3-8B")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cp.ensure_env()
    sample = json.loads(Path(args.sample_file).read_text())
    header, _ = cp.load_manifest(cp.DEFAULT_MANIFEST)
    teachers = header["teachers"]
    recs = cp.load_teacher_records(header, teachers)
    chats = []
    for teacher in teachers:
        for tid in sample["task_ids"]:
            r = recs[teacher].get(tid)
            if r is not None:
                chats.append(cp._conversation_as_chat(r["conversations"]))
            if len(chats) >= args.n:
                break
        if len(chats) >= args.n:
            break
    ctx = {"student": args.student, "shared": {}}
    tok, model = cp._student_resources(ctx)
    report = {"student": args.student, "n_trajectories": len(chats),
              "rsr": check_rsr(tok, model, chats),
              "aslec": check_aslec(tok, model, chats),
              "scas": check_scas(tok, model, chats)}
    out = Path(args.out or Path(args.sample_file).parent / "upstream_equivalence.json")
    out.write_text(json.dumps(report, indent=2))
    for k in ("rsr", "aslec", "scas"):
        print(f"[equivalence] {k}: {'PASS' if report[k]['pass'] else 'FAIL'}")
    print(f"[equivalence] report -> {out}")
    return 0 if all(report[k]["pass"] for k in ("rsr", "aslec", "scas")) else 1


if __name__ == "__main__":
    sys.exit(main())
