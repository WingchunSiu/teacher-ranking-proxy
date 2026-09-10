#!/usr/bin/env python3
"""Stage 3: per-teacher, per-task proxy scores (EXPERIMENT_SPEC.md §15.4).

Reads the committed task manifest + the pinned teacher trajectory files and
writes one JSONL per proxy under

    runs/<run_id>/proxy_scores/<student_slug>/<proxy>.jsonl

with rows {"proxy", "teacher", "task_id", "score", <components...>, "meta"}.
Per-task granularity is mandatory (PROXY_SPEC.md §1.3 bootstrap).

Proxy status (PROXY_SPEC.md §10 order):
    teacher_bench   implemented — reads artifacts/teacher_bench_scores.json
                    (sourced numbers only; the file documents provenance)
    traj_length     implemented (CPU; student tokenizer for token counts)
    cmd_error       implemented (verbatim TB2.0 taxonomy + judge prompts,
                    local LLM judge; artifacts/tb2_taxonomy.json)
    tor             implemented (paper formula; align() documented adaptation)
    global_nll      implemented (GPU; GRAPE-style student NLL of teacher
                    assistant tokens; arXiv:2502.04194)
    local_nll_k{1,2,4,8}  implemented (GPU; local-context adaptation of
                    arXiv:2510.03988 — agent-turn adaptation, k sweep)
    aslec_drop/casl implemented (official scoring equations; assistant turns
                    are reasoning-step boundaries)
    rsr             implemented (official repo semantics; ratio-of-means)
    scrf            implemented (§§5-8; TB2.0 labels + §7 recovery judge;
                    agent format errors tracked separately from q_S)
    scas            implemented (official forward-only score, pre-update)
    grace           implemented (official grace() over projected LoRA
                    gradients; teacher-level, evaluate_ranking aggregates)
    car             not exposed (PROXY_SPEC.md §7.2: do not implement)
    lark            STUB — deferred (PROXY_SPEC.md §7.8, decision 2026-08-25)

Scores are cached per (proxy, teacher, task): rerunning skips existing rows
(--resume is implicit; use --force to recompute a proxy from scratch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import random
import re
import shlex
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# Storage root — same contract as generate_trajectories.py (duplicated by
# design; scripts share data through files, EXPERIMENT_SPEC.md §15).
BIG_DISK = Path(os.environ.get("WS_ROOT",
                               "/mnt/hdd_pool_bigsur/userdata/franziska"))
DEFAULT_RUNS_TARGET = BIG_DISK / "teacher_ranking_proxy" / "runs"
RUNS_LINK = SCRIPT_DIR / "runs"
DEFAULT_MANIFEST = SCRIPT_DIR / "artifacts" / "task_manifest.jsonl"
TEACHER_BENCH_FILE = SCRIPT_DIR / "artifacts" / "teacher_bench_scores.json"

LOCAL_NLL_KS = (1, 2, 4, 8)  # PROXY_SPEC.md §7.3 k sweep (agentic adaptation)
ALL_PROXIES = ["teacher_bench", "traj_length", "cmd_error", "error_retry",
               "tor", "egs_post", "egs_loop",
               "global_nll", *[f"local_nll_k{k}" for k in LOCAL_NLL_KS],
               "aslec_drop", "aslec_casl", "rsr", "scas", "grace", "scrf",
               "lark"]
JUDGE_PROXIES = {"cmd_error", "scrf"}
TEACHER_ONLY = {"teacher_bench", "traj_length", "cmd_error", "error_retry",
                "tor", "egs_post", "egs_loop"}

UPSTREAM_COMMITS = {
    "error_retry": {
        "repository": "https://github.com/hanzunye/swe-trajectory-quality-study",
        "commit": "028f15429bb232d3988019818ce77dc74c503331",
    },
    "rsr": {
        "repository": "https://github.com/UmeanNever/RankSurprisalRatio",
        "commit": "59a7c4cdbbb7c26b93f91472da5d79498e29b5c0",
    },
    "aslec": {
        "repository": "https://github.com/wangbing1416/ASLEC",
        "commit": "5737d691deb6cfa78abbdd6773945420fd1210d0",
    },
    "grace": {
        "repository": "https://github.com/abhishekpanigrahi1996/GRACE",
        "commit": "64fc99a10049f79abe05e2be638a178b448985d0",
    },
    "scas": {
        "repository": "https://github.com/ppsmk388/Student-Centric-Answer-Selection",
        "commit": "4cec3a689acebd72ad40580641f8e73bf305b0e4",
    },
}
UPSTREAM_DIR = BIG_DISK / "teacher_ranking_proxy" / "upstream"


def ensure_env() -> None:
    os.environ.setdefault("HF_HOME", str(BIG_DISK / "hf_cache"))
    os.environ.setdefault(
        "HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))


def ensure_runs_root() -> Path:
    DEFAULT_RUNS_TARGET.mkdir(parents=True, exist_ok=True)
    if RUNS_LINK.is_symlink() or RUNS_LINK.exists():
        resolved = RUNS_LINK.resolve()
        if resolved != DEFAULT_RUNS_TARGET.resolve():
            raise SystemExit(f"{RUNS_LINK} resolves to {resolved}, expected "
                             f"{DEFAULT_RUNS_TARGET}; refusing to overwrite.")
    else:
        RUNS_LINK.symlink_to(DEFAULT_RUNS_TARGET)
    return RUNS_LINK.resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def draw_sample(task_ids, n_tasks: int, seed: int) -> list:
    """Pinned sampling algorithm — EXPERIMENT_SPEC.md §3. Do not vary."""
    ids = sorted(set(task_ids))
    if n_tasks > len(ids):
        raise SystemExit(f"--n-tasks {n_tasks} exceeds available tasks ({len(ids)})")
    return random.Random(seed).sample(ids, n_tasks)


def load_manifest(path: Path) -> tuple:
    with open(path) as f:
        header = json.loads(f.readline())
        rows = [json.loads(line) for line in f if line.strip()]
    if header.get("kind") != "task_manifest_header":
        raise SystemExit(f"{path}: first line is not a manifest header; "
                         f"run prepare_dataset.py first")
    return header, rows


def resolve_sample(args, runs_root: Path) -> tuple:
    """Same contract as generate_trajectories.resolve_sample (duplicated by
    design). Returns (run_id, run_dir, task_ids)."""
    if args.sample_file and args.n_tasks is not None:
        raise SystemExit("Pass either --sample-file or --n-tasks/--seed, not both.")
    if args.sample_file:
        sample_path = Path(args.sample_file).resolve()
        sample = json.loads(sample_path.read_text())
        run_id = args.run_id or sample_path.parent.name
        return run_id, sample_path.parent, sample["task_ids"]
    if args.n_tasks is None:
        raise SystemExit("Provide --n-tasks/--seed or --sample-file.")
    header, rows = load_manifest(Path(args.manifest))
    if header["dataset"] != args.dataset:
        raise SystemExit(f"manifest is for dataset {header['dataset']!r}, "
                         f"requested {args.dataset!r}")
    manifest_sha = sha256_file(Path(args.manifest))
    run_id = args.run_id or f"{args.dataset}-n{args.n_tasks}-s{args.seed}"
    run_dir = runs_root / run_id
    sample_path = run_dir / "sampled_task_ids.json"
    if sample_path.exists():
        existing = json.loads(sample_path.read_text())
        for key, want in [("n_tasks", args.n_tasks), ("seed", args.seed),
                          ("manifest_sha256", manifest_sha),
                          ("dataset", args.dataset)]:
            if existing.get(key) != want:
                raise SystemExit(
                    f"{sample_path} exists but {key}={existing.get(key)!r} "
                    f"disagrees with requested {want!r}; pass an explicit "
                    f"--run-id for a fresh directory.")
        return run_id, run_dir, existing["task_ids"]
    sampled = draw_sample([r["task_id"] for r in rows], args.n_tasks, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps({
        "task_ids": sampled, "n_tasks": args.n_tasks, "seed": args.seed,
        "manifest_sha256": manifest_sha, "dataset": args.dataset}, indent=2) + "\n")
    return run_id, run_dir, sampled


# ---------------------------------------------------------------------------
# Teacher trajectory access (pinned revision from the manifest header)
# ---------------------------------------------------------------------------

def load_teacher_records(header: dict, teachers: list[str]) -> dict:
    """{teacher: {task_id: record}} from the per-teacher JSON files."""
    from huggingface_hub import hf_hub_download
    # task_id_field is per-teacher in the manifest header, e.g.
    # {"GLM-5": "metadata.oracle_passed_task", ...}
    id_field_map = header["task_id_field"]
    if isinstance(id_field_map, str):
        id_field_map = {t: id_field_map for t in teachers}
    out = {}
    for teacher in teachers:
        id_field = id_field_map[teacher].split(".")
        fname = header["teacher_files"][teacher]
        local = hf_hub_download(header["traj_repo"], fname, repo_type="dataset",
                                revision=header["traj_revision"])
        recs = json.loads(Path(local).read_text())
        by_task = {}
        for rec in recs:
            node = rec
            for k in id_field:
                node = node[k]
            by_task[node] = rec
        out[teacher] = by_task
        print(f"[traj] {teacher}: {len(by_task)} records "
              f"({fname}@{header['traj_revision'][:8]})")
    return out


def parse_gpt_turn(value: str) -> dict | None:
    """Terminus-2 assistant turns are JSON: {analysis, plan, commands:[{keystrokes,
    duration}...], task_complete}. Returns None when unparseable."""
    try:
        d = json.loads(value)
        return d if isinstance(d, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Score file I/O (cache per (proxy, teacher, task))
# ---------------------------------------------------------------------------

def score_path(run_dir: Path, student: str, proxy: str) -> Path:
    slug = student.replace("/", "__")
    p = run_dir / "proxy_scores" / slug / f"{proxy}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def existing_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["teacher"], r["task_id"]))
    return keys


def append_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Proxy: teacher_bench (teacher-only; PROXY_SPEC.md §2)
# ---------------------------------------------------------------------------

def compute_teacher_bench(ctx) -> list[dict]:
    """Published standalone benchmark performance per teacher. Values must be
    SOURCED — this reads artifacts/teacher_bench_scores.json:
        {"benchmark": "...", "source": "<url/citation>",
         "scores": {"<teacher>": <float>, ...}}
    and refuses to run without it (never invent numbers)."""
    if not TEACHER_BENCH_FILE.exists():
        raise SystemExit(
            f"{TEACHER_BENCH_FILE} missing. Create it with published teacher "
            "benchmark numbers and their source (PROXY_SPEC.md §2 'teacher "
            "benchmark performance'); scores are never invented.")
    spec = json.loads(TEACHER_BENCH_FILE.read_text())
    scores = spec["scores"]
    missing = [t for t in ctx["teachers"] if t not in scores]
    if missing:
        raise SystemExit(f"{TEACHER_BENCH_FILE}: missing scores for {missing}")
    meta = {"benchmark": spec.get("benchmark"), "source": spec.get("source"),
            "direction": "higher_better", "student_dependent": False}
    rows = []
    for teacher in ctx["teachers"]:
        for task_id in ctx["task_ids"]:
            # Constant per teacher by construction; still emitted per-task so
            # the bootstrap machinery treats every proxy identically.
            rows.append({"proxy": "teacher_bench", "teacher": teacher,
                         "task_id": task_id, "score": float(scores[teacher]),
                         "meta": meta})
    return rows


# ---------------------------------------------------------------------------
# Proxy: traj_length (teacher-only; PROXY_SPEC.md §2)
# ---------------------------------------------------------------------------

def compute_traj_length(ctx) -> list[dict]:
    """assistant turns / commands / assistant tokens / total tokens per
    (teacher, task). Token counts use the *student* tokenizer so they reflect
    what the student would be trained on. Both predeclared directions for
    assistant turns and assistant-generated tokens are stored as score views;
    evaluate_ranking.py reports all four without post-hoc direction choice."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ctx["student"], trust_remote_code=True)
    meta = {"tokenizer": ctx["student"], "direction": "multiple_predeclared",
            "student_dependent": False,
            "score_views": ["assistant_turns_more", "assistant_turns_less",
                            "assistant_tokens_more", "assistant_tokens_less"]}
    rows = []
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            if (teacher, task_id) in ctx["done"]:
                continue
            rec = recs.get(task_id)
            if rec is None:
                # allow_intersection manifests can lack a (teacher, task) pair
                rows.append({"proxy": "traj_length", "teacher": teacher,
                             "task_id": task_id, "score": None,
                             "missing": True, "meta": meta})
                continue
            conv = rec["conversations"]
            gpt_turns = [c["value"] for c in conv if c["from"] == "gpt"]
            n_cmds = 0
            unparseable = 0
            for v in gpt_turns:
                d = parse_gpt_turn(v)
                if d is None:
                    unparseable += 1
                else:
                    n_cmds += len(d.get("commands") or [])
            a_tok = sum(len(tok.encode(v, add_special_tokens=False))
                        for v in gpt_turns)
            t_tok = sum(len(tok.encode(c["value"], add_special_tokens=False))
                        for c in conv)
            rows.append({
                "proxy": "traj_length", "teacher": teacher, "task_id": task_id,
                "score": float(-a_tok),
                "score_views": {
                    "assistant_turns_more": float(len(gpt_turns)),
                    "assistant_turns_less": float(-len(gpt_turns)),
                    "assistant_tokens_more": float(a_tok),
                    "assistant_tokens_less": float(-a_tok),
                },
                "assistant_turns": len(gpt_turns), "commands": n_cmds,
                "assistant_tokens": a_tok, "total_tokens": t_tok,
                "unparseable_gpt_turns": unparseable, "meta": meta})
    return rows


# ---------------------------------------------------------------------------
# Proxies: global_nll / local_nll (student-dependent; PROXY_SPEC.md §3)
# ---------------------------------------------------------------------------

def _conversation_as_chat(conv: list[dict]) -> list[dict]:
    role = {"human": "user", "gpt": "assistant"}
    return [{"role": role[c["from"]], "content": c["value"]} for c in conv]


def _load_student(student: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("global_nll/local_nll need a GPU (run inside a Slurm "
                         "allocation); CUDA is not available here.")
    tok = AutoTokenizer.from_pretrained(student, trust_remote_code=True)
    # device_map "cuda" (default, preserves the 8B single-GPU numerics); set
    # TRP_DEVICE_MAP=auto to shard a large student (e.g. 32B) across GPUs so the
    # fp32 vocab log_softmax in _assistant_nll has headroom.
    model = AutoModelForCausalLM.from_pretrained(
        student, torch_dtype=torch.bfloat16,
        device_map=os.environ.get("TRP_DEVICE_MAP", "cuda"),
        trust_remote_code=True)
    model.eval()
    return tok, model


def _student_resources(ctx):
    """Load the student once when several likelihood proxies share a run."""
    shared = ctx.setdefault("shared", {})
    key = ("student_resources", ctx["student"])
    if key not in shared:
        shared[key] = _load_student(ctx["student"])
    return shared[key]


def _scan_response_spans(input_ids: list[int], header_ids: list[int],
                         end_ids: list[int]) -> list[tuple[int, int]]:
    """Assistant-content spans, matching official RSR ``rsr_utils.py``.

    Unlike incremental length subtraction, this excludes chat control/end
    tokens. The benchmark students are Qwen3 models and use the official
    repository's Qwen marker setting.
    """
    spans = []
    n, h, e = len(input_ids), len(header_ids), len(end_ids)
    if not h:
        return spans
    i = 0
    while i <= n - h:
        if input_ids[i:i + h] != header_ids:
            i += 1
            continue
        start = i + h
        j = start
        while j <= n - e and input_ids[j:j + e] != end_ids:
            j += 1
        end = j if j <= n - e else n
        if end > start:
            spans.append((start, end))
        i = j + e if j <= n - e else n
    return spans


def _render_ids_and_assistant_spans(tok, chat: list[dict], max_len: int):
    encoded = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False,
        truncation=True, max_length=max_len,
    )
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    header = tok("<|im_start|>assistant\n", add_special_tokens=False)[
        "input_ids"]
    end = tok("<|im_end|>", add_special_tokens=False)["input_ids"]
    spans = _scan_response_spans(encoded, header, end)
    return encoded, spans


def _assistant_nll(tok, model, chat: list[dict], max_len: int,
                   last_assistant_only: bool = False) -> tuple:
    """Sum NLL over teacher-generated assistant content tokens only."""
    import torch
    untruncated = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False)
    ids, spans = _render_ids_and_assistant_spans(tok, chat, max_len)
    truncated = len(untruncated) > max_len
    if last_assistant_only and spans:
        spans = spans[-1:]
    if len(ids) < 2 or not spans:
        return 0.0, 0, truncated
    input_ids = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    total_nll = 0.0
    n_tok = 0
    for s, e in spans:
        for pos in range(max(s, 1), e):
            total_nll += -logprobs[pos - 1, ids[pos]].item()
            n_tok += 1
    return total_nll, n_tok, truncated


def _assistant_step_logprobs(tok, model, chat: list[dict], max_len: int):
    """Gold-token log probabilities grouped by assistant/action turn."""
    import torch
    untruncated = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False)
    ids, spans = _render_ids_and_assistant_spans(tok, chat, max_len)
    truncated = len(untruncated) > max_len
    if len(ids) < 2 or not spans:
        return [], truncated
    input_ids = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    steps = []
    for start, end in spans:
        positions = list(range(max(start, 1), end))
        values = []
        # Selected rows only: avoid a sequence_length x vocabulary float32
        # log-softmax, which is unnecessarily large on long trajectories.
        for offset in range(0, len(positions), 256):
            chunk = positions[offset:offset + 256]
            rows = logits[[pos - 1 for pos in chunk]].float()
            gold = torch.tensor([ids[pos] for pos in chunk], device=rows.device)
            lp = torch.log_softmax(rows, dim=-1).gather(
                1, gold[:, None]).squeeze(1)
            values.extend(float(value) for value in lp.cpu().tolist())
        if values:
            steps.append(values)
    return steps, truncated


ASLEC_SKIP_TOKENS = 1  # PROXY_SPEC.md §7.4: "first token of every step".
# NOTE: the official repo's driver (merge_cal_limo_ours.py) defaults to
# --skip_token 2 and its published selections use *_skip2 files; the spec's
# definition (1) is used here and the official default is recorded in meta.
ASLEC_OFFICIAL_SKIP_TOKENS = 2


def _aslec_components(steps: list[list[float]],
                      skip_tokens: int = ASLEC_SKIP_TOKENS) -> dict:
    """Official ASLEC sufficient statistics (output_drop_score /
    output_causal_score in merge_cal_limo_ours.py) with ``skip_tokens`` head
    tokens per step."""
    flat = [value for step in steps for value in step]
    heads = [value for step in steps for value in step[:skip_tokens]]
    non_heads = [value for step in steps for value in step[skip_tokens:]]
    if not flat:
        return {"mean_logprob": None, "mean_first": -16.0,
                "mean_nonfirst": -16.0, "first_token_ratio": 0.0,
                "drop_score": None, "n_tokens": 0, "n_steps": 0}
    return {
        "mean_logprob": sum(flat) / len(flat),
        # -16 is the official implementation's empty-region fallback.
        "mean_first": sum(heads) / len(heads) if heads else -16.0,
        "mean_nonfirst": (sum(non_heads) / len(non_heads)
                          if non_heads else -16.0),
        "first_token_ratio": len(heads) / len(flat),
        "drop_score": (sum(non_heads) / len(non_heads)
                       if non_heads else 0.0),
        "n_tokens": len(flat), "n_steps": len(heads),
    }


def _trajectory_aslec_components(ctx, teacher: str, task_id: str, rec: dict):
    shared = ctx.setdefault("shared", {})
    cache = shared.setdefault("aslec_components", {})
    key = (teacher, task_id)
    if key not in cache:
        tok, model = _student_resources(ctx)
        chat = _conversation_as_chat(rec["conversations"])
        steps, truncated = _assistant_step_logprobs(tok, model, chat, 32768)
        cache[key] = _aslec_components(steps) | {"truncated": truncated}
    return cache[key]


def compute_global_nll(ctx) -> list[dict]:
    """GRAPE-style (arXiv:2502.04194): mean NLL of the fixed teacher assistant
    tokens under the pre-SFT student, conditioned on the full preceding
    conversation. Loss over assistant tokens only (instructions, system text
    and terminal observations excluded by construction). Score = mean assistant
    log-likelihood (= -NLL/token), higher = better."""
    tok, model = _student_resources(ctx)
    max_len = 32768
    meta = {"method": "GRAPE-style global student NLL",
            "reference": "arXiv:2502.04194", "max_len": max_len,
            "direction": "higher_better", "student_dependent": True,
            "adaptation": "multi-turn agent trajectory rendered with the "
                          "student chat template; NLL over assistant turns"}
    rows = []
    t0 = time.time()
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            if (teacher, task_id) in ctx["done"]:
                continue
            rec = recs.get(task_id)
            if rec is None:
                rows.append({"proxy": "global_nll", "teacher": teacher,
                             "task_id": task_id, "score": None,
                             "missing": True, "meta": meta})
                continue
            chat = _conversation_as_chat(rec["conversations"])
            nll, n, truncated = _assistant_nll(tok, model, chat, max_len)
            rows.append({
                "proxy": "global_nll", "teacher": teacher, "task_id": task_id,
                "score": (-nll / n) if n else None,
                "sum_nll": nll, "assistant_tokens_scored": n,
                "truncated": truncated, "meta": meta})
        print(f"[global_nll] {teacher} done ({time.time()-t0:.0f}s)")
    return rows


def compute_local_nll(ctx, k: int) -> list[dict]:
    """Agent-turn adaptation of local naturalness (arXiv:2510.03988): for each
    assistant turn, NLL conditioned only on the task context plus the previous
    k action-observation pairs. The target turn alone is scored, then turn
    means are averaged equally (published LALP semantics). One proxy per k in
    LOCAL_NLL_KS (PROXY_SPEC.md §7.3); global_nll is the full-history
    baseline."""
    tok, model = _student_resources(ctx)
    name = f"local_nll_k{k}"
    max_len = 32768
    meta = {"method": "local-turn student NLL (agent-turn adaptation)",
            "reference": "arXiv:2510.03988", "k_action_observation_pairs": k,
            "max_len": max_len, "direction": "higher_better",
            "student_dependent": True,
            "aggregation": "equal_mean_of_assistant_turn_mean_logprobs",
            "adaptation": "assistant/action turn is one reasoning step"}
    rows = []
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            if (teacher, task_id) in ctx["done"]:
                continue
            rec = recs.get(task_id)
            if rec is None:
                rows.append({"proxy": name, "teacher": teacher,
                             "task_id": task_id, "score": None,
                             "missing": True, "meta": meta})
                continue
            chat = _conversation_as_chat(rec["conversations"])
            first_user = chat[0:1] if chat and chat[0]["role"] == "user" else []
            turn_logprobs = []
            tot_tok = 0
            for i, msg in enumerate(chat):
                if msg["role"] != "assistant":
                    continue
                previous_assistants = [j for j in range(i)
                                       if chat[j]["role"] == "assistant"]
                # previous k action-observation pairs (all of them if fewer)
                start = (previous_assistants[max(0, len(previous_assistants) - k)]
                         if k and previous_assistants else i)
                ctx_msgs = chat[start:i]
                local_chat = first_user + ctx_msgs + [msg]
                nll, n, _ = _assistant_nll(
                    tok, model, local_chat, max_len, last_assistant_only=True)
                if n:
                    turn_logprobs.append(-nll / n)
                tot_tok += n
            rows.append({
                "proxy": name, "teacher": teacher, "task_id": task_id,
                "score": (sum(turn_logprobs) / len(turn_logprobs)
                          if turn_logprobs else None),
                "assistant_turns_scored": len(turn_logprobs),
                "assistant_tokens_scored": tot_tok,
                "meta": meta})
        print(f"[{name}] {teacher} done")
    return rows


def compute_aslec(ctx, variant: str) -> list[dict]:
    """ASLEC-DROP or ASLEC-CASL using assistant turns as step boundaries."""
    import numpy as np

    upstream = UPSTREAM_COMMITS["aslec"]
    entries = []
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            rec = recs.get(task_id)
            components = (_trajectory_aslec_components(
                ctx, teacher, task_id, rec) if rec is not None else None)
            entries.append((teacher, task_id, components))
        print(f"[{variant}] token statistics ready for {teacher}")

    valid = [components for _, _, components in entries
             if components and components["mean_logprob"] is not None]
    coefficients = None
    adjusted = {}
    if variant == "aslec_casl" and valid:
        # Equivalent to sklearn LinearRegression used by the official repo:
        # M ~ beta1*M_non + beta2*M_first + gamma*F + intercept.
        x = np.asarray([[c["mean_nonfirst"], c["mean_first"],
                         c["first_token_ratio"], 1.0] for c in valid],
                       dtype=np.float64)
        y = np.asarray([c["mean_logprob"] for c in valid], dtype=np.float64)
        beta1, beta2, gamma, intercept = np.linalg.lstsq(
            x, y, rcond=None)[0].tolist()
        coefficients = {"beta1_nonfirst": beta1, "beta2_first": beta2,
                        "gamma_first_token_ratio": gamma,
                        "intercept": intercept,
                        "fit_n_trajectories": len(valid),
                        "fit_scope": "all teachers on fixed matched task pool"}
        for teacher, task_id, components in entries:
            if components and components["mean_logprob"] is not None:
                adjusted[(teacher, task_id)] = (
                    components["mean_logprob"] -
                    gamma * components["first_token_ratio"])

    meta = {
        "method": "ASLEC-DROP" if variant == "aslec_drop" else "ASLEC-CASL",
        "reference": "arXiv:2604.06834",
        "official_repository": upstream["repository"],
        "official_commit": upstream["commit"],
        "skip_tokens": ASLEC_SKIP_TOKENS,
        "official_driver_default_skip_tokens": ASLEC_OFFICIAL_SKIP_TOKENS,
        "max_len": 32768,
        "direction": "higher_better", "student_dependent": True,
        "adaptation": "Terminus assistant/action turns are reasoning steps",
    }
    if coefficients:
        meta["regression"] = coefficients

    rows = []
    for teacher, task_id, components in entries:
        if (teacher, task_id) in ctx["done"]:
            continue
        if components is None:
            rows.append({"proxy": variant, "teacher": teacher,
                         "task_id": task_id, "score": None,
                         "missing": True, "meta": meta})
            continue
        score = (components["drop_score"] if variant == "aslec_drop" else
                 adjusted.get((teacher, task_id)))
        rows.append({"proxy": variant, "teacher": teacher,
                     "task_id": task_id, "score": score,
                     "components": components, "meta": meta})
    return rows


# ---------------------------------------------------------------------------
# Proxy: scas (student-dependent; PROXY_SPEC.md §7.4)
# ---------------------------------------------------------------------------

SCAS_LAMBDA = 0.5  # official default (--lambda-scas 0.5)


def _scas_target_layer(model) -> str:
    """Official ``infer_target_layer``: last decoder block's mlp.up_proj."""
    n_layers = model.config.num_hidden_layers
    return f"model.layers.{n_layers - 1}.mlp.up_proj"


def _scas_upstream():
    """Official helpers from the pinned Student-Centric-Answer-Selection repo."""
    path = UPSTREAM_DIR / "Student-Centric-Answer-Selection"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from scas.scoring import metric_utils
    return metric_utils


def _trajectory_scas_components(tok, model, chat: list[dict], max_len: int,
                                lambda_scas: float) -> dict:
    """Forward-only SCAS statistics for one trajectory, built from the
    official ``metric_utils`` functions (token NLL from the model's logits,
    special-token mask, normalized target-layer activations, block/score
    combination) with ONE documented adaptation for multi-turn agent
    trajectories: the answer set A is every teacher assistant span (the
    tokens SFT trains on) instead of only the final message, and Q is every
    other non-special token (task text + terminal observations).
    AA = |mu_A|^2 equals the official mean pairwise cosine incl. diagonal;
    AQ = mu_A . mu_Q; S = (1-lambda) d_A^2 AA + lambda d_A d_Q AQ."""
    import torch
    import torch.nn.functional as F
    mu = _scas_upstream()
    untruncated = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False)
    ids, spans = _render_ids_and_assistant_spans(tok, chat, max_len)
    truncated = len(untruncated) > max_len
    if len(ids) < 2 or not spans:
        return {"score": None, "n_answer_tokens": 0, "truncated": truncated}
    store = {}
    layer_name = _scas_target_layer(model)
    handles = mu.register_act_hooks(model, layer_name, store)
    input_ids = torch.tensor([ids], device="cuda")
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
    finally:
        mu.remove_hooks(handles)
    hidden = store[layer_name]
    if isinstance(hidden, tuple):
        hidden = hidden[0]
    hidden = hidden.squeeze(0).float()
    normalized = F.normalize(hidden, p=2, dim=1)
    n_full = len(ids)
    ids_1d = input_ids.squeeze(0)
    is_special = mu.build_special_token_mask(tok, ids_1d)
    answer_mask = torch.zeros(n_full, dtype=torch.bool, device="cuda")
    for s, e in spans:
        answer_mask[s:e] = True
    answer_mask &= ~is_special
    question_mask = (~answer_mask) & (~is_special)
    n_a = int(answer_mask.sum().item())
    n_q = int(question_mask.sum().item())
    if n_a == 0:
        return {"score": None, "n_answer_tokens": 0, "truncated": truncated}
    mu_a = normalized[answer_mask].mean(dim=0)
    aa = float((mu_a @ mu_a).item())
    aa_no_diag = ((aa * n_a * n_a - n_a) / (n_a * n_a - n_a)) if n_a > 1 else 0.0
    aq = float((mu_a @ normalized[question_mask].mean(dim=0)).item()) if n_q else 0.0
    # official token NLL (log_softmax in the logits' own dtype) + masks
    nll_per_pos = mu.nll_per_token_from_logits(outputs.logits, input_ids)
    d_q = float(mu.avg_nll_by_pos_mask(nll_per_pos, question_mask).item())
    d_a = float(mu.avg_nll_by_pos_mask(nll_per_pos, answer_mask).item())
    parts = mu.compute_scas_scores(
        answer_answer_similarity=aa, answer_answer_similarity_no_diag=aa_no_diag,
        answer_question_similarity=aq, question_mean_nll=d_q,
        answer_mean_nll=d_a, lambda_scas=lambda_scas)
    del outputs, hidden, normalized
    return {**parts, "answer_answer_similarity": aa,
            "answer_answer_similarity_no_diag": aa_no_diag,
            "answer_question_similarity": aq,
            "n_answer_tokens": n_a, "n_question_tokens": n_q,
            "truncated": truncated}


def compute_scas(ctx) -> list[dict]:
    """SCAS learning cost (arXiv:2605.26872), official forward-only proxy,
    computed ONCE on the pre-SFT student (static adaptation of a method that
    re-scores every training round — PROXY_SPEC.md §7.7). Score = -S so that
    higher = cheaper to learn = preferred, matching the paper's min-score
    selection. Aggregation to teacher level: arithmetic mean over matched
    tasks (the paper selects per prompt and defines no teacher aggregate)."""
    tok, model = _student_resources(ctx)
    max_len = 32768
    upstream = UPSTREAM_COMMITS["scas"]
    meta = {"method": "SCAS forward-only learning cost (pre-update)",
            "reference": "arXiv:2605.26872",
            "official_code": upstream["repository"],
            "official_commit": upstream["commit"],
            "target_layer": _scas_target_layer(model),
            "lambda_scas": SCAS_LAMBDA, "max_len": max_len,
            "direction": "higher_better", "student_dependent": True,
            "score_definition": "-scas_score (lower cost preferred)",
            "adaptation": "A = all teacher assistant tokens of the multi-turn "
                          "trajectory, Q = all other non-special tokens "
                          "(task text + terminal observations); scored once "
                          "on the pre-SFT student instead of per training "
                          "round; mean over matched tasks per teacher"}
    rows = []
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            if (teacher, task_id) in ctx["done"]:
                continue
            rec = recs.get(task_id)
            if rec is None:
                rows.append({"proxy": "scas", "teacher": teacher,
                             "task_id": task_id, "score": None,
                             "missing": True, "meta": meta})
                continue
            chat = _conversation_as_chat(rec["conversations"])
            comp = _trajectory_scas_components(tok, model, chat, max_len,
                                               SCAS_LAMBDA)
            score = comp.get("scas_score")
            rows.append({"proxy": "scas", "teacher": teacher,
                         "task_id": task_id,
                         "score": (-score) if score is not None else None,
                         "components": comp, "meta": meta})
        print(f"[scas] {teacher} done")
    return rows


# ---------------------------------------------------------------------------
# Proxy: grace (student-dependent, teacher-level; PROXY_SPEC.md §7.6)
# ---------------------------------------------------------------------------

GRACE_PROJ_DIM = 512        # official scripts/grace.sh PROJ_DIM
GRACE_N_SPLITS = 10         # official GRACE_computation.py defaults
GRACE_TEST_FRACTION = 0.1
GRACE_SMOOTH = 1e-3
GRACE_LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
              "target_modules": ["q_proj", "v_proj", "k_proj", "up_proj",
                                 "down_proj", "gate_proj"]}  # official --use-lora


def official_grace():
    """Import the pinned official ``grace()`` (GRACE/GRACE_computation.py)."""
    import importlib.util
    path = UPSTREAM_DIR / "GRACE" / "GRACE" / "GRACE_computation.py"
    if not path.exists():
        raise SystemExit(f"official GRACE code missing at {path}; clone "
                         f"{UPSTREAM_COMMITS['grace']['repository']} at "
                         f"{UPSTREAM_COMMITS['grace']['commit']}")
    spec = importlib.util.spec_from_file_location("grace_official", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.grace


def grace_teacher_score(vectors, n_splits: int = GRACE_N_SPLITS,
                        test_fraction: float = GRACE_TEST_FRACTION,
                        smooth_coeff: float = GRACE_SMOOTH) -> float | None:
    """Official grace() over one teacher's projected trajectory gradients
    (one trajectory per task -> n_gen_per_prompt=1). Lower = better."""
    import numpy as np
    g = np.asarray(vectors, dtype=np.float64)
    if g.ndim != 2 or len(g) < 2:
        return None
    return float(official_grace()(
        g, dim=g.shape[1], n_gen_per_prompt=1, test_fraction=test_fraction,
        n_splits=n_splits, smooth_coeff=smooth_coeff))


def _grace_student(ctx):
    """LoRA-wrapped student (official --use-lora configuration, seed 0) with
    gradient checkpointing; only lora_B gradients are collected."""
    import torch
    from peft import LoraConfig, get_peft_model
    shared = ctx.setdefault("shared", {})
    key = ("grace_student", ctx["student"])
    if key in shared:
        return shared[key]
    tok, base = _student_resources(ctx)
    torch.manual_seed(0)
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **GRACE_LORA))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # HF only checkpoints activations in train() mode; keep every dropout
    # module in eval() so the pass is deterministic and dropout-free (the
    # official script runs the eval-mode model, i.e. no dropout either).
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    params = [(n, p) for n, p in model.named_parameters()
              if p.requires_grad and "lora_B" in n]
    grad_dim = sum(p.numel() for _, p in params)
    shared[key] = (tok, model, params, grad_dim)
    return shared[key]


def _grace_projector(ctx, grad_dim: int):
    """Official projector: trak.projectors.CudaProjector with the exact
    arguments of GRACE/gradient_computation.py (Rademacher, seed 0, bf16,
    block_size 4). Returns None if the fast_jl CUDA extension is missing, in
    which case the torch fallback below is used (recorded in meta)."""
    shared = ctx.setdefault("shared", {})
    key = ("grace_projector", grad_dim)
    if key not in shared:
        try:
            import torch
            from trak.projectors import CudaProjector, ProjectionType
            shared[key] = CudaProjector(
                proj_dim=GRACE_PROJ_DIM, grad_dim=grad_dim, seed=0,
                proj_type=ProjectionType.rademacher, device="cuda",
                # official passes bfloat16; this fast_jl build accepts only
                # fp16/fp32 -> fp32 (no precision loss vs the official call)
                dtype=torch.float32, block_size=4, max_batch_size=8)
        except Exception as e:  # ImportError or fast_jl build issue
            print(f"[grace] trak CudaProjector unavailable ({e!r}); "
                  "using torch Rademacher fallback")
            shared[key] = None
    return shared[key]


def _rademacher_project(flat, proj_dim: int, seed: int = 0, block: int = 1 << 20):
    """Fallback JL projection P g / sqrt(grad_dim), seeded Rademacher blocks
    generated on the GPU (same family/seed/normalization as the official
    CudaProjector call, different random draw)."""
    import torch
    gen = torch.Generator(device=flat.device)
    out = torch.zeros(proj_dim, dtype=torch.float32, device=flat.device)
    n = flat.numel()
    for i, start in enumerate(range(0, n, block)):
        gen.manual_seed(seed * 1_000_003 + i)
        chunk = flat[start:start + block].float()
        r = torch.randint(0, 2, (chunk.numel(), proj_dim), generator=gen,
                          device=flat.device, dtype=torch.int8).float() * 2 - 1
        out += chunk @ r
        del r
    return out / (n ** 0.5)


def _trajectory_projected_gradient(ctx, chat: list[dict], max_len: int):
    """Student NLL over teacher assistant tokens (labels = -100 elsewhere, as
    official tokenize_data.py) -> backward -> lora_B gradient -> projection."""
    import torch
    tok, model, params, grad_dim = _grace_student(ctx)
    untruncated = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False)
    ids, spans = _render_ids_and_assistant_spans(tok, chat, max_len)
    truncated = len(untruncated) > max_len
    if len(ids) < 2 or not spans:
        return None, 0, truncated, None
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    labels = [-100] * len(ids)
    for s, e in spans:
        for pos in range(max(s, 1), e):
            labels[pos] = ids[pos]
    n_sup = sum(1 for x in labels if x != -100)
    input_ids = torch.tensor([ids], device="cuda")
    labels_t = torch.tensor(labels[1:], device="cuda")  # shifted targets
    model.zero_grad(set_to_none=True)
    # Same loss as model(input_ids, labels=...) (mean CE over supervised,
    # shifted targets) but WITHOUT materialising the full [T, vocab] float
    # logits: decoder once (gradient checkpointed), then lm_head + CE in
    # checkpointed chunks. Needed for 32k-token trajectories on one H100.
    causal_lm = model.get_base_model()
    hidden = causal_lm.model(input_ids=input_ids, use_cache=False)[0][0, :-1]
    lm_head = causal_lm.lm_head

    def _chunk_ce(h, y):
        return F.cross_entropy(lm_head(h).float(), y, ignore_index=-100,
                               reduction="sum")

    total = torch.zeros((), device="cuda", dtype=torch.float32)
    step = 2048
    for off in range(0, hidden.shape[0], step):
        h, y = hidden[off:off + step], labels_t[off:off + step]
        if (y != -100).any():
            total = total + checkpoint(_chunk_ce, h, y, use_reentrant=False)
    loss_t = total / max(n_sup, 1)
    loss_t.backward()
    loss = float(loss_t.item())
    flat = torch.cat([p.grad.reshape(-1) for _, p in params])
    norm = float(flat.float().norm().item())
    projector = _grace_projector(ctx, grad_dim)
    if projector is not None:
        proj = (projector.project(flat.unsqueeze(0).float().contiguous(),
                                  model_id=0) / (grad_dim ** 0.5))[0].float()
    else:
        proj = _rademacher_project(flat, GRACE_PROJ_DIM)
    model.zero_grad(set_to_none=True)
    del hidden, total, loss_t, flat
    torch.cuda.empty_cache()
    return proj.cpu().tolist(), n_sup, truncated, {"loss": loss, "grad_norm": norm}


def compute_grace(ctx) -> list[dict]:
    """GRACE (arXiv:2511.02833) with the official code path: per-trajectory
    projected student gradients (LoRA-B, Rademacher JL to 512 dims) stored per
    (teacher, task) row; the teacher-level score is the official grace()
    over that teacher's 200 vectors (n_gen_per_prompt=1, 10 seeded splits,
    test fraction 0.1, smoothing 1e-3). GRACE has NO per-trajectory score by
    construction (PROXY_SPEC.md §7.6: no invented per-trajectory
    approximation), so rows carry score=None plus the gradient vector and
    evaluate_ranking.py aggregates with grace_teacher_score (negated, since
    lower GRACE = better)."""
    max_len = 32768
    upstream = UPSTREAM_COMMITS["grace"]
    _, _, _, grad_dim = _grace_student(ctx)
    official_projector = _grace_projector(ctx, grad_dim) is not None
    meta = {"method": "GRACE (official grace() over projected LoRA gradients)",
            "projector": ("trak.CudaProjector (official)" if official_projector
                          else "torch Rademacher fallback"),
            "reference": "arXiv:2511.02833",
            "official_code": upstream["repository"],
            "official_commit": upstream["commit"],
            "proj_dim": GRACE_PROJ_DIM, "grad_dim": grad_dim,
            "lora": GRACE_LORA, "n_splits": GRACE_N_SPLITS,
            "test_fraction": GRACE_TEST_FRACTION, "smooth_coeff": GRACE_SMOOTH,
            "n_gen_per_prompt": 1, "max_len": max_len,
            "direction": "higher_better", "student_dependent": True,
            "aggregation": "grace_teacher_level",
            "score_definition": "teacher score = -grace(vectors); no "
                                "per-trajectory score",
            "adaptation": "official --use-lora gradient option; loss computed "
                          "with chunked lm_head+CE (identical value) to fit "
                          "32k-token trajectories; labels on teacher "
                          "assistant tokens only; one trajectory per task"}
    rows = []
    t0 = time.time()
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            if (teacher, task_id) in ctx["done"]:
                continue
            rec = recs.get(task_id)
            if rec is None:
                rows.append({"proxy": "grace", "teacher": teacher,
                             "task_id": task_id, "score": None,
                             "missing": True, "meta": meta})
                continue
            chat = _conversation_as_chat(rec["conversations"])
            try:
                vec, n_sup, truncated, stats = _trajectory_projected_gradient(
                    ctx, chat, max_len)
                used_len = max_len
            except __import__("torch").cuda.OutOfMemoryError:
                __import__("torch").cuda.empty_cache()
                vec, n_sup, truncated, stats = _trajectory_projected_gradient(
                    ctx, chat, max_len // 2)
                used_len = max_len // 2
            rows.append({"proxy": "grace", "teacher": teacher,
                         "task_id": task_id, "score": None,
                         "grad_proj": vec, "supervised_tokens": n_sup,
                         "truncated": truncated, "max_len_used": used_len,
                         "components": stats, "meta": meta})
        print(f"[grace] {teacher} done ({time.time()-t0:.0f}s)")
    # Teacher-level summary for the log (evaluate_ranking recomputes it).
    by_teacher = {}
    for r in rows:
        if r.get("grad_proj"):
            by_teacher.setdefault(r["teacher"], []).append(r["grad_proj"])
    for teacher, vecs in by_teacher.items():
        print(f"[grace] {teacher}: n={len(vecs)} grace="
              f"{grace_teacher_score(vecs)}")
    return rows


# ---------------------------------------------------------------------------
# Proxy: tor (teacher-only; Terminal-Lego arXiv:2606.03461)
# ---------------------------------------------------------------------------

# Observation command set listed in the paper ("environment inspection and
# verification commands"). ``pwd`` is additionally needed to track cwd and is
# recorded as an adaptation in score metadata.
TOR_OBSERVATION_CMDS = {
    "cat", "ls", "find", "grep", "head", "wc", "diff", "stat", "pwd",
}
# "state-changing actions such as editing files, installing packages, or
# running scripts" — the paper's upstream code is unreleased, so this action
# classification is OUR documented operationalization (meta.adaptation).
TOR_ACTION_CMDS = {
    "sed", "tee", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "ln", "chmod",
    "chown", "tar", "unzip", "zip", "gzip", "gunzip", "pip", "pip3", "apt",
    "apt-get", "apk", "dnf", "yum", "npm", "yarn", "cargo", "make", "cmake",
    "gcc", "g++", "python", "python3", "node", "bash", "sh", "ruby", "perl",
    "java", "javac", "go", "git", "patch", "install", "dd", "truncate",
}

_PATH_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9_+-]{1,12}$")
_REDIRECT_RE = re.compile(r"(?:^|\s)(?:\d*>>?|<)\s*([^\s;&|]+)")
_TEST_CMDS = {"pytest", "tox", "unittest", "make", "ctest"}


def _shell_tokens(cmdline: str) -> list[str]:
    try:
        return shlex.split(cmdline, comments=False, posix=True)
    except ValueError:
        return re.findall(r"[^\s'\";|&<>()]+", cmdline)


def _normalize_path(path: str, cwd: str) -> str | None:
    path = path.strip("'\" ,:;()[]{}")
    if not path or path.startswith("-") or path in {"/dev/null", "-"}:
        return None
    if any(ch in path for ch in "$`()"):
        return None  # unresolved shell expression: do not pretend alignment
    path = path.rstrip("/") or "/"
    if path.startswith("~/") or path == "~":
        return posixpath.normpath("/home/user" + path[1:])
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(cwd, path))


def _extract_paths(cmdline: str, cwd: str = "/") -> set[str]:
    """Extract and normalize plausible filesystem targets from a shell line.

    This deliberately rejects shell expressions rather than creating false
    path alignments. Redirection targets are retained even when they lack a
    suffix (for example ``> output``).
    """
    tokens = _shell_tokens(cmdline)
    word = _first_cmd_word(cmdline)
    path_operand_cmds = {
        "cat", "ls", "head", "wc", "diff", "stat", "find", "cd", "cp",
        "mv", "rm", "mkdir", "rmdir", "touch", "ln", "chmod", "chown",
        "tee", "tar", "unzip", "zip", "gzip", "gunzip", "patch",
        "install", "truncate", "bash", "sh", "python", "python3", "node",
    }
    positional = [
        (i, tok) for i, tok in enumerate(tokens[1:], start=1)
        if not tok.startswith("-") and tok not in {
            "|", "||", "&&", ";", ">", ">>", "<"}
    ]
    if word == "sed" and positional:
        # sed's program commonly contains slashes but is not a file path.
        positional = positional[-1:]
    out = set()
    for i, tok in enumerate(tokens):
        if i == 0 or tok.startswith("-") or tok in {
                "|", "||", "&&", ";", ">", ">>", "<"}:
            continue
        looks_path = (
            "/" in tok or tok in {".", ".."} or
            bool(_PATH_SUFFIX_RE.search(tok)) or
            (word in path_operand_cmds and (i, tok) in positional) or
            (i > 0 and tokens[i - 1] in {">", ">>", "<"})
        )
        if looks_path:
            normalized = _normalize_path(tok, cwd)
            if normalized:
                out.add(normalized)
    for match in _REDIRECT_RE.finditer(cmdline):
        normalized = _normalize_path(match.group(1), cwd)
        if normalized:
            out.add(normalized)
    return out


def _first_cmd_word(line: str) -> str:
    toks = _shell_tokens(line)
    for t in toks:
        if "=" in t and not t.startswith("/"):
            continue  # skip VAR=val prefixes
        if t in ("sudo", "env", "timeout", "nohup", "xargs"):
            continue
        return t.rsplit("/", 1)[-1]
    return ""


def _initial_cwd(rec: dict) -> str:
    """Best-effort cwd from the Terminus task prompt; default to root."""
    for msg in rec.get("conversations", []):
        if msg.get("from") != "human":
            continue
        text = msg.get("value", "")
        matches = re.findall(
            r"(?:Working Directory|working directory)\s*(?:\n|:)?\s*`?(/[^\s`]+)",
            text,
        )
        if matches:
            return posixpath.normpath(matches[-1].rstrip(".,"))
    return "/"


def _trajectory_events(rec: dict) -> list[dict]:
    """Ordered shell events of a Terminus trajectory, one per executed
    command, from the same per-command segments used by cmd_error
    (_command_segments): command text and its own output as the terminal
    showed them. The cwd comes from the echoed prompt when it carries one
    (Docker: root@id:/cwd#); otherwise from the task prompt + `cd` tracking."""
    conv = rec.get("conversations", [])
    cwd = _initial_cwd(rec)
    events = []
    assistant_turn = -1
    for i, msg in enumerate(conv):
        if msg.get("from") != "gpt":
            continue
        assistant_turn += 1
        parsed = parse_gpt_turn(msg.get("value", ""))
        if not parsed:
            continue
        if i + 1 >= len(conv) or conv[i + 1].get("from") != "human":
            continue
        for seg in _command_segments(_turn_commands(parsed),
                                     conv[i + 1].get("value", "")):
            if not seg["observed"]:
                continue  # command not on screen (omitted); no output to ground
            if seg["cwd"]:
                cwd = seg["cwd"]
            line = seg["input"]
            word = _first_cmd_word(line)
            paths = _extract_paths(line, cwd)
            kind = ("observation" if word in TOR_OBSERVATION_CMDS else
                    "action" if word in TOR_ACTION_CMDS or ">" in line else
                    "other")
            events.append({
                "turn": assistant_turn, "line": line, "word": word,
                "paths": paths, "cwd": cwd, "kind": kind,
                "output": seg["output"],
            })
            if word == "cd":
                toks = _shell_tokens(line)
                if len(toks) >= 2:
                    changed = _normalize_path(toks[1], cwd)
                    if changed:
                        cwd = changed
    return events


def _paths_aligned(obs_paths: set, act_paths: set,
                   strict: bool = False, exact: bool = False) -> bool:
    """align(o,a): 'matches, contains, or is directly related to'.

    loose (strict=False, the original operationalization): exact match,
    directory containment in either direction, or basename equality.

    strict (strict=True): only the paper's three examples — the observation
    targets the same path as the action ("inspecting src/utils.py before
    editing src/utils.py", "reading a script before executing it") or the
    observation targets a directory that contains the action's path
    ("listing src/ before creating a file inside it"). No basename match, no
    containment in the other direction (observing a file inside the directory
    the action targets), so listing the project root only supports actions
    whose target lies under it and does not match by name alone.
    """
    for a in act_paths:
        ab = a.rsplit("/", 1)[-1]
        for o in obs_paths:
            if a == o:
                return True
            if exact:
                continue  # exact: same path only (tightest reading)
            if a.startswith(o + "/"):
                return True
            if strict:
                continue
            if o.startswith(a + "/"):
                return True
            if ab and ab == o.rsplit("/", 1)[-1]:
                return True
    return False


def _error_output(text: str) -> bool:
    """EGS failure detection reuses the B2 error keyword set (one copy)."""
    return _obs_has_error(text)


# TOR variants (arXiv:2606.03461 gives the observation list and three
# alignment examples but no action list; upstream code unreleased). The
# paper's Table 3 TOR is 2.5-13.4% per teacher; the original variant here
# (list_loose) gives 35-56%, so the other three are predeclared to find out
# which choice drives the gap. Action set: "list" = TOR_ACTION_CMDS or a
# redirect (original); "all" = every command that is not an observation and
# not a bare `cd` (pure navigation, no state change). Alignment: see
# _paths_aligned (loose = original, strict = the paper's examples only).
# Window: "any" = the observation is anywhere earlier in the command stream,
# including earlier in the same Terminus-2 turn (the original; but a turn's
# commands are typed as one batch, so the agent has NOT seen that output when
# it decides the action); "prevturn" = the observation is in an earlier
# assistant turn, i.e. its output was in the agent's context.
TOR_VIEWS = tuple(f"{a}_{al}_{w}" for a in ("list", "all")
                  for al in ("loose", "strict", "exact")
                  for w in ("any", "prevturn"))


def _is_action(event: dict, action_set: str) -> bool:
    if action_set == "list":
        return event["kind"] == "action"
    return event["kind"] != "observation" and event["word"] != "cd"


def _trajectory_grounding_components(rec: dict, horizon: int = 3) -> dict:
    events = _trajectory_events(rec)
    actions = [i for i, event in enumerate(events) if event["kind"] == "action"]
    pre_supported = post_verified = loop_supported = 0
    tor_n = {v: 0 for v in TOR_VIEWS}
    tor_sup = {v: 0 for v in TOR_VIEWS}
    observations = []
    for i, event in enumerate(events):
        if event["kind"] == "observation":
            observations.append(event)
            continue
        for view in TOR_VIEWS:
            action_set, align, window = view.split("_")
            if not _is_action(event, action_set):
                continue
            tor_n[view] += 1
            tor_sup[view] += int(bool(event["paths"]) and any(
                _paths_aligned(obs["paths"], event["paths"],
                               strict=(align == "strict"),
                               exact=(align == "exact"))
                for obs in observations
                if obs["paths"] and (window == "any" or
                                     obs["turn"] < event["turn"])))
        if event["kind"] != "action":
            continue
        # inspect -> act: an earlier observation on an aligned path (TOR)
        pre = bool(event["paths"]) and any(
            _paths_aligned(obs["paths"], event["paths"])
            for obs in observations if obs["paths"])
        # act -> verify: within the next `horizon` assistant turns, an
        # observation on an aligned path or a test/build command
        later = [candidate for candidate in events[i + 1:]
                 if candidate["turn"] <= event["turn"] + horizon]
        post = any(
            candidate["kind"] == "observation" and
            candidate["paths"] and event["paths"] and
            _paths_aligned(candidate["paths"], event["paths"])
            for candidate in later
        ) or any(candidate["word"] in _TEST_CMDS for candidate in later)
        pre_supported += int(pre)
        post_verified += int(post)
        loop_supported += int(pre and post)
    n_actions = len(actions)
    return {
        "n_actions": n_actions,
        "n_pre_supported": pre_supported,
        "n_post_verified": post_verified,
        "n_loop_supported": loop_supported,
        "tor_views": {v: (tor_sup[v] / tor_n[v] if tor_n[v] else None)
                      for v in TOR_VIEWS},
        "tor_counts": {v: {"n_actions": tor_n[v], "n_supported": tor_sup[v]}
                       for v in TOR_VIEWS},
        "n_commands": len(events),
        "n_observations": len(observations),
        "n_turns": (events[-1]["turn"] + 1) if events else 0,
        "tor": pre_supported / n_actions if n_actions else None,
        "egs_post": post_verified / n_actions if n_actions else None,
        "egs_loop": loop_supported / n_actions if n_actions else None,
    }


def compute_tor(ctx) -> list[dict]:
    """Targeted Observation Ratio (arXiv:2606.03461):
        TOR = |{a in A : exists o in O, o before a and align(o,a)}| / |A|
    computed over the teacher trajectory's command stream (Terminus-2
    commands[].keystrokes, split into lines). Higher is better. Upstream code
    is unreleased ("available upon acceptance"); the action-command set and
    align() rules here are documented adaptations, revisit when it ships."""
    meta = {"method": "Targeted Observation Ratio",
            "reference": "arXiv:2606.03461", "direction": "higher_better",
            "student_dependent": False,
            "observation_cmds": sorted(TOR_OBSERVATION_CMDS),
            "score_view_definitions": {
                "views": "<actions>_<align>_<window>",
                "actions": {"list": "TOR_ACTION_CMDS or redirect (original)",
                            "all": "every non-observation command except cd"},
                "align": {"loose": "exact/containment either way/basename "
                                   "(original)",
                          "strict": "same path, or observed directory "
                                    "contains the action path (the paper's "
                                    "three examples only)",
                          "exact": "same path only"},
                "window": {"any": "observation anywhere earlier in the "
                                  "command stream, same turn allowed "
                                  "(original)",
                           "prevturn": "observation in an earlier assistant "
                                       "turn, so its output was in context"},
                "original": "list_loose_any"},
            "adaptation": "cwd-aware normalized paths; pwd added; action set "
                          "and align() predeclared in four views because the "
                          "paper lists no action commands and upstream code "
                          "is unreleased; paper Table 3 TOR is 2.5-13.4%"}
    rows = []
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            if (teacher, task_id) in ctx["done"]:
                continue
            rec = recs.get(task_id)
            if rec is None:
                rows.append({"proxy": "tor", "teacher": teacher,
                             "task_id": task_id, "score": None,
                             "missing": True, "meta": meta})
                continue
            components = _trajectory_grounding_components(rec)
            n_actions = components["n_actions"]
            n_supported = components["n_pre_supported"]
            rows.append({
                "proxy": "tor", "teacher": teacher, "task_id": task_id,
                "score": (n_supported / n_actions) if n_actions else None,
                "score_views": components["tor_views"],
                "counts": components["tor_counts"],
                "n_commands": components["n_commands"],
                "n_observations": components["n_observations"],
                "n_turns": components["n_turns"],
                "n_actions": n_actions, "n_supported": n_supported,
                "meta": meta})
    return rows


EGS_DEFINITIONS = {
    "tor": "inspect -> act: fraction of actions preceded by an observation on "
           "an aligned path (Terminal-Lego TOR)",
    "egs_post": "act -> verify: fraction of actions followed within 3 "
                "assistant turns by an observation on an aligned path or a "
                "test/build command",
    "egs_loop": "inspect -> act -> verify: fraction of actions satisfying "
                "both conditions",
}


def compute_egs(ctx, component: str) -> list[dict]:
    """Extended-EGS components (PROXY_SPEC.md §6.6), reported separately —
    intentionally no weighted composite."""
    meta = {
        "method": "Extended Environment-Grounded Supervision",
        "reference": "Terminal-Lego, arXiv:2606.03461",
        "component": component, "definition": EGS_DEFINITIONS[component],
        "temporal_window_assistant_turns": 3,
        "direction": "higher_better", "student_dependent": False,
        "adaptation": "deterministic cwd/path-aligned rules on per-command "
                      "screen segments; components reported separately",
    }
    rows = []
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            if (teacher, task_id) in ctx["done"]:
                continue
            rec = recs.get(task_id)
            if rec is None:
                rows.append({"proxy": component, "teacher": teacher,
                             "task_id": task_id, "score": None,
                             "missing": True, "meta": meta})
                continue
            components = _trajectory_grounding_components(rec)
            rows.append({
                "proxy": component, "teacher": teacher, "task_id": task_id,
                "score": components[component], "components": components,
                "meta": meta,
            })
    return rows


# --- B2 Error-Retry, copied verbatim from hanzunye/swe-trajectory-quality-study
#     @028f15429bb232d3988019818ce77dc74c503331
#     scripts/scoring/scoring_config.py (B2_ERROR_KEYWORDS) and
#     scripts/scoring/analysis.py (_obs_has_error, _actions_similar,
#     _compute_b2_error_retry). Only the docstrings were shortened. (Upstream's
#     B2_MAX_CYCLES=10 normalization is dropped: we score on raw cycle counts.)
B2_ERROR_KEYWORDS = {
    "traceback", "error", "exception", "failed", "failure",
    "syntaxerror", "typeerror", "valueerror", "assertionerror",
    "nameerror", "attributeerror", "importerror", "keyerror",
    "runtimeerror", "oserror", "errno", "stderr",
    "command not found", "no such file", "permission denied",
}


def _obs_has_error(obs_text: str) -> bool:
    lower = obs_text.lower()
    return any(kw in lower for kw in B2_ERROR_KEYWORDS)


def _actions_similar(sig_a: tuple[str, str], sig_b: tuple[str, str]) -> bool:
    return sig_a[0] == sig_b[0]


def _compute_b2_error_retry(action_sigs: list[tuple[str, str]],
                            obs_texts: list[str]) -> tuple[int, int]:
    """(cycle_count, total_action_pairs): action -> error observation ->
    similar action counts as one cycle."""
    n = len(action_sigs)
    if n < 2:
        return 0, 0
    cycles = 0
    for i in range(n - 1):
        obs = obs_texts[i]
        if obs and _obs_has_error(obs) and _actions_similar(action_sigs[i], action_sigs[i + 1]):
            cycles += 1
    return cycles, n - 1


def _error_retry_cross_turn(events: list[dict]) -> tuple[int, int]:
    """Turn-aware Error-Retry (our primary; Terminus adaptation of B2).

    Verbatim B2 assumes one tool call per agent step, so it walks per-command
    pairs. A Terminus turn (one agent JSON) batches several commands committed
    BEFORE the agent sees any output, so an error-then-similar-command inside
    one turn is not a retry (no feedback yet). We work per turn and key on the
    command that actually errored: a cycle is a tool (first command word) whose
    own command errored in turn t AND errored again with the same tool in the
    next turn t+1 (the first turn after the agent saw the error). Requiring the
    retry to fail again -- persistent same-tool failure -- is a deliberate
    deviation from B2 (which counts any same-tool reuse after an error).
    Returns (cycles, turn_pairs). `events` are in trajectory order with a
    `turn` index, `word` (first command word), and `output`."""
    from itertools import groupby
    failed_by_turn = [
        {e["word"] for e in evs if _obs_has_error(e["output"])}
        for _, evs in ((t, list(g)) for t, g in
                       groupby(events, key=lambda e: e["turn"]))
    ]
    if len(failed_by_turn) < 2:
        return 0, 0
    cycles = sum(1 for a, b in zip(failed_by_turn, failed_by_turn[1:]) if a & b)
    return cycles, len(failed_by_turn) - 1


def compute_error_retry(ctx) -> list[dict]:
    """Error-Retry (B2, arXiv:2607.17205). Fewer retry cycles = better teacher.

    Score is the RAW cycle count (negated so higher = better); no arbitrary
    normalization. Two Terminus adaptations, both documented in PROXY_SPEC 6.4:
      * tool name -> first word of the command line (_first_cmd_word), our
        stand-in for B2's named-tool signature (Terminus has only a shell).
      * PRIMARY score is turn-aware (_error_retry_cross_turn): a cycle is a tool
        whose command errored in turn t and errored again in turn t+1 (the first
        turn after the agent saw the error). This fixes the fact that a Terminus
        turn batches commands with no feedback between them.
    The verbatim per-command B2 (flattens turns, counts intra-turn pairs, any
    same-tool reuse after an error) is kept ONLY as the `verbatim_old` view for
    comparison -- it is not the score."""
    upstream = UPSTREAM_COMMITS["error_retry"]
    meta = {
        "method": "B2 Error-Retry (turn-aware persistent same-tool failure)",
        "reference": "arXiv:2607.17205",
        "official_repository": upstream["repository"],
        "official_commit": upstream["commit"],
        "direction": "higher_better",  # higher = fewer cycles = better
        "student_dependent": False,
        "score_def": "-cross_turn_cycles (raw count; a tool that errored in turn "
                     "t errors again with the same tool in turn t+1)",
        "adaptation": "tool = first word of command (_first_cmd_word); primary "
                      "score is per-turn cross-boundary (verbatim per-command B2 "
                      "kept only as the verbatim_old view)",
        "score_views": ["cross_turn_persist", "cross_turn_persist_rate",
                        "verbatim_old"],
    }
    rows = []
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            if (teacher, task_id) in ctx["done"]:
                continue
            rec = recs.get(task_id)
            if rec is None:
                rows.append({"proxy": "error_retry", "teacher": teacher,
                             "task_id": task_id, "score": None,
                             "missing": True, "meta": meta})
                continue
            # Per-command events (each with its own output slice) in order.
            events = [e for e in _trajectory_events(rec) if e["output"]]
            # Primary: turn-aware persistent same-tool failure (raw cycle count).
            ct_cycles, ct_pairs = _error_retry_cross_turn(events)
            # verbatim_old: literal per-command B2 (raw cycle count).
            sigs = [(e["word"], e["line"]) for e in events]
            vb_cycles, vb_pairs = _compute_b2_error_retry(
                sigs, [e["output"] for e in events])
            rows.append({
                "proxy": "error_retry", "teacher": teacher, "task_id": task_id,
                "score": float(-ct_cycles),
                "score_views": {
                    "cross_turn_persist": float(-ct_cycles),
                    "cross_turn_persist_rate":
                        (-ct_cycles / ct_pairs) if ct_pairs else 0.0,
                    "verbatim_old": float(-vb_cycles),
                },
                "cross_turn_cycles": ct_cycles, "turn_pairs": ct_pairs,
                "verbatim_cycles": vb_cycles, "action_pairs": vb_pairs,
                "meta": meta,
            })
    return rows


# ---------------------------------------------------------------------------
# Proxy: rsr (student-dependent; arXiv:2601.14249, official repo
# UmeanNever/RankSurprisalRatio — rsr_cal.py semantics reproduced exactly)
# ---------------------------------------------------------------------------

RSR_RANK_CLIP = 100  # official default rank_clip_r


def _assistant_rank_surprisal(tok, model, chat, max_len) -> tuple:
    """Per assistant token: 1-indexed rank of the gold token among the top
    rank_clip_r logits (clipped), and NLL. Mirrors rsr_cal.py:
        rank = 1 + (topk(logits).values > logit[gold]).sum(); clamp(max=clip)
    Returns (sum_rank, sum_nll, n_tokens, truncated)."""
    import torch
    untruncated = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False)
    ids, spans = _render_ids_and_assistant_spans(tok, chat, max_len)
    truncated = len(untruncated) > max_len
    if len(ids) < 2 or not spans:
        return 0.0, 0.0, 0, truncated
    input_ids = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        logits = model(input_ids).logits[0].float()
    sum_rank = 0.0
    sum_nll = 0.0
    n = 0
    for s, e in spans:
        for pos in range(max(s, 1), e):
            row = logits[pos - 1]
            gold = ids[pos]
            gold_logit = row[gold]
            topv = torch.topk(row, k=min(RSR_RANK_CLIP, row.shape[-1])).values
            rank = 1 + int((topv > gold_logit).sum().item())
            rank = min(rank, RSR_RANK_CLIP)
            nll = float(torch.logsumexp(row, dim=-1).item() - gold_logit.item())
            sum_rank += rank
            sum_nll += nll
            n += 1
    return sum_rank, sum_nll, n, truncated


def compute_rsr(ctx) -> list[dict]:
    """Rank-Surprisal Ratio, official formulation: dataset RSR =
    mean(per-trajectory avg rank) / mean(per-trajectory avg surprisal).
    Per-task rows carry the components (ratio_num = avg rank, ratio_den =
    avg surprisal); teacher-level aggregation MUST be ratio-of-means —
    meta.aggregation tells evaluate_ranking.py to do exactly that. The
    per-task 'score' (avg_rank/avg_surprisal) is a convenience view only.
    Exact token ranks (top-100 comparison), no top-k API approximation."""
    tok, model = _student_resources(ctx)
    max_len = 32768
    meta = {"method": "Rank-Surprisal Ratio",
            "reference": "arXiv:2601.14249",
            "official_code": UPSTREAM_COMMITS["rsr"]["repository"],
            "official_commit": UPSTREAM_COMMITS["rsr"]["commit"],
            "rank_clip": RSR_RANK_CLIP, "max_len": max_len,
            "direction": "higher_better", "student_dependent": True,
            "aggregation": "ratio_of_means"}
    rows = []
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            if (teacher, task_id) in ctx["done"]:
                continue
            rec = recs.get(task_id)
            if rec is None:
                rows.append({"proxy": "rsr", "teacher": teacher,
                             "task_id": task_id, "score": None,
                             "missing": True, "meta": meta})
                continue
            chat = _conversation_as_chat(rec["conversations"])
            s_rank, s_nll, n, truncated = _assistant_rank_surprisal(
                tok, model, chat, max_len)
            avg_rank = (s_rank / n) if n else None
            avg_sur = (s_nll / n) if n else None
            rows.append({
                "proxy": "rsr", "teacher": teacher, "task_id": task_id,
                "score": (avg_rank / avg_sur) if n and avg_sur else None,
                "ratio_num": avg_rank, "ratio_den": avg_sur,
                "tokens_scored": n, "truncated": truncated, "meta": meta})
        print(f"[rsr] {teacher} done")
    return rows


# ---------------------------------------------------------------------------
# Shared TB2.0 turn judging (used by cmd_error and scrf)
# ---------------------------------------------------------------------------

_OUTPUT_TAIL = 4000  # chars of terminal output shown to the judge (tail)


# A shell prompt line as echoed on the captured screen: `root@<id>:<cwd># cmd`
# (Docker, teacher data) or `Apptainer> cmd` (student runs on Capella).
_OMISSION_MARKER = "interior bytes omitted"  # Terminus-2 _limit_output_length
_PROMPT_LINE = re.compile(
    r"^\s*(?:\(\S+\)\s+)?(?:[^\s@]+@[0-9a-f]+:(?P<cwd>[^\n]*?)#|Apptainer>) ?(?P<cmd>.*)$",
    re.M)  # optional leading (venv)/(conda) prompt prefix


_HEREDOC_RE = re.compile(r"<<-?\s*[\'\"]?([A-Za-z_][A-Za-z0-9_]*)[\'\"]?")


def _scan_line_state(line: str, quote: str) -> tuple[str, bool]:
    """Advance shell quoting state across one line. `quote` is the currently
    open quote char ('' if none). Returns (new_quote, backslash_continuation)."""
    esc = False
    for ch in line:
        if esc:
            esc = False
            continue
        if ch == "\\" and quote != "'":
            esc = True
            continue
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
    cont = line.endswith("\\") and not quote  # trailing backslash outside quotes
    return quote, cont


def _split_keystrokes(keystrokes: str) -> list[str]:
    """Shell commands in one keystrokes string, in order. A newline starts a
    new command only at top level: not inside an open quote, a heredoc body, or
    after a backslash continuation. Returns each command's first line (what the
    shell echoes after the prompt)."""
    lines = keystrokes.split("\n")
    cmds = []
    quote = ""            # open quote carried across lines
    cont = False          # previous line ended with backslash continuation
    heredoc = None        # open heredoc delimiter
    i = 0
    while i < len(lines):
        line = lines[i]
        if heredoc is not None:
            if line.strip() == heredoc:
                heredoc = None
            i += 1
            continue
        if quote or cont:
            quote, cont = _scan_line_state(line, quote)
            i += 1
            continue
        if line.strip():
            cmds.append(line.strip())
            delims = _HEREDOC_RE.findall(line)
            quote, cont = _scan_line_state(line, "")
            if delims and not quote:
                heredoc = delims[0]
        i += 1
    return cmds


def _turn_commands(parsed: dict) -> list[str]:
    """Commands issued in one Terminus-2 turn, from the JSON response, in order
    (heredoc bodies and continuation lines are not separate commands)."""
    cmds = []
    for c in parsed.get("commands") or []:
        cmds.extend(_split_keystrokes(c.get("keystrokes") or ""))
    return cmds


def _echo_match(json_cmd: str, screen_cmd: str) -> bool:
    """A JSON command matches its screen echo if, after stripping, the shorter
    is a prefix of the longer (the echo can be wrapped or truncated)."""
    a, b = json_cmd.strip(), screen_cmd.strip()
    if not a or not b:
        return False
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    return hi.startswith(lo)


def _command_segments(commands: list[str], screen: str) -> list[dict]:
    """Pair each command issued in the turn (authoritative list from the JSON,
    `commands`) with its own output on the captured screen.

    The screen's prompt lines mark where each command ran; we align the JSON
    commands to those echoes in order (so we never depend on the prompt format
    to decide *what* a command is, and an output line that merely looks like a
    prompt cannot become a spurious command). A command whose echo is not on
    the screen (its prompt fell inside Terminus-2's omitted >10 KB middle, or
    scrolled off) is reported with observed=False and no output. Output for a
    matched command runs to the next matched command's echo; a >10 KB omission
    marker inside it truncates it (so it cannot absorb a later command)."""
    hits = list(_PROMPT_LINE.finditer(screen))
    segs = []
    h = 0  # screen-prompt cursor, advanced in order
    for ci, cmd in enumerate(commands):
        # find the next screen prompt whose echo matches this command
        m = None
        k = h
        while k < len(hits):
            if _echo_match(cmd, hits[k].group("cmd") or ""):
                m = hits[k]
                h = k + 1
                break
            k += 1
        if m is None:
            segs.append({"input": cmd[-_OUTPUT_TAIL:], "output": "",
                         "cwd": None, "observed": False})
            continue
        # output = from this echo to the next MATCHED command echo (or screen end)
        end = len(screen)
        if ci + 1 < len(commands):
            for k2 in range(h, len(hits)):
                if _echo_match(commands[ci + 1], hits[k2].group("cmd") or ""):
                    end = hits[k2].start()
                    break
        out = screen[m.end():end].strip("\n")
        cut = out.find(_OMISSION_MARKER)
        if cut >= 0:
            out = out[:cut].rstrip("\n") + "\n[... interior output omitted ...]"
        segs.append({"input": cmd[-_OUTPUT_TAIL:], "output": out[-_OUTPUT_TAIL:],
                     "cwd": m.group("cwd") or None, "observed": True})
    return segs


def _teacher_turn_segments(rec: dict) -> list[dict]:
    """Per-command segments of a teacher trajectory, [{turn, input, output}],
    cut from the screen that followed each command-bearing assistant turn
    (see _command_segments). `turn` indexes the assistant turn for the
    recovery window."""
    conv = rec["conversations"]
    segs = []
    for i, c in enumerate(conv):
        if c["from"] != "gpt":
            continue
        d = parse_gpt_turn(c["value"])
        if not d:
            continue
        cmds = _turn_commands(d)
        if not cmds or i + 1 >= len(conv):
            continue
        for s in _command_segments(cmds, conv[i + 1]["value"]):
            if s["observed"]:
                segs.append({"turn": i, **s})
    return segs


def _judge_teacher_turns(ctx, cache) -> dict:
    """{(teacher, task_id): [ {turn, input, output, failure, category,
    subcategory, ...} ]} for every sampled (teacher, task). Fully cached."""
    from concurrent.futures import ThreadPoolExecutor
    from judge import judge_segment
    labels = {}
    todo = []
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            rec = recs.get(task_id)
            if rec is None:
                continue
            segs = _teacher_turn_segments(rec)
            labels[(teacher, task_id)] = segs
            todo.extend(segs)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(lambda s: s.update(
            judge_segment(cache, s["input"], s["output"])), todo))
    print(f"[tb2-judge] {len(todo)} turns judged in {time.time()-t0:.0f}s "
          f"({len(labels)} teacher-task pairs)")
    return labels


def _judge_cache(ctx):
    """One cache file per judge model: concurrent jobs never share a file."""
    import judge
    slug = judge.JUDGE_MODEL.split("/")[-1].lower()
    return judge.JudgeCache(ctx["run_dir"] / "judge_cache" / f"tb2_{slug}.jsonl")


def _ensure_judge(ctx):
    import judge as judge_mod
    judge_mod.ensure_server(venv_python=sys.executable)
    return judge_mod


# ---------------------------------------------------------------------------
# Proxy: cmd_error (teacher-only; TB2.0 taxonomy, LLM judge)
# ---------------------------------------------------------------------------

def compute_cmd_error(ctx) -> list[dict]:
    """Command error rate per (teacher, task).

    TAXONOMY-AGNOSTIC: the score uses ONLY the binary failure flag from the
    TB2.0 E.3 failure-identification prompt (is this command a failure?):
        score = -(failed commands / commands)
    'fewer command errors = better' is declared a priori; both readings are
    stored (score_views fewer_errors/more_errors). The 11/91 error TAXONOMY
    (E.4) is NOT part of this score -- that weighting is SCRF's job. We still
    record category_counts here as free reporting metadata, because the shared
    judge (judge_segment) classifies every failure for SCRF anyway; nothing in
    the cmd_error number depends on the category.

    Segments are PER COMMAND (JSON `commands` aligned to the screen's prompt
    echoes, see _command_segments), matching TB2.0's per-command unit.
    Deviation from TB2.0: a local open judge model instead of GPT-5-high."""
    jm = _ensure_judge(ctx)
    cache = _judge_cache(ctx)
    labels = _judge_teacher_turns(ctx, cache)
    meta = {"score_basis": "E.3 failure flag only (taxonomy-agnostic); "
            "category_counts are reporting metadata, not scored",
            "judge_model": jm.JUDGE_MODEL, "judge_url": jm.JUDGE_URL,
            "granularity": "per command (JSON commands aligned to screen prompt echoes)",
            "output_tail_chars": _OUTPUT_TAIL,
            "direction": "multiple_predeclared", "student_dependent": False,
            "score_views": ["fewer_errors", "more_errors"]}
    rows = []
    for teacher in ctx["teachers"]:
        for task_id in ctx["task_ids"]:
            if (teacher, task_id) in ctx["done"]:
                continue
            segs = labels.get((teacher, task_id))
            if segs is None:
                rows.append({"proxy": "cmd_error", "teacher": teacher,
                             "task_id": task_id, "score": None,
                             "missing": True, "meta": meta})
                continue
            n = len(segs)
            failed = [s for s in segs if s["failure"]]
            cats = {}
            for s in failed:
                key = f"{s['category']} :: {s['subcategory']}"
                cats[key] = cats.get(key, 0) + 1
            error_rate = (len(failed) / n) if n else None
            rows.append({
                "proxy": "cmd_error", "teacher": teacher, "task_id": task_id,
                "score": -error_rate if error_rate is not None else None,
                "score_views": ({"fewer_errors": -error_rate,
                                 "more_errors": error_rate}
                                if error_rate is not None else {}),
                "cmd_segments": n, "failed_turns": len(failed),
                "category_counts": cats,
                "invalid_taxonomy_pairs": sum(
                    1 for s in failed if not s.get("valid_pair")),
                "judge_unparseable": sum(
                    1 for s in segs if not s.get("judge_parse_ok", True)),
                "meta": meta})
    pooled = {}
    for r in rows:
        if r.get("score") is None:
            continue
        f, n = pooled.get(r["teacher"], (0, 0))
        pooled[r["teacher"]] = (f + r["failed_turns"], n + r["cmd_segments"])
    for teacher, (f, n) in pooled.items():
        print(f"[cmd_error] {teacher}: {f}/{n} commands failed = {f / n:.3f}" if n
              else f"[cmd_error] {teacher}: no commands")
    return rows


# ---------------------------------------------------------------------------
# Proxy: scrf (student-dependent; PROXY_SPEC §§5-8)
# ---------------------------------------------------------------------------

_FORMAT_RETRY_MARKER = "parsing error"
# Extra error class outside the TB2.0 taxonomy (proposed 2026-08-25): the
# agent's response is not a valid Terminus-2 tool call, so no command runs.
FORMAT_CATEGORY = "Agent format"
FORMAT_SUBCATEGORY = "invalid Terminus-2 JSON / parse retry"


def _student_segments_and_format_stats(student_runs: list[Path]) -> tuple:
    """Walk student trial dirs (harbor_jobs/*/task_*/agent/episodes).
    Returns (segments, stats). An episode is:
      * format_error: its response is unparseable as the Terminus-2 JSON OR
        the FOLLOWING episode's prompt is a parse-retry prompt. These are
        agent response-format failures — a property of the agent-model
        interface, NOT a TB2.0 command failure; they are counted separately
        (q_S never includes them) per the 2026-08-18 analysis showing student
        trials dominated by parse-retry loops.
      * command-bearing: parseable with non-empty keystrokes -> judged.
    """
    segments = []  # {trial, reward, input, output}
    stats = {"episodes": 0, "format_error_episodes": 0,
             "command_episodes": 0, "trials": 0}
    for run_dir in student_runs:
        for trial in sorted(run_dir.glob("traces/*/apptainer/harbor_jobs/*/task_*")):
            eps = sorted((trial / "agent").glob("episode-*"),
                         key=lambda q: int(q.name.split("-")[-1]))
            if not eps:
                continue
            stats["trials"] += 1
            reward = None
            rj = trial / "result.json"
            if rj.exists():
                r = json.loads(rj.read_text())
                reward = ((r.get("verifier_result") or {}).get("rewards")
                          or {}).get("reward")
            for j, ep in enumerate(eps):
                resp = ep / "response.txt"
                if not resp.exists():
                    continue
                stats["episodes"] += 1
                nxt_prompt = ""
                if j + 1 < len(eps) and (eps[j + 1] / "prompt.txt").exists():
                    nxt_prompt = (eps[j + 1] / "prompt.txt").read_text(
                        errors="replace")
                d = parse_gpt_turn(resp.read_text(errors="replace"))
                # Harbor's own verdict (its parse-retry message opens the next
                # prompt) defines a format episode; our parse only decides the
                # last episode, which has no next prompt. (n=200: 598/603 agree.)
                if (_FORMAT_RETRY_MARKER in nxt_prompt.lower()[:600]
                        if nxt_prompt else d is None):
                    stats["format_error_episodes"] += 1
                    segments.append({
                        "trial": f"{run_dir.name}/{trial.name}", "reward": reward,
                        "episode": j, "kind": "format",
                        "input": resp.read_text(errors="replace")[-_OUTPUT_TAIL:],
                        "output": nxt_prompt[-_OUTPUT_TAIL:],
                        "failure": True, "category": FORMAT_CATEGORY,
                        "subcategory": FORMAT_SUBCATEGORY, "valid_pair": True})
                    continue
                cmds = _turn_commands(d) if d else []
                if not cmds:
                    continue  # no command ran (task_complete turn, empty list)
                stats["command_episodes"] += 1
                for s in _command_segments(cmds, nxt_prompt):
                    if not s["observed"]:
                        continue
                    segments.append({
                        "trial": f"{run_dir.name}/{trial.name}", "reward": reward,
                        "episode": j, "kind": "command", **s})
    trials = {}
    for s in segments:
        trials.setdefault(s["trial"], []).append(s["kind"])
    stats["never_started_trials"] = sum(
        1 for kinds in trials.values() if "command" not in kinds)
    return segments, stats


def _student_following_turns(segments: list[dict], idx: int, k: int) -> str:
    """The student's next k command episodes (same trial) after segment idx,
    rendered like the teacher-side recovery window."""
    seg = segments[idx]
    out = []
    for other in segments[idx + 1:]:
        if other["trial"] != seg["trial"]:
            break
        out.append(f"## turn +{len(out) + 1} (agent)\n{other['input'][-1500:]}"
                   f"\n## output\n{other['output'][-1500:]}")
        if len(out) >= k:
            break
    return "\n".join(out)


def compute_scrf(ctx) -> list[dict]:
    """Student-Conditioned Recovery Fit (PROXY_SPEC §§5-8), first fixed
    formulation: SCRF(T, task) = sum_e q_S(e) * E_T,task(e) * R_T,task(e).

    q_S(e): frequency of TB2.0 category e among the STUDENT's judged
    command-bearing episodes (primary weighting, declared a priori:
    frequency over ALL student command episodes; failed-/successful-
    trajectory variants recorded in the summary file). Agent response-format
    failures are counted separately and excluded from q_S.
    E_T,task(e): fraction of the teacher trajectory's commands judged
    as category e.  R_T,task(e): fraction of those occurrences with a
    positive §7 local-recovery judgment (K=3, fixed rubric, judge blind to
    teacher identity). Controls (§8) stored per row.
    """
    jm = _ensure_judge(ctx)
    from judge import judge_recovery, judge_segment  # noqa: F401
    cache = _judge_cache(ctx)
    K = 3

    runs_root = ctx["run_dir"].parent
    student_run_names = [s.strip() for s in
                         ctx["args"].student_runs.split(",") if s.strip()]
    student_runs = [runs_root / n for n in student_run_names]
    for r in student_runs:
        if not r.exists():
            raise SystemExit(f"[scrf] student run dir missing: {r} — SCRF "
                             "needs student trajectories (PROXY_SPEC §5).")

    # ---- student error profile q_S (§5)
    segs, fstats = _student_segments_and_format_stats(student_runs)
    print(f"[scrf] student episodes: {fstats}")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(lambda s: s.update(
            judge_segment(cache, s["input"], s["output"])),
            [s for s in segs if s["kind"] == "command"]))
    # Format errors: "recovered" = the next episode of the trial is a valid
    # command episode (no judge involved).
    for i, s in enumerate(segs):
        if s["kind"] == "format":
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            s["recovered"] = bool(nxt and nxt["trial"] == s["trial"]
                                  and nxt["kind"] == "command")
    cmd_segs = [s for s in segs if s["kind"] == "command"]
    # Student-side local recovery: the SAME fixed rubric/window as for the
    # teachers (§9.3), so q_S can weight errors the student did NOT recover
    # from (second predeclared view; refinement proposed 2026-08-25).
    failed_idx = [i for i, s in enumerate(segs)
                  if s["failure"] and s["kind"] == "command"]
    def _judge_student_recovery(i):
        s = segs[i]
        err_text = (f"# Failed commands\n{s['input']}\n"
                    f"# Terminal output (failure)\n{s['output']}")
        s["recovered"] = judge_recovery(
            cache, err_text, _student_following_turns(segs, i, K), K)["recovered"]
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(_judge_student_recovery, failed_idx))

    def _profile(subset, only_unrecovered=False, denominator=None):
        prof = {}
        for s in subset:
            if s["failure"] and (not only_unrecovered or not s.get("recovered")):
                key = f"{s['category']} :: {s['subcategory']}"
                prof[key] = prof.get(key, 0) + 1
        n = denominator if denominator is not None else len(subset)
        return {k: v / n for k, v in prof.items()} if n else {}
    # TB2.0-only views use command episodes; the "+format" view adds the
    # agent-format class over ALL episodes (command + format).
    q_all = _profile(cmd_segs)
    q_unrecovered = _profile(cmd_segs, only_unrecovered=True)
    q_failed = _profile([s for s in cmd_segs if s.get("reward") in (0, 0.0, None)])
    q_success = _profile([s for s in cmd_segs if s.get("reward") == 1.0])
    student_occ = _profile(segs, denominator=1)
    student_unrec = _profile(segs, only_unrecovered=True, denominator=1)
    student_recovery_rate = {k: 1 - student_unrec.get(k, 0) / v
                             for k, v in student_occ.items()}
    q_views_sub = {"qS_all_episodes": q_all, "qS_unrecovered": q_unrecovered,
                   "qS_failed_trajectories": q_failed}
    # Same weightings aggregated to the 11 top-level TB2.0 categories (the two
    # judges agree far better at category than at subcategory level, §8), kept
    # as separate views so both granularities are always reported.
    def _to_category(q):
        out = {}
        for key, v in q.items():
            out[key.split(" :: ")[0]] = out.get(key.split(" :: ")[0], 0.0) + v
        return out
    q_views = {**q_views_sub,
               **{f"{v}_cat": _to_category(q) for v, q in q_views_sub.items()}}
    profile_out = ctx["run_dir"] / "judge_cache" / "student_error_profile.json"
    profile_out.parent.mkdir(parents=True, exist_ok=True)
    profile_out.write_text(json.dumps({
        "student_runs": student_run_names, "episode_stats": fstats,
        "agent_format_error_rate": (
            fstats["format_error_episodes"] / fstats["episodes"]
            if fstats["episodes"] else None),
        "q_all_episodes (PRIMARY)": q_all,
        "q_unrecovered (student errors not locally recovered, same K/rubric)":
            q_unrecovered,
        "student_local_recovery_rate_R_S": student_recovery_rate,
        "student_error_counts": student_occ,
        "q_failed_trajectories": q_failed,
        "q_successful_trajectories": q_success}, indent=2) + "\n")

    # ---- teacher exposure + recovery (§§6-7)
    labels = _judge_teacher_turns(ctx, cache)
    review_samples = []
    meta = {"formulation": "SCRF = sum_e q_S(e)*E_T(e)*R_T(e) (PROXY_SPEC §8)",
            "q_S_weighting": "frequency over all student command episodes "
                             "(a priori primary; variants in "
                             "student_error_profile.json)",
            "recovery_horizon_K": K, "judge_model": jm.JUDGE_MODEL,
            "agent_format_errors_excluded_from_qS": True,
            "direction": "higher_better", "student_dependent": True,
            "score_views": list(q_views),
            "score_views_note": "score == qS_all_episodes; qS_unrecovered "
                                "weights by student errors the same recovery "
                                "judge marked unrecovered; "
                                "qS_failed_trajectories by errors in failed "
                                "student trials. Agent-format episodes are "
                                "counted in episode_stats only (teachers "
                                "never make them)."}
    rows = []
    for teacher in ctx["teachers"]:
        recs = ctx["teacher_records"][teacher]
        for task_id in ctx["task_ids"]:
            if (teacher, task_id) in ctx["done"]:
                continue
            segs_t = labels.get((teacher, task_id))
            if segs_t is None:
                rows.append({"proxy": "scrf", "teacher": teacher,
                             "task_id": task_id, "score": None,
                             "missing": True, "meta": meta})
                continue
            conv = recs[task_id]["conversations"]
            gpt_idx = [i for i, c in enumerate(conv) if c["from"] == "gpt"]
            format_segs = []
            for pos, i in enumerate(gpt_idx):
                if parse_gpt_turn(conv[i]["value"]) is None:
                    nxt_ok = (pos + 1 < len(gpt_idx) and
                              parse_gpt_turn(conv[gpt_idx[pos + 1]]["value"]) is not None)
                    format_segs.append({"turn": i, "failure": True,
                                        "category": FORMAT_CATEGORY,
                                        "subcategory": FORMAT_SUBCATEGORY,
                                        "format_recovered": nxt_ok})
            n = len(segs_t) + len(format_segs)  # all assistant turns considered
            exposure, recovered, occurrences = {}, {}, {}
            n_recov_events = 0
            for s in format_segs:
                key = f"{s['category']} :: {s['subcategory']}"
                occurrences[key] = occurrences.get(key, 0) + 1
                if s["format_recovered"]:
                    recovered[key] = recovered.get(key, 0) + 1
                    n_recov_events += 1
            for s in segs_t:
                if not s["failure"]:
                    continue
                key = f"{s['category']} :: {s['subcategory']}"
                occurrences[key] = occurrences.get(key, 0) + 1
                # following K assistant turns (with their observations)
                following = []
                count = 0
                for i in range(s["turn"] + 1, len(conv)):
                    if conv[i]["from"] == "gpt":
                        count += 1
                        obs = (conv[i + 1]["value"][-1500:]
                               if i + 1 < len(conv) else "")
                        following.append(
                            f"## turn +{count} (agent)\n{conv[i]['value'][-1500:]}"
                            f"\n## output\n{obs}")
                        if count >= K:
                            break
                err_text = (f"# Failed commands\n{s['input']}\n"
                            f"# Terminal output (failure)\n{s['output']}")
                verdict = judge_recovery(cache, err_text,
                                         "\n".join(following), K)
                if verdict["recovered"]:
                    recovered[key] = recovered.get(key, 0) + 1
                    n_recov_events += 1
                if len(review_samples) < 12:
                    review_samples.append({
                        "task_id": task_id, "turn": s["turn"],
                        "category": key, "verdict": verdict,
                        "error_excerpt": s["input"][:300]})
            # subcategory-level components
            comp = {}
            for key in occurrences:
                E = occurrences[key] / n if n else 0.0
                R = recovered.get(key, 0) / occurrences[key]
                comp[key] = {"E_T": E, "R_T": R,
                             **{f"q_S[{v}]": q_views_sub[v].get(key, 0.0)
                                for v in q_views_sub}}
            # category-level components (aggregate occurrences/recovered by cat)
            occ_cat, rec_cat = {}, {}
            for key, occ in occurrences.items():
                cat = key.split(" :: ")[0]
                occ_cat[cat] = occ_cat.get(cat, 0) + occ
                rec_cat[cat] = rec_cat.get(cat, 0) + recovered.get(key, 0)
            comp_cat = {cat: {"E_T": occ_cat[cat] / n if n else 0.0,
                              "R_T": rec_cat[cat] / occ_cat[cat]}
                        for cat in occ_cat}
            views = {}
            for v, q in q_views.items():
                cmap = comp_cat if v.endswith("_cat") else comp
                views[v] = sum(q.get(k, 0.0) * c["E_T"] * c["R_T"]
                               for k, c in cmap.items())
            score = views["qS_all_episodes"]
            n_failed = sum(occurrences.values())
            rows.append({
                "proxy": "scrf", "teacher": teacher, "task_id": task_id,
                "score": score if n else None,
                "score_views": ({v: s for v, s in views.items()} if n else {}),
                "components": comp, "cmd_segments": n,
                "control_unconditioned_recovery_density": (
                    sum(c["E_T"] * c["R_T"] for c in comp.values()) if n else None),
                "control_total_error_rate": (n_failed / n) if n else None,
                "control_recovery_event_rate": (n_recov_events / n) if n else None,
                "control_recovery_rate_unweighted": (
                    n_recov_events / n_failed if n_failed else None),
                "control_error_similarity": _cosine(
                    q_all, {k: v / n for k, v in occurrences.items()} if n else {}),
                "meta": meta})
        print(f"[scrf] {teacher} done")
    review_path = ctx["run_dir"] / "judge_cache" / "scrf_review_samples.json"
    review_path.write_text(json.dumps(review_samples, indent=2) + "\n")
    print(f"[scrf] review samples for manual validation: {review_path}")
    return rows


def _cosine(a: dict, b: dict) -> float | None:
    import math
    keys = set(a) | set(b)
    if not keys:
        return None
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if not na or not nb:
        return 0.0
    return sum(a.get(k, 0) * b.get(k, 0) for k in keys) / (na * nb)


# ---------------------------------------------------------------------------
# Stubs — refuse loudly instead of guessing (PROXY_SPEC.md §11 procedure)
# ---------------------------------------------------------------------------

_STUB_REASON = {
    "lark": "arXiv:2605.30651 — listed in PROXY_SPEC.md §7.8 for completeness; "
            "implementation deferred by decision of 2026-08-25",
}


def compute_stub(name):
    def _fn(ctx):
        raise SystemExit(f"proxy {name!r} is not implemented yet: "
                         f"{_STUB_REASON[name]} (see PROXY_SPEC.md).")
    return _fn


PROXY_FNS = {
    "teacher_bench": compute_teacher_bench,
    "traj_length": compute_traj_length,
    "tor": compute_tor,
    "egs_post": lambda ctx: compute_egs(ctx, "egs_post"),
    "egs_loop": lambda ctx: compute_egs(ctx, "egs_loop"),
    "error_retry": compute_error_retry,
    "global_nll": compute_global_nll,
    **{f"local_nll_k{k}": (lambda ctx, k=k: compute_local_nll(ctx, k))
       for k in LOCAL_NLL_KS},
    "scas": compute_scas,
    "grace": compute_grace,
    "aslec_drop": lambda ctx: compute_aslec(ctx, "aslec_drop"),
    "aslec_casl": lambda ctx: compute_aslec(ctx, "aslec_casl"),
    "rsr": compute_rsr,
    "cmd_error": compute_cmd_error,
    "scrf": compute_scrf,
    **{name: compute_stub(name) for name in ["lark"]},
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="terminal_lego")
    p.add_argument("--student", default="Qwen/Qwen3-8B")
    p.add_argument("--proxy", action="append", choices=ALL_PROXIES,
                   required=True)
    p.add_argument("--n-tasks", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-file", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--student-runs", default="terminal_lego-n10-s42,terminal_lego-n10-s42-rayv",
                   help="comma list of run-ids whose student traces feed the "
                        "SCRF error profile q_S (PROXY_SPEC §5)")
    p.add_argument("--judge-tag", default="",
                   help="suffix for judge-based score files (cmd_error, scrf), "
                        "e.g. '@gpt-5.5' when TRP_JUDGE_MODEL/URL point at a "
                        "second judge; default '' = the Qwen3-32B files")
    p.add_argument("--force", action="store_true",
                   help="recompute the proxy from scratch (drops its file)")
    args = p.parse_args()

    ensure_env()
    runs_root = ensure_runs_root()
    run_id, run_dir, task_ids = resolve_sample(args, runs_root)
    header, _rows = load_manifest(Path(args.manifest))
    teachers = header["teachers"]

    needs_traj = [x for x in args.proxy if x not in ("teacher_bench",)]
    teacher_records = (load_teacher_records(header, teachers)
                       if needs_traj else {})

    print(f"[proxies] run={run_id} student={args.student} "
          f"tasks={len(task_ids)} proxies={args.proxy}")
    shared = {}
    for proxy in args.proxy:
        path = score_path(run_dir, args.student,
                          proxy + (args.judge_tag if proxy in JUDGE_PROXIES else ""))
        if args.force and path.exists():
            path.unlink()
        done = existing_keys(path)
        expected = {(t, tid) for t in teachers for tid in task_ids}
        if expected <= done:
            print(f"[{proxy}] all {len(expected)} rows cached -> {path}")
            continue
        ctx = {"args": args, "student": args.student, "teachers": teachers,
               "task_ids": task_ids, "teacher_records": teacher_records,
               "header": header, "run_dir": run_dir, "done": done,
               "shared": shared}
        t0 = time.time()
        rows = [r for r in PROXY_FNS[proxy](ctx)
                if (r["teacher"], r["task_id"]) not in done]
        wall = time.time() - t0
        for r in rows:
            r.setdefault("meta", {})
            r["meta"]["wall_clock_s_batch"] = round(wall, 1)
        append_rows(path, rows)
        print(f"[{proxy}] wrote {len(rows)} rows in {wall:.1f}s -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
