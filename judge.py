"""LLM judge for the teacher-ranking proxy pipeline.

Serves two fixed rubrics:
  * TB2.0 command-failure judging (arXiv:2601.11868 App E.3/E.4) — prompts and
    taxonomy are transcribed VERBATIM in artifacts/tb2_taxonomy.json; never
    edit them casually.
  * SCRF local-recovery judging (PROXY_SPEC.md §7) — fixed rubric below;
    judge is blind to teacher identity and published rankings (segments carry
    no identity, and none is ever put in a prompt).

Judge model: a locally served open model via vLLM (OpenAI-compatible API).
Default Qwen/Qwen3-32B on the allocation's GPU — no external API, no cost,
reproducible. The exact model + endpoint are recorded per call in the cache
and per row in score meta. (The TB2.0 paper used GPT-5-high, 82% agreement
with humans; our judge substitution is a documented deviation.)

Env:
  TRP_JUDGE_URL    default http://127.0.0.1:8001/v1
  TRP_JUDGE_MODEL  default Qwen/Qwen3-32B

Every call is cached (jsonl, sha256 of rubric+model+payload) so reruns and
resumes never re-judge.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TAXONOMY_FILE = SCRIPT_DIR / "artifacts" / "tb2_taxonomy.json"

JUDGE_URL = os.environ.get("TRP_JUDGE_URL", "http://127.0.0.1:8001/v1")
JUDGE_MODEL = os.environ.get("TRP_JUDGE_MODEL", "Qwen/Qwen3-32B")
# Remote OpenAI endpoint (second, independent labeling for judge agreement):
#   TRP_JUDGE_URL=https://api.openai.com/v1 TRP_JUDGE_MODEL=gpt-5.5
# The key comes from OPENAI_API_KEY (secrets file); nothing else changes —
# prompts, rubric, parsing and caching are identical, cache keys include the
# model id.
REMOTE_JUDGE = "127.0.0.1" not in JUDGE_URL and "localhost" not in JUDGE_URL
OPENAI_REASONING = REMOTE_JUDGE and "openai.com" in JUDGE_URL  # gpt-5.x payload
GPT_OSS = "gpt-oss" in JUDGE_MODEL  # local openai/gpt-oss-* via vLLM
# TRP_JUDGE_CACHE_ONLY=1: never start or call a judge; every judgment must be
# in the cache (recompute score files from cached labels on a CPU node).
CACHE_ONLY = os.environ.get("TRP_JUDGE_CACHE_ONLY") == "1"


def _auth_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if REMOTE_JUDGE:
        # key: TRP_JUDGE_API_KEY, else by provider (DEEPSEEK_API_KEY for
        # api.deepseek.com, OPENAI_API_KEY otherwise)
        key = os.environ.get("TRP_JUDGE_API_KEY") or os.environ.get(
            "DEEPSEEK_API_KEY" if "deepseek" in JUDGE_URL else "OPENAI_API_KEY")
        if not key:
            raise SystemExit("remote judge needs an API key in the environment")
        h["Authorization"] = f"Bearer {key}"
    return h

# PROXY_SPEC.md §9.3: fixed local-recovery rubric; K (inspection horizon) is
# passed by the caller and recorded in results. The judge sees only text.
RECOVERY_RUBRIC = """You are an expert at analyzing terminal agent behavior. You are shown:
(1) an agent turn in which a command failed, with the terminal output showing the failure, and
(2) the agent's next {k} turns (commands and terminal outputs).

Decide whether the agent genuinely recovered from this specific failure within the shown turns.
Answer recovered = true only if both hold:
(a) the agent takes a changed action that addresses the cause of this failure (fixes the command, its arguments, the file, or the environment, or switches to a working alternative for the same goal), and
(b) the shown outputs give evidence that this operation then succeeded or that the cause is gone.

Not a recovery: repeating the same command unchanged; moving on to unrelated work; declaring the task complete; only inspecting or explaining the error without a corrective action; a corrective action whose result is not visible in the shown turns.

Output only JSON with three keys:
"recovered": true or false
"fix_action": the changed action that addressed the cause, or "" if none
"evidence": one sentence citing the concrete output that shows success, or why recovery is absent"""

# Appended to the verbatim TB2.0 E.3 failure prompt (our one addition).
E3_CLARIFICATION = "\nAn error message that is the intended outcome of the command is not a failure."


def _load_tax() -> dict:
    return json.loads(TAXONOMY_FILE.read_text())


def taxonomy_text() -> str:
    tax = _load_tax()["taxonomy"]
    lines = []
    for cat, subs in tax.items():
        lines.append(f"* {cat}")
        for sub, desc in subs.items():
            lines.append(f"  - {sub}: {desc}")
    return "\n".join(lines)


def valid_pairs() -> set:
    tax = _load_tax()["taxonomy"]
    return {(c, s) for c, subs in tax.items() for s in subs}


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def _healthy(url: str) -> bool:
    try:
        req = urllib.request.Request(f"{url}/models", headers=_auth_headers())
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_server(venv_python: str | None = None, max_wait_s: int = 2400,
                  max_model_len: int = 16384) -> None:
    """Reuse a healthy judge endpoint, else launch vLLM on the local GPU.
    Blocks until healthy. The server is left running (killed with the job)."""
    if CACHE_ONLY:
        print("[judge] cache-only mode: no server; uncached calls will abort")
        return
    if _healthy(JUDGE_URL):
        return
    if REMOTE_JUDGE:
        raise SystemExit(f"[judge] remote endpoint {JUDGE_URL} not reachable")
    port = JUDGE_URL.rsplit(":", 1)[-1].split("/")[0]
    py = venv_python or sys.executable
    log = Path(os.environ.get("TMPDIR", "/tmp")) / "trp_judge_vllm.log"
    cmd = [py, "-m", "vllm.entrypoints.openai.api_server",
           "--model", JUDGE_MODEL, "--host", "127.0.0.1", "--port", port,
           "--max-model-len", str(max_model_len),
           "--gpu-memory-utilization", "0.92", "--disable-log-requests"]
    if GPT_OSS:
        # MXFP4 checkpoint; separate the analysis channel from the answer
        cmd += ["--reasoning-parser", "openai_gptoss"]
    else:
        cmd += ["--dtype", "bfloat16"]
    print(f"[judge] launching {JUDGE_MODEL} on :{port} (log: {log})")
    with open(log, "ab") as lf:
        subprocess.Popen(cmd, stdout=lf, stderr=lf, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        if _healthy(JUDGE_URL):
            print(f"[judge] up after {time.time()-t0:.0f}s")
            return
        time.sleep(10)
    raise SystemExit(f"[judge] server not healthy after {max_wait_s}s; see {log}")


# ---------------------------------------------------------------------------
# Cached chat call
# ---------------------------------------------------------------------------

class JudgeCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem = {}
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self._mem[r["key"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue  # torn line from a concurrent writer; re-judged

    def get(self, key):
        return self._mem.get(key)

    def put(self, rec):
        with self._lock:
            self._mem[rec["key"]] = rec
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")  # one write per record


def _chat(messages: list[dict], max_tokens: int = 512) -> str:
    payload = {"model": JUDGE_MODEL, "messages": messages}
    if OPENAI_REASONING:
        # OpenAI reasoning models: no temperature / chat_template_kwargs;
        # budget covers hidden reasoning + the short JSON answer.
        payload["max_completion_tokens"] = max(max_tokens, 2048)
        payload["reasoning_effort"] = "low"
    elif REMOTE_JUDGE:
        # OpenAI-compatible providers (DeepSeek): plain chat completion,
        # thinking disabled (it otherwise consumes the answer budget)
        payload.update({"temperature": 0.0, "max_tokens": max(max_tokens, 1024)})
        if "deepseek" in JUDGE_URL:
            payload["thinking"] = {"type": "disabled"}
    elif GPT_OSS:
        # gpt-oss always reasons; give headroom so the JSON answer is not
        # truncated by the reasoning (a truncated turn returns content=None)
        payload.update({"temperature": 0.0, "max_tokens": max(max_tokens, 4096),
                        "chat_template_kwargs": {"reasoning_effort": "low"}})
    else:
        payload.update({
            "temperature": 0.0, "max_tokens": max_tokens,
            # Qwen3: judging needs no long CoT; disable thinking for determinism+speed
            "chat_template_kwargs": {"enable_thinking": False}})
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{JUDGE_URL}/chat/completions", data=body, headers=_auth_headers())
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"].get("content") or ""
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`\n")
        if text.startswith("json"):
            text = text[4:]
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        # last-resort: first {...} block
        s, e = text.find("{"), text.rfind("}")
        if 0 <= s < e:
            try:
                d = json.loads(text[s:e + 1])
                return d if isinstance(d, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _cached_judgment(cache: JudgeCache, kind: str, system: str, user: str,
                     max_tokens: int = 512) -> dict | None:
    key = hashlib.sha256(
        f"{kind}\x00{JUDGE_MODEL}\x00{system}\x00{user}".encode()).hexdigest()
    hit = cache.get(key)
    if hit is not None and hit.get("parsed") is not None:
        return hit["parsed"]  # unparseable cached answers are retried
    if CACHE_ONLY:
        raise SystemExit(f"[judge] cache-only mode but {kind} judgment for "
                         f"{JUDGE_MODEL} is not cached")
    raw = _chat([{"role": "system", "content": system},
                 {"role": "user", "content": user}], max_tokens)
    parsed = _parse_json(raw)
    cache.put({"key": key, "kind": kind, "model": JUDGE_MODEL,
               "user_sha": hashlib.sha256(user.encode()).hexdigest(),
               "raw": raw, "parsed": parsed, "ts": time.time()})
    return parsed


# ---------------------------------------------------------------------------
# Public rubric calls
# ---------------------------------------------------------------------------

def judge_segment(cache: JudgeCache, input_text: str, output_text: str) -> dict:
    """TB2.0 two-stage judging of one command→output segment.
    Returns {"failure": bool, "category": str, "subcategory": str,
             "valid_pair": bool}."""
    prompts = _load_tax()["judge_prompts"]
    seg = (f"# Input (commands sent to the terminal)\n{input_text}\n\n"
           f"# Captured output\n{output_text}")
    p1 = _cached_judgment(cache, "tb2_failure",
                          prompts["failure_identification_E3"] + E3_CLARIFICATION,
                          seg, 128)
    failure = bool(p1 and p1.get("is_failure_present"))
    if not failure:
        return {"failure": False, "category": "", "subcategory": "",
                "valid_pair": True, "judge_parse_ok": p1 is not None}
    sys_prompt = prompts["taxonomy_classification_E4"].replace(
        "{{taxonomy}}", taxonomy_text())
    p2 = _cached_judgment(cache, "tb2_taxonomy", sys_prompt, seg, 256)
    cat = (p2 or {}).get("error_category", "") or ""
    sub = (p2 or {}).get("error_subcategory", "") or ""
    return {"failure": True, "category": cat, "subcategory": sub,
            "valid_pair": (cat, sub) in valid_pairs(),
            "judge_parse_ok": p2 is not None}


def judge_recovery(cache: JudgeCache, error_turn_text: str,
                   following_turns_text: str, k: int) -> dict:
    """PROXY_SPEC §7 local-recovery judgment. Blind: caller must pass text
    containing no teacher identity. Returns {"recovered": bool|None,
    "evidence": str}."""
    system = RECOVERY_RUBRIC.format(k=k)
    user = (f"# Error turn (command failed)\n{error_turn_text}\n\n"
            f"# Next {k} turns\n{following_turns_text}")
    p = _cached_judgment(cache, "scrf_recovery", system, user, 256)
    if p is None:
        return {"recovered": None, "evidence": "judge output unparseable"}
    return {"recovered": bool(p.get("recovered")),
            "fix_action": str(p.get("fix_action", ""))[:300],
            "evidence": str(p.get("evidence", ""))[:500]}
