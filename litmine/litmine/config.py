"""Configuration: env-driven, never logs secrets."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent

# .env next to teacher_ranking_proxy (symlink to the user's secrets file) is
# loaded if present, but env vars already set always win.
_ENV_FILES = [ROOT_DIR.parent / ".env", ROOT_DIR / ".env"]


def load_dotenv() -> None:
    for p in _ENV_FILES:
        try:
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
        except OSError:
            continue


load_dotenv()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL,
            timeout=10).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class BackendConfig:
    name: str                      # openai | deepseek | fake
    base_url: str
    api_key_env: str
    model: str
    temperature: float | None      # None => provider default / not sent
    reasoning_effort: str | None = None
    timeout: int = 600

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key and self.name != "fake":
            raise RuntimeError(
                f"{self.api_key_env} is not set (needed for backend {self.name})")
        return key


def backend_config(name: str) -> BackendConfig:
    name = name.lower()
    if name == "openai":
        return BackendConfig(
            name="openai",
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key_env="OPENAI_API_KEY",
            model=os.environ.get("LITMINE_OPENAI_MODEL", "gpt-5.5"),
            temperature=None,  # gpt-5.x reasoning models reject temperature
            reasoning_effort=os.environ.get("LITMINE_OPENAI_REASONING", "medium"),
        )
    if name == "deepseek":
        return BackendConfig(
            name="deepseek",
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key_env="DEEPSEEK_API_KEY",
            model=os.environ.get("LITMINE_DEEPSEEK_MODEL", "deepseek-v4-flash"),
            temperature=float(os.environ.get("LITMINE_DEEPSEEK_TEMPERATURE", "0.0")),
        )
    if name == "fake":
        return BackendConfig(name="fake", base_url="", api_key_env="",
                             model="fake-model", temperature=0.0)
    raise ValueError(f"unknown LLM backend: {name}")


def backends_for(setting: str) -> list[str]:
    """LLM_BACKEND=openai|deepseek|both|fake -> list of backend names."""
    s = (setting or "openai").lower()
    if s == "both":
        return ["openai", "deepseek"]
    return [s]


@dataclass
class Settings:
    work_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("LITMINE_WORK_DIR", str(ROOT_DIR / "work"))))
    llm_backend: str = field(default_factory=lambda: os.environ.get("LLM_BACKEND", "openai"))
    adjudicator: str = field(default_factory=lambda: os.environ.get("LITMINE_ADJUDICATOR", "openai"))
    max_doc_chars: int = field(default_factory=lambda: int(
        os.environ.get("LITMINE_MAX_DOC_CHARS", "260000")))
    max_candidates: int = field(default_factory=lambda: int(
        os.environ.get("LITMINE_MAX_CANDIDATES", "5000")))
    per_query_results: int = field(default_factory=lambda: int(
        os.environ.get("LITMINE_PER_QUERY", "100")))
    citation_depth: int = field(default_factory=lambda: int(
        os.environ.get("LITMINE_CITATION_DEPTH", "2")))
    # cheaper model for the title/abstract prescreen (same provider as the primary backend)
    prescreen_model: str | None = field(default_factory=lambda: os.environ.get("LITMINE_PRESCREEN_MODEL") or None)
    screen_model: str | None = field(default_factory=lambda: os.environ.get("LITMINE_SCREEN_MODEL") or None)
    # backend for the title/abstract prescreen; default: deepseek when a key is
    # present (about 10x cheaper than gpt-5.5), else the primary backend
    prescreen_backend: str | None = field(default_factory=lambda: os.environ.get(
        "LITMINE_PRESCREEN_BACKEND") or ("deepseek" if os.environ.get("DEEPSEEK_API_KEY") else None))
    # free regex gate (agentic AND training-data vocabulary) before any LLM prescreen
    keyword_gate: bool = field(default_factory=lambda: os.environ.get("LITMINE_KEYWORD_GATE", "1") != "0")
    # JSON file {candidate_key: {relevant, reason, confidence}}; when set, prescreen
    # decisions come from this file and no prescreen LLM calls are made
    prescreen_file: str | None = field(default_factory=lambda: os.environ.get("LITMINE_PRESCREEN_FILE") or None)
    http_timeout: int = 60
    user_agent: str = "litmine/0.1 (teacher-ranking literature mining; contact via repo)"

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"

    @property
    def out_dir(self) -> Path:
        return self.work_dir / "out"

    def backends(self) -> list[str]:
        return backends_for(self.llm_backend)
