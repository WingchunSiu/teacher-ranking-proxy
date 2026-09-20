"""CLI:  python -m litmine run --seeds 2606.03461 [--backend both] [--no-search]
        python -m litmine paper 2606.03461          # one paper end-to-end
        python -m litmine export                    # re-export from state (no LLM calls)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings
from .discovery import Discovery, keyword_gate
from .pipeline import Pipeline

# Rough per-paper USD costs measured on the first sweep (gpt-5.5 $1.25/$10 per M,
# deepseek-v4-flash $0.14/$0.28 per M). Only used for the printed estimate.
EST = {"prescreen": {"openai": 0.004, "deepseek": 0.0004},
       "screen": {"openai": 0.04, "deepseek": 0.008},          # full-text screening, per backend
       "extract": {"openai": 0.13, "deepseek": 0.02}}          # extraction + adjudication + artifacts, per backend


def cost_note(pipe: Pipeline, n_prescreen: int = 0, n_screen: int = 0) -> str:
    pb = pipe.prescreen_client.cfg.name
    backs = list(pipe.clients)
    lines = []
    if n_prescreen:
        lines.append(f"estimated prescreen cost for {n_prescreen} papers on {pb}: "
                     f"${n_prescreen * EST['prescreen'].get(pb, 0.004):.2f}")
    if n_screen:
        scr = sum(EST["screen"].get(b, 0.04) for b in backs)
        ext = sum(EST["extract"].get(b, 0.13) for b in backs)
        lines.append(f"estimated full-text screening of {n_screen} papers on {backs}: ${n_screen * scr:.2f}; "
                     f"extraction if ~20% pass: ${0.2 * n_screen * ext:.2f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litmine")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--backend", default=None, help="openai|deepseek|both (env LLM_BACKEND)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="full pipeline")
    r.add_argument("--seeds", nargs="*", default=["2606.03461"])
    r.add_argument("--no-search", action="store_true", help="seeds + citation expansion only")
    r.add_argument("--no-github", action="store_true")
    r.add_argument("--citation-depth", type=int, default=None)
    r.add_argument("--max-candidates", type=int, default=None)
    r.add_argument("--force", action="store_true", help="ignore per-paper state (LLM cache still used)")
    r.add_argument("--prescreen-only", action="store_true",
                   help="discover + prescreen, then stop and print what full-text screening would cost")
    p = sub.add_parser("paper", help="process a single arXiv id or URL")
    p.add_argument("ref")
    p.add_argument("--force", action="store_true")
    sub.add_parser("export", help="regenerate CSV/JSONL/review/summary from state")
    d = sub.add_parser("discover", help="discovery only")
    d.add_argument("--seeds", nargs="*", default=["2606.03461"])
    d.add_argument("--citation-depth", type=int, default=None)
    d.add_argument("--no-github", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    s = Settings()
    if a.work_dir:
        s.work_dir = a.work_dir
    if a.backend:
        s.llm_backend = a.backend

    if a.cmd == "export":
        pipe = Pipeline(s)  # LLM clients are constructed lazily; export makes no calls
        paths = pipe.export()
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))
        return 0
    pipe = Pipeline(s)
    if a.cmd == "discover":
        cands = pipe.discover(a.seeds, citation_depth=a.citation_depth, use_github=not a.no_github)
        pipe.fill_metadata(cands)
        gated = sum(1 for c in cands if keyword_gate(c.title, c.abstract))
        print(json.dumps({"sources": pipe.stats.get("discovery_sources"),
                          "seed_titles": pipe.stats.get("seed_titles_resolved")}, indent=1))
        print(f"{len(cands)} candidates, {gated} pass the keyword gate -> {pipe.state_dir / 'candidates.json'}")
        print(cost_note(pipe, n_prescreen=gated))
        return 0
    if a.cmd == "paper":
        disc = Discovery(pipe.fetcher, pipe.cache)
        cand = disc.add_seed(a.ref)
        res = pipe.process_candidate(cand, force=a.force)
        pipe.upsert_records(cand.arxiv_id or cand.url, res["records"])
        paths = pipe.export()
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))
        print(json.dumps({"decision": res.get("decision"), "n_records": len(res["records"]),
                          "records": [{"id": r["record_id"], "status": r["classification"]["status"],
                                       "student": (r["experiment"].get("student_model") or {}).get("value"),
                                       "ranking": r["ranking"]["ranking_string"],
                                       "benchmark": r["ranking"]["benchmark"]}
                                      for r in res["records"]]}, indent=1))
        return 0
    if a.cmd == "run":
        recs = pipe.run(a.seeds, run_search=not a.no_search, max_candidates=a.max_candidates,
                        citation_depth=a.citation_depth, use_github=not a.no_github, force=a.force,
                        prescreen_only=a.prescreen_only)
        if a.prescreen_only:
            c = pipe.stats["prescreen_counts"]
            n = min(c["yes"] + c["unclear"], a.max_candidates or s.max_candidates)
            print(json.dumps(c, indent=1))
            print(cost_note(pipe, n_screen=n))
            print(f"prescreen decisions -> {pipe.state_dir / 'prescreen.json'}")
            return 0
        paths = pipe.export(recs)
        print(json.dumps({"records": len(recs), **{k: str(v) for k, v in paths.items()}}, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
