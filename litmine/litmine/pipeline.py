"""Pipeline orchestration (PIPELINE.md §4): every stage is cached and
independently rerunnable; the state of a run lives in work/state/ as plain
JSON so a crashed run resumes from the last completed paper."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import PIPELINE_VERSION
from .artifacts import ArtifactVerifier
from .cache import Cache, cache_key
from .classify import classify, rank_columns, rank_from_scores, ranking_string
from .config import Settings, git_commit
from .discovery import Candidate, Discovery, keyword_gate
from .llm import LLMClient, LLMError, make_client
from .multimodel import adjudicate, diff_experiments, merge_with_adjudication, primary_scores, _align
from .prompts import (ADJUDICATION_VERSION, ARTIFACT_VERSION, EXTRACTION_SYSTEM, EXTRACTION_VERSION,
                      PRESCREEN_SYSTEM, PRESCREEN_VERSION, SCREENING_SYSTEM, SCREENING_VERSION,
                      document_user, prescreen_user)
from .retrieval import Fetcher, retrieve_document
from .schemas import validate_extraction, validate_screening

log = logging.getLogger("litmine.pipeline")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _validate_prescreen(obj: Any) -> dict:
    if not isinstance(obj, dict) or str(obj.get("relevant", "")).lower() not in ("yes", "no", "unclear"):
        raise ValueError("prescreen must return relevant: yes|no|unclear")
    return {"relevant": str(obj["relevant"]).lower(), "reason": str(obj.get("reason", "")),
            "confidence": obj.get("confidence")}


@dataclass
class Pipeline:
    settings: Settings
    clients: dict[str, LLMClient] = field(default_factory=dict)
    adjudicator: LLMClient | None = None
    fetcher: Fetcher | None = None
    cache: Cache | None = None
    stats: dict = field(default_factory=dict)

    def __post_init__(self):
        s = self.settings
        s.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache = self.cache or Cache(s.cache_dir)
        self.fetcher = self.fetcher or Fetcher(self.cache, s.user_agent, s.http_timeout)
        if not self.clients:
            self.clients = {b: make_client(b, self.cache) for b in s.backends()}
        if self.adjudicator is None and len(self.clients) > 1:
            self.adjudicator = self.clients.get(s.adjudicator) or make_client(s.adjudicator, self.cache)
        self.verifier = ArtifactVerifier(self.fetcher, self.cache)
        # optional cheaper models for the gating stages (same provider as primary)
        self.prescreen_client = self._variant_client(s.prescreen_model)
        if (s.prescreen_backend and type(self.primary) is LLMClient      # never swap in a real backend under a test double
                and s.prescreen_backend != self.primary.cfg.name):
            base = self.clients.get(s.prescreen_backend) or make_client(s.prescreen_backend, self.cache)
            self.prescreen_client = base
            if s.prescreen_model and s.prescreen_model != base.cfg.model:
                import copy
                cfg = copy.copy(base.cfg)
                cfg.model = s.prescreen_model
                self.prescreen_client = LLMClient(cfg, self.cache)
        self.screen_clients = {n: (self._variant_client(s.screen_model, n) if s.screen_model else c)
                               for n, c in self.clients.items()}
        self.state_dir = s.work_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.stats.setdefault("pipeline_version", PIPELINE_VERSION)
        self.stats.setdefault("git_commit", git_commit())

    def _variant_client(self, model: str | None, name: str | None = None) -> LLMClient:
        base = self.clients[name] if name else self.primary
        if not model or model == base.cfg.model or base.cfg.name == "fake":
            return base
        import copy
        cfg = copy.copy(base.cfg)
        cfg.model = model
        return LLMClient(cfg, self.cache)

    @property
    def primary(self) -> LLMClient:
        return next(iter(self.clients.values()))

    # ------------------------------------------------------------------ stages
    def discover(self, seeds: list[str], *, run_search: bool = True, citation_depth: int | None = None,
                 use_github: bool = True) -> list[Candidate]:
        d = Discovery(self.fetcher, self.cache, per_query=self.settings.per_query_results)
        depth = self.settings.citation_depth if citation_depth is None else citation_depth
        if run_search:
            cands = d.run(seeds, citation_depth=depth, use_github=use_github)
        else:
            cands = d.run(seeds, arxiv_queries=[], free_queries=[], hf_queries=[], github_queries=[],
                          citation_depth=depth, use_github=False)
        self.stats["papers_discovered"] = len(cands)
        self.stats["discovery_sources"] = d.stats
        self.stats["seed_titles_resolved"] = d.seed_resolution
        self.stats["papers_keyword_gated"] = sum(1 for c in cands if keyword_gate(c.title, c.abstract))
        self._discovery = d
        (self.state_dir / "candidates.json").write_text(
            json.dumps([c.to_dict() for c in cands], indent=1, ensure_ascii=False))
        log.info("discovery: %d candidates (%s)", len(cands), d.stats)
        return cands

    def retrieve(self, cand: Candidate) -> dict:
        doc = retrieve_document(self.fetcher, self.cache, arxiv_id=cand.arxiv_id,
                                url=None if cand.arxiv_id else cand.url,
                                max_chars=self.settings.max_doc_chars)
        if doc.get("title"):
            cand.title = cand.title or doc["title"]
        if doc.get("abstract"):
            cand.abstract = cand.abstract or doc["abstract"]
        cand.year = cand.year or doc.get("year")
        return doc

    def prescreen(self, cand: Candidate) -> dict:
        """Title/abstract screening with the primary backend only (cheap gate)."""
        if "seed" in cand.sources:
            return {"relevant": "yes", "reason": "seed paper", "confidence": 1.0, "skipped": True}
        if not cand.abstract:
            return {"relevant": "unclear", "reason": "no abstract available", "confidence": None}
        if self.settings.keyword_gate and not keyword_gate(cand.title, cand.abstract):
            return {"relevant": "no", "reason": "keyword gate: no agentic + training-data vocabulary",
                    "confidence": None, "gated": True}
        if self.settings.prescreen_file:
            ext = self._external_prescreen()
            d = ext.get(cand.key)
            if d is not None:
                return {"relevant": str(d.get("relevant", "unclear")).lower(),
                        "reason": str(d.get("reason", "")), "confidence": d.get("confidence"),
                        "external": True}
            log.warning("candidate %s missing from prescreen file; treating as no", cand.key)
            return {"relevant": "no", "reason": "not in external prescreen file", "confidence": None,
                    "external": True}
        extra = f"TAGS: {', '.join(cand.tags)}" if cand.tags else ""
        res = self.prescreen_client.complete_json("prescreen", PRESCREEN_VERSION, PRESCREEN_SYSTEM,
                                         prescreen_user(cand.title, cand.abstract, extra),
                                         validator=_validate_prescreen, max_tokens=2000)
        return {**res.parsed, "llm": res.meta()}

    def _external_prescreen(self) -> dict:
        if not hasattr(self, "_external_prescreen_cache"):
            self._external_prescreen_cache = json.loads(Path(self.settings.prescreen_file).read_text())
        return self._external_prescreen_cache

    def screen(self, doc: dict) -> dict[str, dict]:
        out = {}
        user = document_user(doc.get("title", ""), doc.get("source_url", ""), doc["text"])
        for name, c in self.screen_clients.items():
            res = c.complete_json("screen", SCREENING_VERSION, SCREENING_SYSTEM, user,
                                  validator=validate_screening, max_tokens=8000,
                                  extra_key=doc["doc_hash"])
            out[name] = {**res.parsed, "llm": res.meta()}
        return out

    def extract(self, doc: dict, focus: str = "") -> dict[str, dict]:
        out = {}
        user = document_user(doc.get("title", ""), doc.get("source_url", ""), doc["text"], focus)
        for name, c in self.clients.items():
            res = c.complete_json("extract", EXTRACTION_VERSION, EXTRACTION_SYSTEM, user,
                                  validator=validate_extraction, max_tokens=24000,
                                  extra_key=doc["doc_hash"])
            out[name] = {**res.parsed, "llm": res.meta()}
        return out

    def reconcile(self, doc: dict, extractions: dict[str, dict]) -> list[dict]:
        """Align per-backend experiments; adjudicate disagreements. Returns a
        list of {experiment, per_backend, disagreements, adjudication, unresolved}."""
        names = list(extractions)
        if len(names) == 1:
            n = names[0]
            return [{"experiment": e, "per_backend": {n: e}, "disagreements": {}, "adjudication": None,
                     "unresolved": [], "flags": []} for e in extractions[n]["experiments"]]
        na, nb = names[0], names[1]
        ea, eb = extractions[na]["experiments"], extractions[nb]["experiments"]
        out = []
        for i, j in _align(ea, eb):
            if i is None or j is None:
                exp = ea[i] if i is not None else eb[j]
                only = na if i is not None else nb
                out.append({"experiment": exp, "per_backend": {only: exp}, "disagreements": {
                    "experiment_presence": {"a": "present" if i is not None else "absent",
                                            "b": "present" if j is not None else "absent"}},
                    "adjudication": None, "unresolved": ["experiment_presence"],
                    "flags": [f"experiment extracted by {only} only"]})
                continue
            dis = diff_experiments(ea[i], eb[j])
            adj = None
            flags: list[str] = []
            if dis and self.adjudicator is not None:
                try:
                    adj = adjudicate(self.adjudicator, (na, nb), ea[i], eb[j], dis, doc["text"],
                                     doc.get("title", ""))
                except LLMError as e:
                    flags.append(f"adjudication failed: {e}")
            merged, unresolved = merge_with_adjudication((na, nb), ea[i], eb[j], dis, adj)
            if dis:
                flags.append(f"{na}/{nb} disagreement on {len(dis)} field(s); "
                             f"{len(unresolved)} unresolved")
            out.append({"experiment": merged, "per_backend": {na: ea[i], nb: eb[j]},
                        "disagreements": dis, "adjudication": adj, "unresolved": unresolved,
                        "flags": flags})
        return out

    def build_record(self, cand: Candidate, doc: dict, idx: int, rec_in: dict,
                     screening: dict, extractions: dict) -> dict:
        exp = rec_in["experiment"]
        art = self.verifier.verify_experiment(exp, self.primary, doc["text"])
        bench, scores = primary_scores(exp)
        teachers = [t["name"] for t in exp.get("teacher_models", [])]
        ranking = rank_from_scores(scores) if scores else []
        from .multimodel import norm_name
        scored = {norm_name(t) for t in scores}
        scores_complete = bool(teachers) and all(norm_name(t) in scored for t in teachers)
        flags = list(rec_in["flags"])
        if rec_in["unresolved"]:
            flags.append("unresolved multi-model disagreement: " + ", ".join(rec_in["unresolved"]))
        cls = classify(exp.get("criteria", {}), num_teachers=len(teachers),
                       scores_complete=scores_complete, artifact_verification=art,
                       exclusion_flags=exp.get("exclusion_flags"))
        if rec_in["unresolved"] and cls["status"] in ("gold", "silver", "results_only"):
            # an unresolved multi-model disagreement must be settled by a human;
            # resolved disagreements keep their status but stay in the review queue
            cls = {**cls, "status": "needs_review", "review_flags": cls["review_flags"] + flags,
                   "reasons": cls["reasons"] + flags}
        return {
            "record_id": f"{cand.arxiv_id or cand.key}#{idx + 1}",
            "paper": {"title": doc.get("title") or cand.title, "arxiv_id": cand.arxiv_id,
                      "url": doc.get("source_url") or cand.url, "year": doc.get("year") or cand.year,
                      "doc_hash": doc["doc_hash"], "text_format": doc.get("text_format"),
                      "text_url": doc.get("text_url"), "truncated": doc.get("truncated", False),
                      "discovery_sources": cand.sources},
            "experiment": exp,
            "extractions": rec_in["per_backend"],
            "screening": screening,
            "disagreements": rec_in["disagreements"],
            "adjudication": rec_in["adjudication"],
            "artifact_verification": art,
            "ranking": {"benchmark": bench, "scores": scores, "ranking": ranking,
                        "ranking_string": ranking_string(ranking), "rank_columns": rank_columns(ranking),
                        "scores_complete": scores_complete},
            "classification": cls,
            "review_flags": flags,
            "provenance": {
                "source_url": doc.get("source_url") or cand.url, "arxiv_id": cand.arxiv_id,
                "doc_hash": doc["doc_hash"], "text_format": doc.get("text_format"),
                "llm": {n: (x.get("llm") or {}) for n, x in extractions.items()},
                "adjudicator": (rec_in["adjudication"] or {}).get("llm"),
                "pipeline_version": PIPELINE_VERSION, "git_commit": self.stats["git_commit"],
                "timestamp": _now(),
            },
        }

    # --------------------------------------------------------------- per paper
    def process_candidate(self, cand: Candidate, *, force: bool = False) -> dict:
        """Run retrieval -> screening -> extraction -> verification for one paper.
        Result is persisted under state/papers/<key>.json (resume unit)."""
        pkey = cand.key.replace(":", "_").replace("/", "_")
        state_file = self.state_dir / "papers" / f"{pkey}.json"
        state_file.parent.mkdir(exist_ok=True)
        doc = self.retrieve(cand)
        # resume: reuse per-paper result when the doc hash and prompt versions match
        stamp = cache_key(doc_hash=doc["doc_hash"], screen=SCREENING_VERSION, extract=EXTRACTION_VERSION,
                          artifact=ARTIFACT_VERSION, adjudicate=ADJUDICATION_VERSION,
                          pipeline=PIPELINE_VERSION, backends=sorted(self.clients),
                          models=sorted(c.cfg.model for c in self.clients.values()))
        if state_file.exists() and not force:
            try:
                prev = json.loads(state_file.read_text())
                if prev.get("stamp") == stamp:
                    return prev
            except json.JSONDecodeError:
                pass
        result: dict[str, Any] = {"candidate": cand.to_dict(), "stamp": stamp, "doc_hash": doc["doc_hash"],
                                  "text_format": doc.get("text_format"), "records": [], "stage": "retrieved"}
        if doc.get("text_format") == "abstract_only":
            result["note"] = "full text unavailable; screened on abstract only"
        screening = self.screen(doc)
        result["screening"] = screening
        result["stage"] = "screened"
        rel = {n: s["relevant"] for n, s in screening.items()}
        if all(r == "no" for r in rel.values()):
            result["decision"] = "screened_out"
            result["reason"] = "; ".join(s.get("summary", "") for s in screening.values())
            state_file.write_text(json.dumps(result, indent=1, ensure_ascii=False))
            return result
        result["decision"] = "extract"
        if len(set(rel.values())) > 1:
            result["screening_disagreement"] = rel
        focus = "\n".join(f"- ({n}) {c}" for n, s in screening.items() for c in s.get("candidate_experiments", []))
        extractions = self.extract(doc, focus)
        result["extractions"] = extractions
        result["stage"] = "extracted"
        recs = self.reconcile(doc, extractions)
        for i, r in enumerate(recs):
            result["records"].append(self.build_record(cand, doc, i, r, screening, extractions))
        if not recs:
            result["reason"] = "; ".join(x.get("no_qualifying_experiment_reason", "") for x in extractions.values())
        result["stage"] = "done"
        state_file.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        return result

    # --------------------------------------------------------------------- run
    def fill_metadata(self, cands: list[Candidate]) -> None:
        """Batch title/abstract lookup for arXiv candidates that lack them
        (100 ids per request instead of one full-document retrieval each)."""
        missing = [c for c in cands if c.arxiv_id and not (c.title and c.abstract)]
        if not missing:
            return
        d = getattr(self, "_discovery", None) or Discovery(self.fetcher, self.cache)
        meta = d.arxiv_metadata([c.arxiv_id for c in missing])
        for c in missing:
            m = meta.get(c.arxiv_id) or meta.get(c.arxiv_id.split("v")[0])
            if m:
                c.title, c.abstract, c.year = c.title or m["title"], c.abstract or m["abstract"], c.year or m["year"]
        for c in cands:
            if not c.title and not c.abstract:
                try:
                    self.retrieve(c)   # non-arXiv or still missing: full retrieval
                except Exception as e:
                    log.warning("metadata retrieval failed for %s: %s", c.key, e)

    def run(self, seeds: list[str], *, run_search: bool = True, max_candidates: int | None = None,
            citation_depth: int | None = None, use_github: bool = True, force: bool = False,
            prescreen_only: bool = False) -> list[dict]:
        cands = self.discover(seeds, run_search=run_search, citation_depth=citation_depth, use_github=use_github)
        limit = max_candidates or self.settings.max_candidates
        # prescreen everything (cheap), full-text only what passes, seeds always
        kept: list[tuple[tuple, Candidate]] = []
        pres_log = []
        self.fill_metadata(cands)
        n_llm = 0
        for i, c in enumerate(cands, 1):
            try:
                p = self.prescreen(c)
            except LLMError as e:
                p = {"relevant": "unclear", "reason": f"prescreen failed: {e}"}
            if "llm" in p:
                n_llm += 1
                if n_llm % 100 == 0:
                    log.info("prescreen %d/%d (%d LLM calls)", i, len(cands), n_llm)
            pres_log.append({"key": c.key, "title": c.title, "year": c.year, "hops": c.hops,
                             "sources": c.sources[:3], **{k: v for k, v in p.items() if k != "llm"}})
            if p["relevant"] != "no":
                # priority: seeds, then prescreen "yes" by confidence, then "unclear";
                # arXiv candidates before bare DOIs (full text is far more likely)
                prio = ("seed" not in c.sources, p["relevant"] != "yes",
                        -(p.get("confidence") or 0.0), c.arxiv_id is None, c.hops, c.key)
                kept.append((prio, c))
        (self.state_dir / "prescreen.json").write_text(json.dumps(pres_log, indent=1, ensure_ascii=False))
        self.stats["papers_prescreened"] = len(pres_log)
        counts = {"yes": 0, "unclear": 0, "no": 0, "gated": 0}
        for p in pres_log:
            counts["gated" if p.get("gated") else p["relevant"]] += 1
        self.stats["prescreen_counts"] = counts
        kept = [c for _, c in sorted(kept, key=lambda t: t[0])][:limit]
        n_pos = sum(1 for p in pres_log if p["relevant"] != "no")
        if prescreen_only:
            self.stats.update(papers_screened=0, generated_at=_now())
            (self.state_dir / "stats.json").write_text(json.dumps(self.stats, indent=1))
            log.info("prescreen only: %s; %d would go to full-text screening", counts, min(n_pos, limit))
            return []
        if len(kept) < n_pos:
            log.warning("max_candidates=%d truncates %d prescreen-positive papers", limit, n_pos)
            self.stats["papers_truncated_by_cap"] = n_pos - len(kept)
        records, rejected = [], []
        for i, c in enumerate(kept, 1):
            log.info("[%d/%d] %s %s", i, len(kept), c.key, (c.title or "")[:70])
            try:
                res = self.process_candidate(c, force=force)
                log.info("[%d/%d] -> %s, %d record(s)", i, len(kept), res.get("decision"), len(res["records"]))
            except Exception as e:
                log.exception("paper %s failed: %s", c.key, e)
                rejected.append({"arxiv_id": c.arxiv_id, "url": c.url, "title": c.title, "reason": f"ERROR: {e}"})
                continue
            if res.get("decision") == "screened_out" or (res.get("decision") == "extract" and not res["records"]):
                rejected.append({"arxiv_id": c.arxiv_id, "url": c.url, "title": c.title,
                                 "reason": res.get("reason", "")})
            records += res["records"]
        self.stats.update(papers_screened=len(kept), papers_extracted=sum(
            1 for c in kept if (self.state_dir / "papers" / f"{c.key.replace(':', '_').replace('/', '_')}.json").exists()),
            rejected_papers=rejected, generated_at=_now())
        self.save_records(records)
        return records

    def upsert_records(self, paper_key: str, records: list[dict]) -> Path:
        """Replace this paper's records in state/records.jsonl (used by `paper`)."""
        kept = [r for r in self.load_records() if r["paper"].get("arxiv_id") != paper_key
                and r["paper"].get("url") != paper_key]
        self.stats.setdefault("generated_at", _now())
        return self.save_records(kept + records)

    def save_records(self, records: list[dict]) -> Path:
        p = self.state_dir / "records.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        (self.state_dir / "stats.json").write_text(json.dumps(self.stats, indent=1, ensure_ascii=False))
        return p

    def load_records(self) -> list[dict]:
        p = self.state_dir / "records.jsonl"
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def export(self, records: list[dict] | None = None) -> dict[str, Path]:
        from .export import export_all
        if records is None:
            records = self.load_records()
        sp = self.state_dir / "stats.json"
        stats = json.loads(sp.read_text()) if sp.exists() else dict(self.stats)
        stats.setdefault("generated_at", _now())
        return export_all(records, self.settings.out_dir, stats)
