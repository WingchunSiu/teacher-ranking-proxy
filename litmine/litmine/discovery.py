"""Candidate discovery across arXiv, OpenAlex, Semantic Scholar (optional,
rate-limited without a key), Hugging Face and GitHub, plus citation/reference
expansion from seed papers via OpenAlex (and S2 when available).

Design (revised after the first sweep found only 237 candidates):
  * discovery is BROAD and cheap -- topic-level arXiv queries with paging,
    known agentic-SFT-data papers as seeds (resolved by title), HF trajectory
    datasets, and depth-2 citation expansion;
  * precision comes later from a free keyword gate (`keyword_gate`) and the
    LLM prescreen, not from narrow search phrases.

Every source query is cached; discovery is deterministic given cached results.
Candidates are keyed by arXiv id when one is known, else by URL.
"""
from __future__ import annotations

import itertools
import logging
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable

from .cache import Cache, cache_key
from .retrieval import Fetcher, norm_arxiv_id

log = logging.getLogger("litmine.discovery")

# Concept groups (PIPELINE.md §4.1). Queries are built from cross-products of
# one term per group so discovery never hinges on a single keyword phrase.
TEACHER_TERMS = ["teacher trajectories", "multiple teachers", "teacher model ablation",
                 "teacher selection", "teacher-generated data", "trajectory distillation",
                 "knowledge distillation"]
TRAINING_TERMS = ["SFT", "supervised fine-tuning", "trajectory dataset"]
AGENT_TERMS = ["agent", "agentic", "terminal", "SWE", "software engineering agent",
               "tool use", "interactive environment"]

# Broad topic queries (arXiv boolean syntax, abstract field). Each is paged up
# to BROAD_QUERY_RESULTS results, newest first. The target ranking usually sits
# in an ablation table of a paper whose abstract never says "teacher", so these
# deliberately cover "agentic SFT data" as a whole.
_AGENT = ('(abs:agent OR abs:agents OR abs:agentic OR abs:"tool use" OR abs:"tool-use" '
          'OR abs:"function calling" OR abs:"tool calling")')
_TRAIN = ('(abs:"supervised fine-tuning" OR abs:SFT OR abs:"fine-tuning" OR abs:"fine-tune" '
          'OR abs:finetuning OR abs:finetune OR abs:distillation OR abs:distill OR abs:distilled)')
_DATA = ('(abs:trajectories OR abs:trajectory OR abs:"training data" OR abs:"synthetic data" '
         'OR abs:"data recipe" OR abs:"data curation" OR abs:"data synthesis" OR abs:traces '
         'OR abs:rollouts OR abs:demonstrations)')
_BENCH = ('(abs:"SWE-bench" OR abs:"Terminal-Bench" OR abs:"terminal-bench" OR abs:"tau-bench" '
          'OR abs:BFCL OR abs:WebArena OR abs:OSWorld OR abs:"Mind2Web" OR abs:AgentBench '
          'OR abs:GAIA OR abs:"SWE-Gym" OR abs:"ToolBench" OR abs:"AppWorld")')
_DOMAIN = ('(abs:"software engineering" OR abs:"code agent" OR abs:"coding agent" OR abs:"SWE agent" '
           'OR abs:"terminal" OR abs:"command line" OR abs:"web agent" OR abs:"GUI agent" '
           'OR abs:"computer use" OR abs:"tool-augmented" OR abs:"tool-integrated")')
BROAD_ARXIV_QUERIES = [
    f"{_AGENT} AND {_TRAIN} AND {_DATA}",
    f"{_BENCH} AND {_TRAIN}",
    f"{_DOMAIN} AND {_TRAIN} AND {_DATA}",
    f"{_AGENT} AND (abs:teacher OR abs:teachers) AND {_TRAIN}",
    f"{_AGENT} AND {_TRAIN} AND (abs:ablation OR abs:ablations) AND {_DATA}",
    f'({_AGENT} OR {_DOMAIN}) AND (abs:"open-source" OR abs:"open source" OR abs:release OR abs:dataset) '
    f'AND (abs:trajectories OR abs:trajectory)',
]
BROAD_QUERY_RESULTS = int(os.environ.get("LITMINE_BROAD_RESULTS", "1500"))
ARXIV_MIN_YEAR = int(os.environ.get("LITMINE_MIN_YEAR", "2023"))

# Known agentic-SFT-data papers, resolved to arXiv ids by exact-title search
# (ids are not hard-coded so a typo cannot silently seed the wrong paper).
# Their references/citations are the densest neighbourhood for the target.
SEED_TITLES = [
    "OpenThoughts-Agent: Data Recipes for Agentic Models",
    "SWE-smith: Scaling Data for Software Engineering Agents",
    "Training Software Engineering Agents and Verifiers with SWE-Gym",
    "R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents",
    "AgentTuning: Enabling Generalized Agent Abilities for LLMs",
    "Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models",
    "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets",
    "ToolACE: Winning the Points of LLM Function Calling",
    "FireAct: Toward Language Agent Fine-tuning",
    "AgentOhana: Design Unified Data and Training Pipeline for Effective Agent Learning",
    "Skywork-SWE: Unveiling Data Scaling Laws for Software Engineering in LLMs",
    "SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling",
    "Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents",
    "SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolving",
    "Nemotron-Terminal: Scaling Terminal Task Environments for Agentic Data Generation",
    "SWE-Lego: Pushing the Limits of Supervised Fine-tuning for Software Issue Resolving",
    "CoderForge: Scaling Coding Agent Trajectories",
    "Nex-N1: Agentic Models Trained via a Unified Ecosystem for Large-Scale Environment Construction",
    "OpenHands: An Open Platform for AI Software Developers as Generalist Agents",
    "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution",
    "AgentInstruct: Toward Generative Teaching with Agentic Flows",
    "Terminal-Lego",
    "xLAM: A Family of Large Action Models to Empower AI Agent Systems",
    "AgentGym: Evolving Large Language Model-based Agents across Diverse Environments",
    "Executable Code Actions Elicit Better LLM Agents",
    "Lemur: Harmonizing Natural Language and Code for Language Agents",
    "SWE-Synth: Synthesizing Verifiable Bug-Fix Data to Enable LLMs in Resolving Real-World Bugs",
    "Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents",
    "OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis",
    "Synatra: Turning Indirect Knowledge into Direct Demonstrations for Digital Agents at Scale",
    "AgentBank: Towards Generalized LLM Agents via Fine-Tuning on 50000+ Interaction Trajectories",
    "TerminalBench",
]

DEFAULT_HF_QUERIES = ["teacher trajectories", "agent trajectories sft", "terminal agent",
                      "swe agent trajectories", "tool use trajectories", "trajectory distillation",
                      "agent traces", "swe-bench trajectories", "terminal-bench trajectories",
                      "gui agent trajectories", "web agent trajectories", "agentic sft data",
                      "distilled trajectories", "agent distillation",
                      "trajectories", "traj", "agentic", "swe sft", "openhands trajectories",
                      "codeact", "tool calling sft", "function calling sft", "agent sft",
                      "terminal traces", "swe-gym", "swe-smith", "nemotron terminal",
                      "computer use trajectories", "browser agent trajectories", "mcp trajectories"]
# HF orgs that publish agentic trajectory datasets; every dataset with an
# arxiv tag becomes a candidate.
DEFAULT_HF_AUTHORS = ["open-thoughts", "Lego-X", "SWE-Gym", "SWE-bench", "R2E-Gym", "nebius",
                      "Salesforce", "THUDM", "OpenHands", "all-hands", "xingyaoww", "nvidia",
                      "Kwai-Kolors", "internlm", "AgentGym", "Skywork", "Team-ACE", "TIGER-Lab",
                      "OS-Copilot", "OpenGVLab", "Nex-AGI", "allenai", "togethercomputer",
                      "ScaleAI", "PrimeIntellect", "laion", "penfever", "SWE-Lego", "SWE-agent"]
DEFAULT_GITHUB_QUERIES = ["teacher trajectories SFT agent", "agent trajectory distillation",
                          "terminal agent SFT trajectories", "SWE agent SFT trajectories dataset",
                          "agent trajectories dataset fine-tuning"]

# ---- free keyword gate ----------------------------------------------------
# A candidate must look agentic AND look like training-data work before any
# LLM call. Deliberately loose; precision comes from the LLM prescreen.
_AGENT_RE = re.compile(
    r"\bagent(s|ic)?\b|tool[- ]?(use|using|calling|learning|augmented|integrated)|function[- ]calling"
    r"|\bterminal\b|command[- ]line|\bshell\b|swe[- ]?bench|software engineering|issue resol"
    r"|web (navigation|browsing)|\bgui\b|computer[- ]use|\bcodeact\b|interactive environment"
    r"|multi[- ]turn|\benvironments?\b|\brollouts?\b|\bmcp\b", re.I)
_TRAIN_RE = re.compile(
    r"fine[- ]?tun|\bsft\b|distill|trajector|training data|synthetic data|imitation|behavio(u)?r cloning"
    r"|rejection sampling|\btraces?\b|demonstrations|data (recipe|curation|synthesis|pipeline|engine)"
    r"|supervised|post[- ]training|\btrain(ed|ing)?\b", re.I)


def keyword_gate(title: str, abstract: str) -> bool:
    text = f"{title or ''} {abstract or ''}"
    return bool(_AGENT_RE.search(text)) and bool(_TRAIN_RE.search(text))


@dataclass
class Candidate:
    key: str                      # arxiv:<id> | url:<url>
    arxiv_id: str | None
    url: str | None
    title: str = ""
    abstract: str = ""
    year: int | None = None
    sources: list[str] = field(default_factory=list)   # provenance of discovery
    tags: list[str] = field(default_factory=list)
    hops: int = 0

    def merge(self, other: "Candidate") -> None:
        self.title = self.title or other.title
        self.abstract = self.abstract or other.abstract
        self.year = self.year or other.year
        self.url = self.url or other.url
        for s in other.sources:
            if s not in self.sources:
                self.sources.append(s)
        for t in other.tags:
            if t not in self.tags:
                self.tags.append(t)
        self.hops = min(self.hops, other.hops)

    def to_dict(self) -> dict:
        return {"key": self.key, "arxiv_id": self.arxiv_id, "url": self.url,
                "title": self.title, "abstract": self.abstract, "year": self.year,
                "sources": self.sources, "tags": self.tags, "hops": self.hops}


def build_arxiv_queries(max_queries: int | None = None) -> list[str]:
    qs = []
    for t, tr, a in itertools.product(TEACHER_TERMS, TRAINING_TERMS, AGENT_TERMS):
        qs.append(f'all:"{t}" AND all:"{tr}" AND all:"{a}"')
    # a few broader phrasings without the training group
    for t, a in itertools.product(["teacher trajectories", "trajectory distillation",
                                   "teacher-generated data"], ["agent", "terminal", "SWE"]):
        qs.append(f'all:"{t}" AND all:"{a}"')
    return qs[:max_queries] if max_queries else qs


def build_free_text_queries() -> list[str]:
    """Natural-language queries for OpenAlex / S2 (no boolean syntax)."""
    return [
        "teacher trajectories supervised fine-tuning agent",
        "multiple teachers trajectory distillation agent SFT",
        "teacher model ablation agentic supervised fine-tuning",
        "teacher selection trajectory data software engineering agent",
        "terminal agent SFT teacher-generated trajectories",
        "tool use agent knowledge distillation trajectories teachers",
        "interactive environment trajectories distillation student model teachers",
        "what makes trajectories effective for training agents",
        "SWE agent trajectory dataset teacher models fine-tuning comparison",
        "comparing teacher models for agent distillation downstream performance",
        "which teacher model generates the best training trajectories for agents",
        "student model fine-tuned on trajectories from different teacher LLMs",
        "agentic supervised fine-tuning data source ablation teacher",
        "GUI agent trajectories distilled from multiple teacher models",
        "web agent trajectory synthesis teacher model comparison fine-tuning",
        "tool-calling agent SFT data generated by different LLMs comparison",
        "terminal-bench SFT trajectories teacher",
        "SWE-bench agent SFT trajectories generated by Claude GPT comparison",
        "data recipes for agentic models supervised fine-tuning trajectories",
        "scaling data for software engineering agents trajectories fine-tuning",
        "agent trajectory dataset release open-source fine-tuning ablation",
    ]


def _parse_arxiv_feed(body: bytes) -> list[dict]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = []
    for e in ET.fromstring(body).findall("a:entry", ns):
        aid = norm_arxiv_id(e.findtext("a:id", "", ns))
        if not aid:
            continue
        pub = e.findtext("a:published", "", ns) or ""
        entries.append({"arxiv_id": aid,
                        "title": " ".join((e.findtext("a:title", "", ns) or "").split()),
                        "abstract": " ".join((e.findtext("a:summary", "", ns) or "").split()),
                        "year": int(pub[:4]) if pub[:4].isdigit() else None})
    return entries


class Discovery:
    def __init__(self, fetcher: Fetcher, cache: Cache, per_query: int = 25):
        self.f = fetcher
        self.cache = cache
        self.per_query = per_query
        self.candidates: dict[str, Candidate] = {}
        self.stats: dict[str, int] = {}
        self.seed_resolution: dict[str, str | None] = {}

    # ---- helpers ------------------------------------------------------------
    def _add(self, c: Candidate) -> None:
        if c.key in self.candidates:
            self.candidates[c.key].merge(c)
        else:
            self.candidates[c.key] = c
        self.stats[c.sources[0].split(":")[0]] = self.stats.get(c.sources[0].split(":")[0], 0) + 1

    @staticmethod
    def _oa(params: dict) -> str:
        params = dict(params, mailto=os.environ.get("OPENALEX_MAILTO", "litmine@example.org"))
        return "https://api.openalex.org/works?" + urllib.parse.urlencode(params)

    def _cached(self, stage: str, **parts):
        key = cache_key(**parts)
        hit = self.cache.get(stage, key)
        return key, hit

    def _entries_to_candidates(self, entries: list[dict], source: str, hops: int = 0) -> list[Candidate]:
        out = []
        for e in entries:
            c = Candidate(key=f"arxiv:{e['arxiv_id']}", arxiv_id=e["arxiv_id"],
                          url=f"https://arxiv.org/abs/{e['arxiv_id']}", title=e["title"],
                          abstract=e["abstract"], year=e["year"], sources=[source], hops=hops)
            self._add(c)
            out.append(c)
        return out

    # ---- arXiv -----------------------------------------------------------------
    def arxiv_search(self, query: str, max_results: int | None = None) -> list[Candidate]:
        n = max_results or self.per_query
        key, hit = self._cached("search_arxiv", query=query, n=n)
        if hit is None:
            url = ("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                {"search_query": query, "max_results": n, "sortBy": "relevance"}))
            status, body, _ = self.f.get(url)
            entries = _parse_arxiv_feed(body) if status == 200 else []
            hit = self.cache.put("search_arxiv", key, {"query": query, "status": status,
                                                       "entries": entries})
        return self._entries_to_candidates(hit["entries"], f"arxiv:{query}")

    def arxiv_search_paged(self, query: str, total: int, page: int = 200,
                           min_year: int | None = None) -> list[Candidate]:
        """Page through up to `total` results (newest first). Stops early when a
        page is empty or every entry is older than `min_year`."""
        out: list[Candidate] = []
        min_year = ARXIV_MIN_YEAR if min_year is None else min_year
        for start in range(0, total, page):
            key, hit = self._cached("search_arxiv_paged", query=query, start=start, page=page)
            if hit is None:
                url = ("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                    {"search_query": query, "start": start, "max_results": page,
                     "sortBy": "submittedDate", "sortOrder": "descending"}))
                status, body, _ = self.f.get(url)
                entries = []
                if status == 200:
                    try:
                        entries = _parse_arxiv_feed(body)
                    except ET.ParseError as e:
                        log.warning("arxiv page parse failed (%s): %s", query[:60], e)
                hit = {"query": query, "status": status, "entries": entries}
                if status == 200:
                    self.cache.put("search_arxiv_paged", key, hit)
            entries = hit["entries"]
            if not entries:
                break
            kept = [e for e in entries if not e["year"] or e["year"] >= min_year]
            out += self._entries_to_candidates(kept, f"arxiv_broad:{query[:80]}")
            if len(kept) < len(entries) and all((e["year"] or 0) < min_year for e in entries[-20:]):
                break
        return out

    def arxiv_title_lookup(self, title: str) -> Candidate | None:
        """Resolve a paper title to an arXiv id (exact-ish title match)."""
        short = re.sub(r"[:\-–].*$", "", title).strip() if len(title) > 40 else title
        phrase = title.replace('"', "")
        key, hit = self._cached("arxiv_title", title=title)
        if hit is None:
            entries = []
            for q in (f'ti:"{phrase}"', f'ti:"{short}"'):
                url = ("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                    {"search_query": q, "max_results": 10}))
                status, body, _ = self.f.get(url)
                if status == 200:
                    try:
                        entries = _parse_arxiv_feed(body)
                    except ET.ParseError:
                        entries = []
                if entries:
                    break
            hit = self.cache.put("arxiv_title", key, {"entries": entries})
        want = _norm_title(title)
        for e in hit["entries"]:
            got = _norm_title(e["title"])
            if got == want or got.startswith(want) or want.startswith(got) or _norm_title(short) in got:
                return self._entries_to_candidates([e], "seed")[0]
        return None

    def arxiv_metadata(self, ids: list[str]) -> dict[str, dict]:
        """Batch title/abstract lookup via id_list (100 per request)."""
        out: dict[str, dict] = {}
        ids = [i for i in ids if i]
        for i in range(0, len(ids), 100):
            batch = sorted(set(ids[i:i + 100]))
            key, hit = self._cached("arxiv_meta_batch", ids=batch)
            if hit is None:
                url = ("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                    {"id_list": ",".join(batch), "max_results": len(batch)}))
                status, body, _ = self.f.get(url)
                entries = []
                if status == 200:
                    try:
                        entries = _parse_arxiv_feed(body)
                    except ET.ParseError:
                        entries = []
                hit = {"entries": entries}
                if status == 200:
                    self.cache.put("arxiv_meta_batch", key, hit)
            for e in hit["entries"]:
                out[e["arxiv_id"]] = e
                base = e["arxiv_id"].split("v")[0]
                out.setdefault(base, e)
        return out

    # ---- OpenAlex ------------------------------------------------------------
    def _openalex_work_to_candidate(self, w: dict, source: str, hops: int = 0) -> Candidate | None:
        arxiv = None
        for loc in (w.get("locations") or []) + [w.get("primary_location") or {}]:
            for u in (loc.get("landing_page_url"), loc.get("pdf_url")):
                if u and "arxiv.org" in u:
                    arxiv = norm_arxiv_id(u)
                    break
            if arxiv:
                break
        ids = w.get("ids") or {}
        url = ids.get("doi") or (w.get("primary_location") or {}).get("landing_page_url") or w.get("id")
        abstract = ""
        inv = w.get("abstract_inverted_index")
        if inv:
            pos = sorted((p, t) for t, ps in inv.items() for p in ps)
            abstract = " ".join(t for _, t in pos)
        if not arxiv and not url:
            return None
        key = f"arxiv:{arxiv}" if arxiv else f"url:{url}"
        return Candidate(key=key, arxiv_id=arxiv, url=f"https://arxiv.org/abs/{arxiv}" if arxiv else url,
                         title=w.get("title") or w.get("display_name") or "", abstract=abstract,
                         year=w.get("publication_year"), sources=[source],
                         tags=[f"openalex:{w.get('id')}"], hops=hops)

    def openalex_search(self, query: str, max_results: int | None = None) -> list[Candidate]:
        n = max_results or self.per_query
        key, hit = self._cached("search_openalex", query=query, n=n)
        if hit is None:
            url = self._oa(
                {"search": query, "per-page": n,
                 "select": "id,title,display_name,publication_year,ids,primary_location,locations,abstract_inverted_index,referenced_works,cited_by_api_url"})
            try:
                data = self.f.get_json(url)
                hit = {"query": query, "results": data.get("results", [])}
            except Exception as e:
                log.warning("openalex search failed for %r: %s", query, e)
                hit = {"query": query, "results": [], "error": str(e)}
            self.cache.put("search_openalex", key, hit)
        out = []
        for w in hit["results"]:
            c = self._openalex_work_to_candidate(w, f"openalex:{query}")
            if c:
                self._add(c)
                out.append(c)
        return out

    def openalex_lookup(self, arxiv_id: str | None = None, title: str | None = None) -> dict | None:
        """Find the OpenAlex work for a paper (by arXiv id via locations, else title)."""
        key, hit = self._cached("openalex_lookup", arxiv_id=arxiv_id, title=title)
        if hit is not None:
            return hit.get("work")
        work = None
        sel = "id,title,display_name,publication_year,ids,primary_location,locations,abstract_inverted_index,referenced_works,cited_by_api_url,cited_by_count"
        tries = []
        if arxiv_id:
            tries.append(self._oa(
                {"filter": f"locations.landing_page_url:https://arxiv.org/abs/{arxiv_id}",
                 "select": sel, "per-page": 5}))
        if title:
            tries.append(self._oa(
                {"search": re.sub(r"[*?\"]", " ", title), "select": sel, "per-page": 5}))
        for url in tries:
            try:
                res = self.f.get_json(url).get("results", [])
            except Exception as e:
                log.warning("openalex lookup failed: %s", e)
                continue
            for w in res:
                c = self._openalex_work_to_candidate(w, "lookup")
                if arxiv_id and c and c.arxiv_id == arxiv_id:
                    work = w
                    break
                if title and c and _norm_title(c.title) == _norm_title(title):
                    work = w
                    break
            if work:
                break
        self.cache.put("openalex_lookup", key, {"work": work})
        return work

    def openalex_expand(self, work: dict, hops: int, max_results: int = 1000) -> list[Candidate]:
        """References + citations of an OpenAlex work (citations paged by cursor)."""
        out: list[Candidate] = []
        wid = work.get("id")
        sel = "id,title,display_name,publication_year,ids,primary_location,locations,abstract_inverted_index"
        # citations (cursor paging, 200 per page)
        cursor, got = "*", 0
        while cursor and got < max_results:
            key, hit = self._cached("openalex_cites", wid=wid, n=200, cursor=cursor)
            if hit is None:
                url = self._oa(
                    {"filter": f"cites:{wid.rsplit('/', 1)[-1]}", "per-page": 200, "cursor": cursor,
                     "select": sel})
                try:
                    data = self.f.get_json(url)
                    hit = {"results": data.get("results", []),
                           "next_cursor": (data.get("meta") or {}).get("next_cursor")}
                except Exception as e:
                    hit = {"results": [], "error": str(e), "next_cursor": None}
                self.cache.put("openalex_cites", key, hit)
            for w in hit["results"]:
                c = self._openalex_work_to_candidate(w, f"cites:{wid}", hops)
                if c:
                    self._add(c)
                    out.append(c)
            got += len(hit["results"])
            cursor = hit.get("next_cursor") if hit["results"] else None
        # references (batched by id filter)
        refs = work.get("referenced_works") or []
        for i in range(0, len(refs), 50):
            batch = refs[i:i + 50]
            key, hit = self._cached("openalex_refs", ids=batch)
            if hit is None:
                url = self._oa(
                    {"filter": "openalex:" + "|".join(r.rsplit("/", 1)[-1] for r in batch),
                     "per-page": 50, "select": sel})
                try:
                    hit = {"results": self.f.get_json(url).get("results", [])}
                except Exception as e:
                    hit = {"results": [], "error": str(e)}
                self.cache.put("openalex_refs", key, hit)
            for w in hit["results"]:
                c = self._openalex_work_to_candidate(w, f"referenced_by:{wid}", hops)
                if c:
                    self._add(c)
                    out.append(c)
        return out

    # ---- Semantic Scholar (optional) ------------------------------------------
    def s2_expand(self, arxiv_id: str, hops: int) -> list[Candidate]:
        """Citations + references via Semantic Scholar. Skipped gracefully on 429
        (unauthenticated quota); set S2_API_KEY to enable reliably."""
        out: list[Candidate] = []
        hdr = {}
        if os.environ.get("S2_API_KEY"):
            hdr["x-api-key"] = os.environ["S2_API_KEY"]
        for kind in ("citations", "references"):
            key, hit = self._cached("s2_graph", arxiv_id=arxiv_id, kind=kind)
            if hit is None:
                url = (f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}/{kind}"
                       "?fields=title,abstract,year,externalIds,url&limit=500")
                try:
                    status, body, _ = self.f.get(url, headers=hdr)
                    import json
                    hit = {"status": status, "data": json.loads(body).get("data", []) if status == 200 else []}
                except Exception as e:
                    hit = {"status": 0, "data": [], "error": str(e)}
                if hit["status"] == 200:
                    self.cache.put("s2_graph", key, hit)
                else:
                    log.info("S2 %s for %s unavailable (status %s)", kind, arxiv_id, hit["status"])
            for row in hit["data"]:
                p = row.get("citingPaper") or row.get("citedPaper") or {}
                ext = p.get("externalIds") or {}
                aid = ext.get("ArXiv")
                url = f"https://arxiv.org/abs/{aid}" if aid else p.get("url")
                if not url:
                    continue
                c = Candidate(key=f"arxiv:{aid}" if aid else f"url:{url}", arxiv_id=aid, url=url,
                              title=p.get("title") or "", abstract=p.get("abstract") or "",
                              year=p.get("year"), sources=[f"s2_{kind}:{arxiv_id}"], hops=hops)
                self._add(c)
                out.append(c)
        return out

    # ---- Hugging Face ----------------------------------------------------------
    def _hf_datasets_to_candidates(self, datasets: list[dict], source: str) -> list[Candidate]:
        out = []
        for d in datasets:
            tags = d.get("tags") or []
            arxivs = [t.split(":", 1)[1] for t in tags if t.startswith("arxiv:")]
            for aid in arxivs:
                aid = norm_arxiv_id(aid)
                if not aid:
                    continue
                c = Candidate(key=f"arxiv:{aid}", arxiv_id=aid, url=f"https://arxiv.org/abs/{aid}",
                              sources=[source], tags=[f"hf_dataset:{d.get('id')}"])
                self._add(c)
                out.append(c)
        return out

    def hf_search(self, query: str, limit: int = 100) -> list[Candidate]:
        """Datasets whose card cites an arXiv paper become paper candidates; the
        dataset id is kept as a tag so artifact verification can use it."""
        key, hit = self._cached("search_hf", query=query, limit=limit)
        if hit is None:
            url = "https://huggingface.co/api/datasets?" + urllib.parse.urlencode(
                {"search": query, "limit": limit, "full": "true"})
            try:
                hit = {"results": self.f.get_json(url)}
            except Exception as e:
                hit = {"results": [], "error": str(e)}
            self.cache.put("search_hf", key, hit)
        return self._hf_datasets_to_candidates(hit["results"], f"hf:{query}")

    def hf_author(self, author: str, limit: int = 500) -> list[Candidate]:
        """All datasets of an HF org/user that carry an arxiv tag."""
        key, hit = self._cached("hf_author", author=author, limit=limit)
        if hit is None:
            url = "https://huggingface.co/api/datasets?" + urllib.parse.urlencode(
                {"author": author, "limit": limit, "full": "true"})
            try:
                hit = {"results": self.f.get_json(url)}
            except Exception as e:
                hit = {"results": [], "error": str(e)}
            self.cache.put("hf_author", key, hit)
        return self._hf_datasets_to_candidates(hit["results"], f"hf_author:{author}")

    # ---- GitHub ----------------------------------------------------------------
    def github_search(self, query: str, limit: int = 20) -> list[Candidate]:
        """Repos whose description/README mention an arXiv id -> paper candidate."""
        key, hit = self._cached("search_github", query=query, limit=limit)
        if hit is None:
            hdr = {"Accept": "application/vnd.github+json"}
            if os.environ.get("GITHUB_TOKEN"):
                hdr["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
            url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
                {"q": query, "per_page": limit})
            try:
                hit = {"items": self.f.get_json(url, headers=hdr).get("items", [])}
            except Exception as e:
                hit = {"items": [], "error": str(e)}
            self.cache.put("search_github", key, hit)
        out = []
        for it in hit["items"]:
            text = f"{it.get('description') or ''} {it.get('homepage') or ''}"
            aid = norm_arxiv_id(text) if "arxiv" in text.lower() else None
            if aid:
                c = Candidate(key=f"arxiv:{aid}", arxiv_id=aid, url=f"https://arxiv.org/abs/{aid}",
                              sources=[f"github:{query}"], tags=[f"github_repo:{it.get('full_name')}"])
            else:
                continue
            self._add(c)
            out.append(c)
        return out

    # ---- seeds & expansion -------------------------------------------------
    def add_seed(self, seed: str) -> Candidate:
        aid = norm_arxiv_id(seed)
        if aid:
            c = Candidate(key=f"arxiv:{aid}", arxiv_id=aid, url=f"https://arxiv.org/abs/{aid}",
                          sources=["seed"])
        else:
            c = Candidate(key=f"url:{seed}", arxiv_id=None, url=seed, sources=["seed"])
        self._add(c)
        return c

    def resolve_seed_titles(self, titles: Iterable[str]) -> list[Candidate]:
        """Title-listed seeds become `seed_title` candidates: they are expanded
        like seeds but still go through prescreen/screening (a known dataset
        paper is not automatically a teacher-ranking paper)."""
        out = []
        for t in titles:
            c = self.arxiv_title_lookup(t)
            self.seed_resolution[t] = c.arxiv_id if c else None
            if c is None:
                log.info("seed title not resolved on arXiv: %r", t)
                continue
            c.sources = [s for s in c.sources if s != "seed"] + ["seed_title"]
            out.append(c)
        return out

    def expand_citations(self, seeds: Iterable[Candidate], depth: int = 1,
                         max_frontier: int = 400) -> None:
        """Depth-limited expansion. Beyond hop 1 only candidates passing the
        keyword gate are expanded further (bounded by `max_frontier` per hop,
        ranked by how often they were reached), so hop 2 stays on-topic."""
        frontier = list(seeds)
        seen: set[str] = set()
        for hop in range(1, depth + 1):
            nxt: dict[str, Candidate] = {}
            if hop > 1:
                gated = [c for c in frontier if keyword_gate(c.title, c.abstract) and c.key not in seen]
                gated.sort(key=lambda c: (-len(c.sources), c.key))
                if len(gated) > max_frontier:
                    log.info("hop %d frontier truncated %d -> %d", hop, len(gated), max_frontier)
                frontier = gated[:max_frontier]
            for c in frontier:
                if c.key in seen:
                    continue
                seen.add(c.key)
                work = self.openalex_lookup(arxiv_id=c.arxiv_id, title=c.title or None)
                if work:
                    for n in self.openalex_expand(work, hops=hop):
                        nxt.setdefault(n.key, n)
                # S2 without a key answers 429 at 1.5 s per request: only worth it for hop 1
                if c.arxiv_id and (hop == 1 or os.environ.get("S2_API_KEY")):
                    for n in self.s2_expand(c.arxiv_id, hops=hop):
                        nxt.setdefault(n.key, n)
            frontier = list(nxt.values())

    def run(self, seeds: list[str], *, arxiv_queries: list[str] | None = None,
            free_queries: list[str] | None = None, hf_queries: list[str] | None = None,
            github_queries: list[str] | None = None, citation_depth: int = 1,
            use_github: bool = True, broad_queries: list[str] | None = None,
            seed_titles: list[str] | None = None, hf_authors: list[str] | None = None) -> list[Candidate]:
        # full-search mode (no explicit query lists) turns every source on;
        # callers that pass explicit lists get exactly what they asked for.
        full = arxiv_queries is None
        seed_cands = [self.add_seed(s) for s in seeds]
        seed_cands += self.resolve_seed_titles(
            seed_titles if seed_titles is not None else (SEED_TITLES if full else []))
        for q in (broad_queries if broad_queries is not None else (BROAD_ARXIV_QUERIES if full else [])):
            self.arxiv_search_paged(q, BROAD_QUERY_RESULTS)
        for q in (arxiv_queries if arxiv_queries is not None else build_arxiv_queries()):
            self.arxiv_search(q)
        for q in (free_queries if free_queries is not None else build_free_text_queries()):
            self.openalex_search(q)
        for q in (hf_queries if hf_queries is not None else DEFAULT_HF_QUERIES):
            self.hf_search(q)
        for a in (hf_authors if hf_authors is not None else (DEFAULT_HF_AUTHORS if full else [])):
            self.hf_author(a)
        if use_github:
            for q in (github_queries if github_queries is not None else DEFAULT_GITHUB_QUERIES):
                self.github_search(q)
        if citation_depth > 0:
            self.expand_citations(seed_cands, depth=citation_depth)
        return sorted(self.candidates.values(), key=lambda c: (c.hops, c.key))


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()
