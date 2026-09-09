"""Paper / webpage retrieval and text extraction.

Priority for arXiv papers: arXiv HTML (tables survive as pipe-separated rows)
-> PDF via pypdf -> abstract only. Generic URLs go through the same HTML
converter. Everything is cached by URL; the returned text carries a sha256
that downstream stages use as the source-document hash.
"""
from __future__ import annotations

import html
import io
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

from .cache import Cache, cache_key, sha256_text

log = logging.getLogger("litmine.retrieval")

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def norm_arxiv_id(s: str) -> str | None:
    m = ARXIV_ID_RE.search(s or "")
    return m.group(1) if m else None


class _TextHTML(HTMLParser):
    """HTML -> readable text. Keeps headings, paragraphs, list items and
    tables (cells joined by ' | ', rows on separate lines). Drops scripts,
    styles, nav and MathML annotations' duplicates."""
    BLOCK = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
             "li", "tr", "br", "figcaption", "blockquote", "pre", "header", "footer",
             "table", "caption", "dd", "dt"}
    SKIP = {"script", "style", "noscript", "nav", "svg", "annotation", "annotation-xml",
            "button", "form", "select", "option"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._cell = 0
        self._row = 0
        self._stack: list[str] = []
        self._math = 0
        self._math_buf: list[str] = []

    @staticmethod
    def _role(tag: str, a: dict) -> str:
        cls = a.get("class") or ""
        if tag in ("td", "th") or "ltx_td" in cls or "ltx_th" in cls:
            return "cell"
        if tag == "tr" or "ltx_tr" in cls:
            return "row"
        return ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self._skip += 1
            return
        if tag == "math":
            self._math += 1
            alt = a.get("alttext")
            if alt:
                self.parts.append(f" {alt} ")
            return
        role = self._role(tag, a)
        self._stack.append(role)
        if role == "cell":
            self._cell += 1
            self.parts.append(" | ")
        elif role == "row":
            self._row += 1
            self.parts.append("\n")
        elif self._cell:
            if tag in self.BLOCK or tag == "li":
                self.parts.append(" ")      # keep a table cell on one line
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "img" and a.get("alt"):
            self.parts.append(f"[image: {a['alt'][:200]}]")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if tag == "math":
            self._math = max(0, self._math - 1)
            return
        role = self._stack.pop() if self._stack else ""
        if role == "cell":
            self._cell = max(0, self._cell - 1)
            return
        if role == "row":
            self._row = max(0, self._row - 1)
            self.parts.append(" |\n")
            return
        if self._cell:
            return
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip or self._math:
            return
        if self._cell:
            data = " ".join(data.split())
            if data:
                self.parts.append(data)
        elif self._row:
            return          # whitespace / stray text between cells
        else:
            self.parts.append(data)

    def text(self) -> str:
        t = "".join(self.parts)
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        t = re.sub(r" ?\| ?", " | ", t)
        t = re.sub(r"\n ", "\n", t)
        t = re.sub(r"\n- *(?=\n)", "\n", t)          # empty list items (nav menus)
        t = re.sub(r"\n(?: *\| *)+\n", "\n", t)       # empty table rows
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()


def strip_arxiv_chrome(t: str) -> str:
    """Drop the arXiv HTML page header (nav, license line) that precedes the
    paper's own title heading."""
    m = re.search(r"\nLicense: .*\n", t)
    if m and m.start() < 20000:
        t = t[m.end():]
    m = re.search(r"\nInstructions for reporting errors", t)
    if m:
        t = t[:m.start()]
    return t.strip()


def html_to_text(raw: str) -> str:
    p = _TextHTML()
    p.feed(raw)
    return p.text()


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(f"\n\n[page {i + 1}]\n" + (page.extract_text() or ""))
        except Exception as e:  # pragma: no cover - pypdf quirks
            pages.append(f"\n\n[page {i + 1}: extraction failed: {e}]")
    return "".join(pages).strip()


class Fetcher:
    """HTTP GET with retries, polite rate limiting per host and byte caching."""

    HOST_INTERVALS = {"export.arxiv.org": 3.0, "api.github.com": 6.5,
                      "api.semanticscholar.org": 1.5, "api.openalex.org": 0.4}

    def __init__(self, cache: Cache, user_agent: str, timeout: int = 60,
                 min_interval: float = 0.5, sleep=time.sleep, max_retries: int = 4):
        self.cache = cache
        self.ua = user_agent
        self.timeout = timeout
        self.min_interval = min_interval
        self.host_intervals = dict(self.HOST_INTERVALS)
        self._last: dict[str, float] = {}
        self._sleep = sleep
        self.max_retries = max_retries

    def _throttle(self, url: str):
        host = urllib.parse.urlparse(url).netloc
        now = time.monotonic()
        wait = self._last.get(host, 0) + self.host_intervals.get(host, self.min_interval) - now
        if wait > 0:
            self._sleep(wait)
        self._last[host] = time.monotonic()

    def _open(self, url: str, headers: dict) -> tuple[int, bytes, dict]:
        """Single HTTP transaction (no retries). HTTP errors are returned as a
        status code; network failures raise OSError."""
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            try:
                body = e.read()
            except Exception:
                body = b""
            return e.code, body, dict(e.headers)

    def get(self, url: str, headers: dict | None = None, use_cache: bool = True,
            cache_stage: str = "http") -> tuple[int, bytes, dict]:
        key = cache_key(url=url, headers=sorted((headers or {}).items()))
        if use_cache:
            hit = self.cache.get(cache_stage, key)
            if hit is not None and hit.get("status") == 200:
                return hit["status"], bytes.fromhex(hit["body_hex"]), hit["headers"]
        h = {"User-Agent": self.ua, "Accept": "*/*"}
        h.update(headers or {})
        last_err: Exception | None = None
        status, body, rh = 0, b"", {}
        for attempt in range(self.max_retries):
            self._throttle(url)
            try:
                status, body, rh = self._open(url, h)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                wait = min(60, 3.0 * 2 ** attempt)
                log.warning("network error for %s: %s; retrying in %.0fs", url, e, wait)
                self._sleep(wait)
                continue
            if status in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                ra = rh.get("Retry-After")
                # honour Retry-After but never sleep more than 60s (a daily-quota
                # header can say hours; better to fail fast and resume from cache)
                wait = min(60.0, float(ra)) if ra and str(ra).isdigit() else min(60, 3.0 * 2 ** attempt)
                log.warning("HTTP %d for %s; retrying in %.0fs", status, url, wait)
                self._sleep(wait)
                continue
            break
        if status == 0:
            raise ConnectionError(f"failed to fetch {url}: {last_err}")
        rec = {"url": url, "status": status, "headers": {k: v for k, v in rh.items()
               if k.lower() in ("content-type", "content-length", "last-modified", "etag")},
               "body_hex": body.hex(), "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if status == 200:
            self.cache.put(cache_stage, key, rec)
        return status, body, rec["headers"]

    def get_json(self, url: str, headers: dict | None = None, use_cache: bool = True) -> Any:
        import json
        status, body, _ = self.get(url, headers, use_cache)
        if status != 200:
            raise urllib.error.HTTPError(url, status, body[:200].decode(errors="replace"), {}, None)
        return json.loads(body.decode("utf-8", errors="replace"))


def _arxiv_abs_meta(fetcher: Fetcher, arxiv_id: str) -> dict:
    import xml.etree.ElementTree as ET
    status, body, _ = fetcher.get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}")
    meta = {"arxiv_id": arxiv_id, "title": "", "abstract": "", "year": None, "authors": []}
    if status != 200:
        return meta
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(body)
    entry = root.find("a:entry", ns)
    if entry is None:
        return meta
    meta["title"] = " ".join((entry.findtext("a:title", "", ns) or "").split())
    meta["abstract"] = " ".join((entry.findtext("a:summary", "", ns) or "").split())
    pub = entry.findtext("a:published", "", ns) or ""
    meta["year"] = int(pub[:4]) if pub[:4].isdigit() else None
    meta["authors"] = [a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)]
    return meta


def retrieve_document(fetcher: Fetcher, cache: Cache, *, arxiv_id: str | None = None,
                      url: str | None = None, max_chars: int = 260000) -> dict:
    """Return {source_url, arxiv_id, title, abstract, year, text, text_format,
    doc_hash, truncated}. Cached by (arxiv_id, url, max_chars)."""
    key = cache_key(arxiv_id=arxiv_id, url=url, max_chars=max_chars, v=1)
    hit = cache.get("document", key)
    if hit is not None:
        return hit
    doc: dict[str, Any] = {"source_url": url, "arxiv_id": arxiv_id, "title": "", "abstract": "",
                           "year": None, "text": "", "text_format": "none", "truncated": False,
                           "retrieval_errors": []}
    if arxiv_id:
        doc.update({k: v for k, v in _arxiv_abs_meta(fetcher, arxiv_id).items()})
        doc["source_url"] = f"https://arxiv.org/abs/{arxiv_id}"
        for fmt, u in (("arxiv_html", f"https://arxiv.org/html/{arxiv_id}"),
                       ("pdf", f"https://arxiv.org/pdf/{arxiv_id}")):
            try:
                status, body, hdr = fetcher.get(u)
            except ConnectionError as e:
                doc["retrieval_errors"].append(f"{fmt}: {e}")
                continue
            if status != 200:
                doc["retrieval_errors"].append(f"{fmt}: HTTP {status}")
                continue
            try:
                text = pdf_to_text(body) if fmt == "pdf" or body[:5] == b"%PDF-" \
                    else strip_arxiv_chrome(html_to_text(body.decode("utf-8", errors="replace")))
            except Exception as e:
                doc["retrieval_errors"].append(f"{fmt}: parse failed: {e}")
                continue
            if len(text) > 2000:
                doc["text"], doc["text_format"], doc["text_url"] = text, fmt, u
                break
        if not doc["text"]:
            doc["text"] = f"# {doc['title']}\n\n{doc['abstract']}"
            doc["text_format"] = "abstract_only"
    elif url:
        status, body, hdr = fetcher.get(url)
        if status != 200:
            doc["retrieval_errors"].append(f"HTTP {status}")
        else:
            ct = hdr.get("Content-Type", "")
            if "pdf" in ct or body[:5] == b"%PDF-":
                doc["text"], doc["text_format"] = pdf_to_text(body), "pdf"
            else:
                raw = body.decode("utf-8", errors="replace")
                m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I)
                doc["title"] = html.unescape(" ".join(m.group(1).split())) if m else url
                doc["text"], doc["text_format"] = html_to_text(raw), "html"
    else:
        raise ValueError("retrieve_document needs arxiv_id or url")
    if len(doc["text"]) > max_chars:
        doc["text"] = doc["text"][:max_chars] + "\n\n[TRUNCATED BY PIPELINE]"
        doc["truncated"] = True
    doc["doc_hash"] = sha256_text(doc["text"])
    doc["char_count"] = len(doc["text"])
    return cache.put("document", key, doc)
