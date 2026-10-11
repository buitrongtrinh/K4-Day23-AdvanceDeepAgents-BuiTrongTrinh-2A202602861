"""tools.py - Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import functools
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()  # EXA_API_KEY from .env, also when this file runs alone (python tools.py)

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRY_STATUS = {429, 500, 502, 503, 504}
ATOM = "{http://www.w3.org/2005/Atom}"
SUMMARY_CHARS = 600       # keep records compact: the agent's context is the scarce resource
WEB_SEARCH_CHARS = 10000
WEB_FETCH_CHARS = 12000
ARXIV_TERMS_MAX = 8
BOOLEAN_WORDS = {"AND", "OR", "NOT", "ANDNOT"}


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Waits `Retry-After` when the server gave one, else base * 2**attempt plus random jitter; never longer than `cap`.
    The last failed attempt re-raises without sleeping. Any other exception is not retried.
    """
    attempts = max(1, attempts)
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(max(float(exc.retry_after), 0.0), cap)
            else:
                delay = min(cap, base * 2 ** attempt)
                jitter = random.uniform(0, delay * 0.25)  # desynchronise clients; at the cap jitter goes downwards
                delay = delay + jitter if delay + jitter <= cap else delay - jitter
            time.sleep(delay)


# ---- provenance: what the tools really returned ----
# A report may only cite what a tool returned in this run. research.py checks it after the run, so that a lead which
# invents sources cannot pass whatever its prompt says. The tools run in parallel threads: appends are locked.
_seen_outputs, _seen_lock = [], threading.Lock()
_PAPER_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|huggingface\.co/papers/)([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?/?$", re.I)


_OPAQUE_ID = re.compile(r"[?&]id=([\w\-]{6,})|(10\.\d{4,9}/[^\s?#]+)", re.I)


def reset_seen():
    with _seen_lock:
        _seen_outputs.clear()


def _logged(fn):
    """Remember what a tool returned (goes under @tool, so the tool description is untouched)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        output = fn(*args, **kwargs)
        with _seen_lock:
            _seen_outputs.append(output)
        return output
    return wrapper


def seen_source(url):
    """True when a tool returned this url, or, for an arXiv / Hugging Face paper, its identifier (the agents may rewrite
    arxiv.org/pdf/X into arxiv.org/abs/X, which is the same paper)."""
    with _seen_lock:
        blob = "\n".join(_seen_outputs).lower()
    bare = re.sub(r"^https?://(www\.)?", "", str(url).strip()).rstrip("/").lower()
    if bare and bare in blob:
        return True
    paper = _PAPER_ID.search(str(url).strip())
    if paper:
        return paper.group(1).lower() in blob
    # the same document under another path: openreview.net/pdf?id=X vs /forum?id=X, a DOI behind doi.org or a publisher
    token = _OPAQUE_ID.search(str(url).strip())
    return bool(token) and (token.group(1) or token.group(2)).lower() in blob


# ---- shared helpers ----
class _Spacer:
    """Thread-safe minimum spacing between the starts of calls (the researchers run in parallel threads)."""

    def __init__(self, interval):
        self._interval = interval
        self._next = float("-inf")
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._interval
        if slot > now:
            time.sleep(slot - now)


_arxiv_spacer = _Spacer(3.0)   # arXiv API etiquette: one call every 3 seconds
_exa_spacer = _Spacer(0.5)     # the free Exa tier rejects bursts


def _clamp(value, low, high):
    return max(low, min(high, int(value)))


def _clean(text):
    return " ".join(str(text or "").split())


def _shorten(text, limit):
    text = _clean(text)
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _retry_after(response):
    """Seconds from a Retry-After header (a number or an HTTP date), else None."""
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def _redact(text):
    """Remove the Exa key from text that is going back to the agent (httpx errors contain the full URL)."""
    text = str(text)
    key = os.getenv("EXA_API_KEY", "").strip()
    if key:
        text = text.replace(key, "***")
    return re.sub(r"(exaApiKey=)[^&\s'\"]+", r"\1***", text)


def _error(exc):
    return f"ERROR: {type(exc).__name__}: {_redact(exc)}"[:600]


def _send(method, url, *, before=None, **kwargs):
    """One HTTP attempt: 429/5xx and network errors become RetryableError; other 4xx raise HTTPStatusError."""
    if before:
        before()
    try:
        response = httpx.request(method, url, timeout=30, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRY_STATUS:
        raise RetryableError(f"HTTP {response.status_code} from {response.url.host}", retry_after=_retry_after(response))
    response.raise_for_status()
    return response


def _request(method, url, *, retry=None, **kwargs):
    return with_retry(lambda: _send(method, url, **kwargs), **(retry or {}))


# ---- TODO 2: arXiv ----
def _arxiv_terms(query):
    words = re.findall(r"[^\W_]+(?:-[^\W_]+)*", str(query))
    return [w for w in words if w.upper() not in BOOLEAN_WORDS][:ARXIV_TERMS_MAX]


@tool
@_logged
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by a few keywords (all terms must match), newest first.
    Use 2-5 plain keywords, no quotes or operators. Returns a JSON list of {id, url, published, title, summary};
    `url` is the canonical https://arxiv.org/abs/<id> link. Best for recent preprints and foundational papers."""
    try:
        terms = _arxiv_terms(query)
        if not terms:
            return "NO RESULTS"
        params = {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate",
                  "sortOrder": "descending", "max_results": _clamp(max_results, 1, 30), "start": 0}
        response = _request("GET", ARXIV_URL, params=params, before=_arxiv_spacer.wait,
                            retry={"attempts": 6, "base": 3.0, "cap": 60.0})
        records = []
        for entry in ET.fromstring(response.text).findall(f"{ATOM}entry"):
            entry_id = _clean(entry.findtext(f"{ATOM}id"))
            if "/abs/" not in entry_id:  # arXiv reports query errors as an <entry> without an abs id
                if _clean(entry.findtext(f"{ATOM}title")).lower() == "error":
                    return f"ERROR: arXiv: {_shorten(entry.findtext(f'{ATOM}summary'), 200)}"
                continue
            paper_id = re.sub(r"v\d+$", "", entry_id.split("/abs/")[-1])
            records.append({"id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}",
                            "published": _clean(entry.findtext(f"{ATOM}published"))[:10],
                            "title": _clean(entry.findtext(f"{ATOM}title")),
                            "summary": _shorten(entry.findtext(f"{ATOM}summary"), SUMMARY_CHARS)})
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001 - a tool never raises
        return _error(exc)


# ---- TODO 3: Hugging Face ----
def _hf_record(item, prefer_ai_summary=False):
    """One API item -> compact record (summary is cut later, filtering uses the full text). None without paper.id."""
    paper = item.get("paper") or {}
    paper_id = paper.get("id")
    if not paper_id:
        return None
    summary = (paper.get("ai_summary") if prefer_ai_summary else None) or paper.get("summary") or item.get("summary")
    return {"id": paper_id, "url": f"https://huggingface.co/papers/{paper_id}",
            "published": _clean(paper.get("publishedAt") or item.get("publishedAt"))[:10],
            "title": _clean(paper.get("title") or item.get("title")), "summary": _clean(summary),
            "upvotes": paper.get("upvotes") or 0, "github": paper.get("githubRepo"), "stars": paper.get("githubStars")}


def _hf_output(records):
    if not records:
        return "NO RESULTS"
    return json.dumps([dict(r, summary=_shorten(r["summary"], SUMMARY_CHARS)) for r in records], ensure_ascii=False)


@tool
@_logged
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        date = date.strip()
        if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            return "ERROR: date must be YYYY-MM-DD"
        limit = _clamp(limit, 1, 100)
        keyword = keyword.strip().lower()
        params = {"limit": 100 if keyword else limit}  # a keyword filters client-side: look at the whole day
        if date:
            params["date"] = date
        items = _request("GET", HF_DAILY_URL, params=params).json()
        records = [r for r in (_hf_record(i) for i in items if isinstance(i, dict)) if r]
        if keyword:
            records = [r for r in records if keyword in f"{r['title']} {r['summary']}".lower()]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return _hf_output(records[:limit])
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
@_logged
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic (any year). Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}; `url` is https://huggingface.co/papers/<id>."""
    try:
        query = _clean(query)
        if not query:
            return "NO RESULTS"
        items = _request("GET", HF_SEARCH_URL, params={"q": query, "limit": _clamp(limit, 1, 50)}).json()
        records = [r for r in (_hf_record(i, prefer_ai_summary=True) for i in items if isinstance(i, dict)) if r]
        return _hf_output(records)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
class ExaError(Exception):
    """Exa answered, but with a JSON-RPC error or a tool error: retrying the same call will not help."""


def _mcp_message(text):
    """The JSON-RPC message of a response: server-sent events ('data: {...}' lines) or a plain JSON body."""
    messages = []
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                messages.append(json.loads(line[5:].strip()))
            except ValueError:
                continue
    if not messages:
        try:
            messages.append(json.loads(text))
        except ValueError:
            pass
    answers = [m for m in messages if isinstance(m, dict) and ("result" in m or "error" in m)]
    if not answers:
        raise ExaError("Exa returned no JSON-RPC answer")
    return answers[-1]


def _exa_rate_limited(result, text):
    """The free tier answers HTTP 200 and flags the limit in result._meta (and in the text) instead of sending 429."""
    meta = result.get("_meta")
    if isinstance(meta, dict) and any("ratelimit" in str(k).lower().replace("_", "") and v for k, v in meta.items()):
        return True
    return "free mcp rate limit" in text[:300].lower()


def _exa_call(tool_name, arguments):
    """Call one Exa MCP tool over plain HTTP (JSON-RPC tools/call). Returns the text; retries when rate limited."""
    key = os.getenv("EXA_API_KEY", "").strip()
    params = {"exaApiKey": key} if key else None
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def attempt():
        response = _send("POST", EXA_URL, json=body, headers=headers, params=params, before=_exa_spacer.wait)
        message = _mcp_message(response.text)
        if "error" in message:
            error = message["error"]
            raise ExaError(error.get("message") if isinstance(error, dict) else str(error))
        result = message.get("result") or {}
        text = "\n".join(p.get("text", "") for p in result.get("content") or [] if p.get("type") == "text").strip()
        if _exa_rate_limited(result, text):
            raise RetryableError("Exa free-tier rate limit")
        if result.get("isError"):
            raise ExaError(text[:300] or "Exa tool error")
        return text

    return with_retry(attempt, attempts=8, base=2.0, cap=60.0)


@tool
@_logged
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs.
    Use it for blogs, surveys, project pages and recent news; every result starts with 'Title: ... URL: ...'."""
    try:
        query = _clean(query)
        if not query:
            return "NO RESULTS"
        arguments = {"query": query, "objective": _clean(objective) or f"Find pages relevant to: {query}",
                     "numResults": _clamp(num_results, 1, 10)}
        text = _exa_call("web_search_exa", arguments)
        return text[:WEB_SEARCH_CHARS] if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
@_logged
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return "ERROR: url must start with http:// or https://"
        text = _exa_call("web_fetch_exa", {"urls": [url]})
        return text[:WEB_FETCH_CHARS] if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
