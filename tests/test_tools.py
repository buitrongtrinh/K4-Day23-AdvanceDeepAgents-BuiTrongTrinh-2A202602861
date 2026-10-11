import json

import httpx
import pytest

import tools
from tools import RetryableError, with_retry


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)
    monkeypatch.setattr(tools._arxiv_spacer, "wait", lambda: None)   # spacing is tested on its own (_Spacer)
    monkeypatch.setattr(tools._exa_spacer, "wait", lambda: None)
    return sleeps


def flaky(failures, error):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise error
        return "done"
    return fn, calls


# ---- with_retry ----
def test_retry_succeeds_after_transient_failures(no_sleep):
    fn, calls = flaky(2, RetryableError("503"))
    assert with_retry(fn) == "done"
    assert calls["n"] == 3 and len(no_sleep) == 2


def test_retry_after_header_is_used_and_capped(no_sleep):
    fn, _ = flaky(1, RetryableError("429", retry_after=7))
    with_retry(fn, cap=30)
    fn, _ = flaky(1, RetryableError("429", retry_after=500))
    with_retry(fn, cap=30)
    assert no_sleep == [7, 30]


def test_backoff_grows_has_jitter_and_respects_cap(no_sleep):
    fn, _ = flaky(6, RetryableError("500"))
    with_retry(fn, attempts=7, base=1.0, cap=10.0)
    assert len(no_sleep) == 6 and all(0 < d <= 10.0 for d in no_sleep)
    assert no_sleep[0] < no_sleep[1] < no_sleep[2]
    assert 1.0 <= no_sleep[0] <= 1.25 and 2.0 <= no_sleep[1] <= 2.5


def test_last_attempt_raises_without_sleeping(no_sleep):
    fn, calls = flaky(99, RetryableError("500"))
    with pytest.raises(RetryableError):
        with_retry(fn, attempts=3)
    assert calls["n"] == 3 and len(no_sleep) == 2


def test_other_errors_are_not_retried(no_sleep):
    fn, calls = flaky(5, ValueError("bug"))
    with pytest.raises(ValueError):
        with_retry(fn)
    assert calls["n"] == 1 and no_sleep == []


# ---- HTTP plumbing ----
def response(status=200, text="", json_body=None, headers=None, url="https://example.org/x"):
    request = httpx.Request("GET", url)
    if json_body is not None:
        return httpx.Response(status, json=json_body, headers=headers, request=request)
    return httpx.Response(status, text=text, headers=headers, request=request)


def fake_requests(monkeypatch, *answers):
    queue, seen = list(answers), []

    def fake(method, url, **kwargs):
        seen.append((method, url, kwargs))
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(tools.httpx, "request", fake)
    return seen


def test_429_with_retry_after_then_success(monkeypatch, no_sleep):
    fake_requests(monkeypatch, response(429, headers={"retry-after": "4"}), response(200, json_body=[]))
    assert tools._request("GET", "https://example.org/x").json() == []
    assert no_sleep == [4.0]


def test_transport_error_is_retried(monkeypatch, no_sleep):
    fake_requests(monkeypatch, httpx.ConnectTimeout("slow"), response(200, text="ok"))
    assert tools._request("GET", "https://example.org/x").text == "ok"
    assert len(no_sleep) == 1


def test_client_error_is_not_retried(monkeypatch, no_sleep):
    seen = fake_requests(monkeypatch, response(400))
    with pytest.raises(httpx.HTTPStatusError):
        tools._request("GET", "https://example.org/x")
    assert len(seen) == 1 and no_sleep == []


# ---- arXiv ----
ATOM = """<?xml version='1.0'?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2501.00001v2</id><published>2025-01-02T10:00:00Z</published>
<title>A  Title
 with breaks</title><summary>  Some
   summary.  </summary></entry></feed>"""


def test_arxiv_terms_are_sanitised():
    assert tools._arxiv_terms('"world model" AND: (self-supervised) OR agents!') == ["world", "model", "self-supervised", "agents"]
    assert tools._arxiv_terms('"" : AND') == []


def test_arxiv_no_terms_makes_no_call(monkeypatch):
    seen = fake_requests(monkeypatch, response(200, text=ATOM))
    assert tools.arxiv_search.invoke({"query": '" : "'}) == "NO RESULTS"
    assert seen == []


def test_arxiv_record_shape(monkeypatch):
    seen = fake_requests(monkeypatch, response(200, text=ATOM))
    records = json.loads(tools.arxiv_search.invoke({"query": "world model", "max_results": 3}))
    assert records == [{"id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "published": "2025-01-02",
                        "title": "A Title with breaks", "summary": "Some summary."}]
    method, url, kwargs = seen[0]
    assert url.startswith("https://") and kwargs["params"]["search_query"] == "all:world AND all:model"


def test_arxiv_spacing_between_calls(monkeypatch, no_sleep):
    now = [100.0]
    monkeypatch.setattr(tools.time, "monotonic", lambda: now[0])
    spacer = tools._Spacer(3.0)
    spacer.wait()
    spacer.wait()
    assert no_sleep == [3.0]


def test_arxiv_empty_feed_and_errors(monkeypatch):
    fake_requests(monkeypatch, response(200, text="<feed xmlns='http://www.w3.org/2005/Atom'></feed>"))
    assert tools.arxiv_search.invoke({"query": "zzz"}) == "NO RESULTS"
    fake_requests(monkeypatch, response(200, text="not xml"))
    assert tools.arxiv_search.invoke({"query": "zzz"}).startswith("ERROR: ParseError")
    fake_requests(monkeypatch, response(503))
    assert tools.arxiv_search.invoke({"query": "zzz"}).startswith("ERROR: RetryableError")


# ---- Hugging Face ----
def hf_item(pid, title, summary, upvotes, **extra):
    return {"paper": {"id": pid, "title": title, "summary": summary, "upvotes": upvotes, "githubRepo": "https://g/x",
                      "githubStars": 5, "publishedAt": "2025-02-03T00:00:00.000Z", **extra}}


def test_hf_daily_filters_sorts_and_skips_items_without_id(monkeypatch):
    items = [hf_item("1", "World models", "about worlds", 3), hf_item("2", "Other", "a world model appears", 9),
             hf_item("3", "Unrelated", "nothing", 50), {"paper": {}}]
    seen = fake_requests(monkeypatch, response(200, json_body=items))
    records = json.loads(tools.hf_daily_papers.invoke({"limit": 5, "keyword": "World Model"}))
    assert [r["id"] for r in records] == ["2", "1"]
    assert records[0]["url"] == "https://huggingface.co/papers/2" and records[0]["published"] == "2025-02-03"
    assert seen[0][2]["params"]["limit"] == 100   # a keyword filters client-side over the whole day


def test_hf_daily_rejects_bad_date_and_reports_no_results(monkeypatch):
    assert tools.hf_daily_papers.invoke({"date": "yesterday"}).startswith("ERROR")
    fake_requests(monkeypatch, response(200, json_body=[]))
    assert tools.hf_daily_papers.invoke({}) == "NO RESULTS"


def test_hf_search_prefers_ai_summary_and_truncates(monkeypatch):
    items = [hf_item("9", "T", "x" * 2000, 1, ai_summary="short ai summary"), hf_item("8", "U", "y" * 2000, 1)]
    fake_requests(monkeypatch, response(200, json_body=items))
    records = json.loads(tools.hf_search_papers.invoke({"query": "world model"}))
    assert records[0]["summary"] == "short ai summary"
    assert len(records[1]["summary"]) <= 600
    assert tools.hf_search_papers.invoke({"query": "  "}) == "NO RESULTS"


# ---- Exa ----
def sse(result=None, error=None):
    message = {"jsonrpc": "2.0", "id": 1, **({"error": error} if error else {"result": result})}
    return response(200, text=f"event: message\ndata: {json.dumps(message)}\n\n")


def test_exa_search_parses_sse_and_sends_arguments(monkeypatch):
    seen = fake_requests(monkeypatch, sse({"content": [{"type": "text", "text": "Title: T\nURL: https://e.org"}]}))
    assert tools.web_search.invoke({"query": "world models"}) == "Title: T\nURL: https://e.org"
    arguments = seen[0][2]["json"]["params"]["arguments"]
    assert arguments["query"] == "world models" and arguments["objective"] and arguments["numResults"] == 5


def test_exa_rate_limit_flag_is_retried(monkeypatch, no_sleep):
    limited = sse({"_meta": {"ai.exa/rateLimited": True}, "content": [{"type": "text", "text": "You've hit Exa's free MCP rate limit."}]})
    ok = sse({"content": [{"type": "text", "text": "real page"}]})
    seen = fake_requests(monkeypatch, limited, limited, ok)
    assert tools.web_fetch.invoke({"url": "https://arxiv.org/abs/1803.10122"}) == "real page"
    assert len(seen) == 3 and len(no_sleep) == 2
    assert seen[0][2]["json"]["params"]["arguments"] == {"urls": ["https://arxiv.org/abs/1803.10122"]}


def test_exa_rate_limit_that_never_ends_returns_error_not_page_text(monkeypatch):
    limited = sse({"_meta": {"ai.exa/rateLimited": True}, "content": [{"type": "text", "text": "You've hit Exa's free MCP rate limit."}]})
    fake_requests(monkeypatch, limited)
    out = tools.web_search.invoke({"query": "x"})
    assert out.startswith("ERROR: RetryableError") and "You've hit" not in out   # the limit text never reaches the agent


def test_exa_jsonrpc_error_and_empty_results(monkeypatch):
    fake_requests(monkeypatch, sse(error={"code": -32602, "message": "bad arguments"}))
    assert tools.web_search.invoke({"query": "x"}) == "ERROR: ExaError: bad arguments"
    fake_requests(monkeypatch, sse({"content": []}))
    assert tools.web_search.invoke({"query": "x"}) == "NO RESULTS"
    assert tools.web_fetch.invoke({"url": "not a url"}).startswith("ERROR")


def test_exa_key_is_sent_but_never_returned(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret-key-123")
    seen = fake_requests(monkeypatch, response(401, url="https://mcp.exa.ai/mcp?exaApiKey=secret-key-123"))
    out = tools.web_search.invoke({"query": "x"})
    assert seen[0][2]["params"] == {"exaApiKey": "secret-key-123"}
    assert out.startswith("ERROR") and "secret-key-123" not in out


def test_redact_removes_key_in_urls_and_text(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "abc123")
    assert "abc123" not in tools._redact("failed for https://mcp.exa.ai/mcp?exaApiKey=zzz999&x=1 and abc123")


def test_registry_has_the_five_tools():
    assert [t.name for t in tools.SOURCE_TOOLS] == ["arxiv_search", "hf_daily_papers", "hf_search_papers", "web_search", "web_fetch"]


# ---- provenance ----
def test_seen_source_knows_what_the_tools_returned(monkeypatch):
    tools.reset_seen()
    fake_requests(monkeypatch, response(200, text=ATOM))
    tools.arxiv_search.invoke({"query": "world model"})
    fake_requests(monkeypatch, sse({"content": [{"type": "text", "text": "Title: T\nURL: https://www.Example.org/page/\nHighlights: x"}]}))
    tools.web_search.invoke({"query": "x"})
    assert tools.seen_source("https://arxiv.org/abs/2501.00001")           # returned by arxiv_search
    assert tools.seen_source("https://arxiv.org/pdf/2501.00001v3")         # same paper, other form
    assert tools.seen_source("http://example.org/page")                    # scheme, www and trailing slash do not matter
    assert not tools.seen_source("https://arxiv.org/abs/2405.12345")       # invented
    assert not tools.seen_source("https://example.org/other")
    tools.reset_seen()
    assert not tools.seen_source("https://arxiv.org/abs/2501.00001")


def test_logging_keeps_the_tool_description():
    assert "Search arXiv papers" in tools.arxiv_search.description
    assert set(tools.arxiv_search.args) == {"query", "max_results"}


def test_seen_source_accepts_the_same_document_under_another_path():
    tools.reset_seen()
    tools._seen_outputs.append("URL: https://openreview.net/forum?id=BZ5a1r-kVsf and https://doi.org/10.1109/PIMRC69400.2026.11711756")
    assert tools.seen_source("https://openreview.net/pdf?id=BZ5a1r-kVsf")
    assert tools.seen_source("https://ieeexplore.example/doi/10.1109/PIMRC69400.2026.11711756")
    assert not tools.seen_source("https://openreview.net/pdf?id=NOTREALID99")
    tools.reset_seen()
