import json
from types import SimpleNamespace

import pytest

import research
from agents import REPORT_PATH, SOURCES_PATH, build_lead_agent, build_subagents


class FakeBackend:
    """Stands in for a sandbox: download_files() answers from a dict (None = file missing)."""

    def __init__(self, files):
        self.files = files

    def download_files(self, paths):
        return [SimpleNamespace(path=p, content=self.files.get(p), error=None if self.files.get(p) is not None else "file_not_found")
                for p in paths]


SOURCES = [{"n": 1, "id": "1", "url": "https://arxiv.org/abs/1", "title": "A", "date": "2025-01-01", "source": "arxiv"},
           {"n": 2, "id": "", "url": "https://example.org", "title": "B", "date": "n.d.", "source": "web"},
           {"n": 3, "id": "3", "url": "https://huggingface.co/papers/3", "title": "C", "date": "2025-02-02", "source": "hf-search"}]


def messages():
    ai = lambda calls, usage=None: SimpleNamespace(tool_calls=calls, usage_metadata=usage)  # noqa: E731
    return [SimpleNamespace(tool_calls=[], usage_metadata=None),
            ai([{"name": "task"}, {"name": "task"}, {"name": "execute"}], {"input_tokens": 100, "output_tokens": 10}),
            ai([{"name": "task"}, {"name": "write_todos"}], {"input_tokens": 50, "output_tokens": 5})]


@pytest.mark.parametrize("topic,expected", [
    ("survey about world model", "survey-about-world-model"),
    ("  LLM agents & tool-use!! ", "llm-agents-tool-use"),
    ("../../x", "x"),
    ("", "topic"), ("   ", "topic"), ("///", "topic"),
])
def test_slugify(topic, expected):
    assert research.slugify(topic) == expected


def test_slugify_is_bounded_and_never_escapes():
    slug = research.slugify("word " * 40)
    assert len(slug) <= 60 and not slug.endswith("-")
    assert "/" not in research.slugify("../../etc/passwd") and ".." not in research.slugify("../../etc/passwd")


def test_summarize_counts_tools_and_tokens():
    summary = research.summarize(messages(), 12.34, "m")
    assert summary == {"model": "m", "elapsed_s": 12.3, "subagent_calls": 3,
                       "tool_calls": {"task": 3, "execute": 1, "write_todos": 1}, "tokens": {"input": 150, "output": 15}}


def test_save_outputs_writes_three_files(tmp_path):
    backend = FakeBackend({REPORT_PATH: "# R\n[1]".encode(), SOURCES_PATH: json.dumps(SOURCES).encode()})
    path = research.save_outputs(backend, "survey about x", messages(), 5.0, "m", reports_dir=tmp_path / "reports")
    assert path.name == "survey-about-x.md" and path.read_text() == "# R\n[1]"
    meta = json.loads((tmp_path / "reports" / "survey-about-x.meta.json").read_text())
    assert meta["topic"] == "survey about x" and meta["n_sources"] == 3 and meta["subagent_calls"] == 3
    assert meta["source_families"] == ["arxiv", "hf-search", "web"]
    assert json.loads((tmp_path / "reports" / "survey-about-x.sources.json").read_text()) == SOURCES


@pytest.mark.parametrize("files", [
    {},                                                                    # nothing produced
    {REPORT_PATH: b"   \n", SOURCES_PATH: json.dumps(SOURCES).encode()},   # empty report
    {REPORT_PATH: b"# R", SOURCES_PATH: b"{not json"},                     # broken sources.json
    {REPORT_PATH: b"# R"},                                                 # no sources.json
    {REPORT_PATH: b"# R", SOURCES_PATH: b"[]"},                            # no sources
    {REPORT_PATH: b"# R", SOURCES_PATH: b'{"a": 1}'},                      # not a list
])
def test_failed_runs_write_nothing(tmp_path, files):
    with pytest.raises(RuntimeError):
        research.save_outputs(FakeBackend(files), "t", messages(), 1.0, "m", reports_dir=tmp_path / "reports")
    assert not (tmp_path / "reports").exists() or list((tmp_path / "reports").iterdir()) == []


def test_main_without_topic_returns_2(capsys):
    assert research.main("   ") == 2
    assert "usage" in capsys.readouterr().err


def test_main_without_model_returns_1(monkeypatch, capsys):
    monkeypatch.delenv("LAB_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_DEPLOYMENT_MODEL", raising=False)
    assert research.main("topic") == 1
    assert "FAILED" in capsys.readouterr().err


# ---- agents ----
def test_subagents_have_limits_and_the_right_tools():
    specs = {s["name"]: s for s in build_subagents()}
    assert [t.name for t in specs["researcher"]["tools"]] == ["arxiv_search", "hf_daily_papers", "hf_search_papers", "web_search", "web_fetch"]
    assert [t.name for t in specs["citation-checker"]["tools"]] == ["web_fetch"]
    for spec in specs.values():
        names = {type(m).__name__ for m in spec["middleware"]}
        assert {"ModelCallLimitMiddleware", "ToolCallLimitMiddleware"} <= names
        assert spec["description"] and spec["system_prompt"]


def test_lead_agent_builds_with_todo_tool_and_limits():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self
    agent = build_lead_agent(None, Fake(messages=iter([])))
    nodes = set(agent.get_graph().nodes)
    assert "tools" in nodes and any("TodoList" in n for n in nodes)
    assert any("ModelCallLimit" in n for n in nodes)


def test_prompt_paths_are_filled_in():
    import re

    from agents import FINALIZER_PATH, LEAD_PROMPT, MASTER_PATH, NOTES_DIR, REPORT_PATH, RESEARCHER_PROMPT, VALIDATOR_PATH
    for path in (NOTES_DIR, MASTER_PATH, REPORT_PATH, FINALIZER_PATH, VALIDATOR_PATH):
        assert path in LEAD_PROMPT
    for prompt in (LEAD_PROMPT, RESEARCHER_PROMPT):
        assert not re.search(r"\{[A-Za-z_]+\}", prompt)   # no unfilled f-string placeholder


# ---- failover to the next model ----
class Req:
    """Stands in for a ModelRequest: only .model and .override(model=...) matter to the middleware."""

    def __init__(self, model="primary"):
        self.model = model

    def override(self, model):
        return Req(model)


def failover(script, fallbacks=("second", "third")):
    """script: {model name: AIMessage | Exception}. Returns (answer or exception, models tried in order)."""
    from types import SimpleNamespace

    from agents import ModelFailover
    tried = []

    def handler(request):
        tried.append(request.model)
        outcome = script[request.model]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(result=[outcome])
    try:
        return ModelFailover(fallbacks).wrap_model_call(Req(), handler), tried
    except Exception as exc:  # noqa: BLE001
        return exc, tried


def test_failover_skips_a_failing_model():
    import openai
    from langchain_core.messages import AIMessage

    down = openai.APIConnectionError(request=__import__("httpx").Request("POST", "http://x"))
    answer, tried = failover({"primary": down, "second": AIMessage(content="from second")})
    assert answer.result[0].content == "from second" and tried == ["primary", "second"]


def test_failover_skips_an_empty_answer_and_accepts_tool_calls():
    from langchain_core.messages import AIMessage
    answer, tried = failover({"primary": AIMessage(content="\n\n\n"), "second": AIMessage(content=" "),
                              "third": AIMessage(content="", tool_calls=[{"name": "ls", "args": {}, "id": "1"}])})
    assert tried == ["primary", "second", "third"] and answer.result[0].tool_calls
    answer, tried = failover({"primary": AIMessage(content=[{"type": "text", "text": "hi"}])})
    assert tried == ["primary"]                     # block-style text is text; the first model answered: no more calls


def test_failover_raises_when_every_model_is_empty_or_down():
    from langchain_core.messages import AIMessage
    from agents import EmptyReplyError, _is_transient
    error, tried = failover({m: AIMessage(content="") for m in ("primary", "second", "third")})
    assert isinstance(error, EmptyReplyError) and tried == ["primary", "second", "third"]
    assert _is_transient(error)                    # so that the retry around the failover backs off and starts over


def test_failover_does_not_hide_real_errors():
    import httpx
    import openai
    from langchain_core.messages import AIMessage
    bad_key = openai.AuthenticationError("bad key", response=httpx.Response(401, request=httpx.Request("POST", "http://x")), body=None)
    error, tried = failover({"primary": bad_key, "second": AIMessage(content="never reached")})
    assert error is bad_key and tried == ["primary"]
    error, tried = failover({"primary": ValueError("a bug"), "second": AIMessage(content="never reached")})
    assert isinstance(error, ValueError) and tried == ["primary"]


def test_fallback_models_are_copies_under_other_names(monkeypatch):
    from agents import fallback_models
    from model import make_model
    monkeypatch.setenv("LAB_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("LAB_MODEL", "lab23")
    monkeypatch.setenv("LAB_API_KEY", "k")
    primary = make_model()
    monkeypatch.delenv("LAB_FALLBACK_MODELS", raising=False)
    assert fallback_models(primary) == []
    monkeypatch.setenv("LAB_FALLBACK_MODELS", " gemini/gemini-3.8-flash , openrouter/x:free ,")
    copies = fallback_models(primary)
    assert [c.model_name for c in copies] == ["gemini/gemini-3.8-flash", "openrouter/x:free"]
    assert primary.model_name == "lab23" and copies[0].root_client is primary.root_client
    assert fallback_models(object()) == []          # no OpenAI client (another provider): nothing to copy


def test_every_agent_has_retry_and_failover():
    from agents import LEAD_LIMITS
    assert not any(type(m).__name__ in {"ModelFailover", "ModelRetryMiddleware"} for m in LEAD_LIMITS)   # limits stay pure
    for spec in build_subagents(fallbacks=["x"]):
        names = [type(m).__name__ for m in spec["middleware"]]
        assert names.index("ModelFailover") == names.index("ModelRetryMiddleware") + 1    # retry (outer) around failover (inner)
        assert "ModelCallLimitMiddleware" in names and "ToolCallLimitMiddleware" in names


def test_http_errors_of_a_rotating_gateway_are_retried():
    import httpx
    import openai
    from agents import _is_transient

    request = httpx.Request("POST", "http://x/v1")

    def err(code):
        return openai.APIStatusError("m", response=httpx.Response(code, request=request), body=None)
    assert all(_is_transient(err(c)) for c in (400, 403, 404, 410, 429, 500, 503))   # one dead model among many
    assert not _is_transient(err(401))                                               # a wrong key never heals
    assert _is_transient(openai.APIConnectionError(request=request)) and _is_transient(TimeoutError())
    assert not _is_transient(ValueError("a bug")) and not _is_transient(KeyError("x"))


def test_context_overflow_is_not_retried():
    import httpx
    from langchain_openai.chat_models.base import OpenAIContextOverflowError
    from agents import _is_transient

    response = httpx.Response(400, request=httpx.Request("POST", "http://x/v1"))
    assert not _is_transient(OpenAIContextOverflowError("too long", response=response, body=None))


def test_limit_request_rate(monkeypatch):
    class Model:
        rate_limiter = None

    monkeypatch.delenv("LAB_MAX_RPS", raising=False)
    assert research.limit_request_rate(Model()).rate_limiter is None
    monkeypatch.setenv("LAB_MAX_RPS", "0.5")
    assert research.limit_request_rate(Model()).rate_limiter is not None


def test_set_request_timeout(monkeypatch):
    class Client:
        timeout, max_retries = 120.0, 2

    class Model:
        root_client = Client()

    monkeypatch.delenv("LAB_TIMEOUT", raising=False)
    model = research.set_request_timeout(Model())
    assert model.root_client.timeout == 60.0 and model.root_client.max_retries == 0
    monkeypatch.setenv("LAB_TIMEOUT", "25")
    assert research.set_request_timeout(Model()).root_client.timeout == 25.0
    assert research.set_request_timeout(object()) is not None          # no OpenAI client: left alone


def test_unverified_sources_are_the_ones_no_tool_returned(monkeypatch):
    import tools
    tools.reset_seen()
    tools._seen_outputs.append('[{"url": "https://arxiv.org/abs/1803.10122"}] URL: https://example.org/real')
    sources = [{"url": "https://arxiv.org/abs/1803.10122"}, {"url": "https://example.org/real"}, {"url": "https://arxiv.org/abs/2405.12345"}]
    backend = FakeBackend({SOURCES_PATH: json.dumps(sources).encode()})
    assert research.unverified_sources(backend) == ["https://arxiv.org/abs/2405.12345"]
    assert research.unverified_sources(FakeBackend({})) == []
    tools.reset_seen()


def test_report_problems_runs_the_full_validator():
    body = "# T\n\n## TL;DR\n- a [1]\n\n## Background\nb\n\n## One\nx\n\n## Two\nx\n\n## Three\nx\n\n## Trends and open problems\nt\n"
    sources = [{"n": i + 1, "url": f"https://example.org/{i}", "title": "T", "date": "n.d.", "source": "web"} for i in range(1)]
    refs = "\n## References\n[1] T. web. https://example.org/0 (n.d.)\n"
    ok = FakeBackend({REPORT_PATH: (body + refs).encode(), SOURCES_PATH: json.dumps(sources).encode()})
    problems = research.report_problems(ok)
    assert any("source families" in p for p in problems) and any("words" in p for p in problems)   # floors apply too
    assert research.report_problems(FakeBackend({})) == []                                          # nothing to judge: save_outputs fails
    broken = FakeBackend({REPORT_PATH: (body + refs).replace("[1] T.", "[1] A; B.").encode(), SOURCES_PATH: json.dumps(sources).encode()})
    assert research.report_problems(broken)


# ---- notes may only cite what a tool returned ----
def test_notes_with_invented_urls_are_rejected_at_write_time():
    import tools
    from agents import NotesProvenance
    tools.reset_seen()
    tools._seen_outputs.append("URL: https://arxiv.org/abs/1803.10122  URL: https://example.org/real")
    called = []

    def handler(request):
        called.append(request)
        return "written"

    def request(name, content):
        from types import SimpleNamespace
        key = "content" if name == "write_file" else "new_string"
        return SimpleNamespace(tool_call={"name": name, "id": "c1", "args": {"file_path": "/n.md", key: content}})
    guard = NotesProvenance()
    good = "## A\n- url: https://arxiv.org/abs/1803.10122\n- source: arxiv\n## B\n- **url**: <https://example.org/real>\n"
    assert guard.wrap_tool_call(request("write_file", good), handler) == "written"
    bad = good + "## C\n- url: https://arxiv.org/abs/2405.12345\n"
    rejected = guard.wrap_tool_call(request("write_file", bad), handler)
    assert rejected.status == "error" and "2405.12345" in rejected.content and len(called) == 1
    assert guard.wrap_tool_call(request("edit_file", bad), handler).status == "error"
    other = request("ls", "")
    other.tool_call["name"] = "ls"
    assert guard.wrap_tool_call(other, handler) == "written"      # other tools pass untouched
    tools.reset_seen()


def test_only_the_researcher_has_the_notes_guard():
    names = {s["name"]: [type(m).__name__ for m in s["middleware"]] for s in build_subagents()}
    assert "NotesProvenance" in names["researcher"]
    assert "NotesProvenance" not in names["citation-checker"] and "NotesProvenance" not in names["general-purpose"]
