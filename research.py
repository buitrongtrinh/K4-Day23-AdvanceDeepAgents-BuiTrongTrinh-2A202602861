"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.errors import GraphRecursionError

from check_citations import check, check_depth, check_structure
from agents import FINALIZER_PATH, MERGER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload
from tools import reset_seen, seen_source

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
MERGER_SOURCE = ROOT / "merge_notes.py"             # builds sources_all.json from the notes


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"\W+", "-", str(topic).lower()).strip("-")[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"TOPIC: {topic}\n\n"
            "Research this topic and write the cited survey report by following your workflow, from the plan to the "
            "spot-check of the citations.")


def _field(message, name):
    return message.get(name) if isinstance(message, dict) else getattr(message, name, None)


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    Walks the lead's messages only: subagent tokens are not included, so this undercounts the real cost.
    """
    tool_calls, tokens = Counter(), {"input": 0, "output": 0}
    for message in messages:
        for call in _field(message, "tool_calls") or []:
            tool_calls[call["name"]] += 1
        usage = _field(message, "usage_metadata") or {}
        tokens["input"] += usage.get("input_tokens", 0)
        tokens["output"] += usage.get("output_tokens", 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": tool_calls["task"],
            "tool_calls": dict(tool_calls), "tokens": tokens}


def _decode(data):
    return data.decode("utf-8") if isinstance(data, bytes) else data


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    Raises RuntimeError and writes NOTHING when the report is missing or empty or sources.json is missing or invalid:
    a failed run must never leave an empty or half-written report behind.
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    try:
        report = _decode(files.get(REPORT_PATH))
        raw_sources = _decode(files.get(SOURCES_PATH))
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"the agent's files are not valid UTF-8: {exc}") from exc
    if not report or not report.strip():
        raise RuntimeError(f"the agent produced no report at {REPORT_PATH}")
    if raw_sources is None:
        raise RuntimeError(f"the agent produced no {SOURCES_PATH}")
    try:
        sources = json.loads(raw_sources)
    except ValueError as exc:
        raise RuntimeError(f"{SOURCES_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources or not all(isinstance(s, dict) for s in sources):
        raise RuntimeError(f"{SOURCES_PATH} must be a non-empty JSON list of objects")

    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({s["source"] for s in sources if s.get("source")})}
    stem = slugify(topic)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{stem}.sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = reports_dir / f"{stem}.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def set_request_timeout(model):
    """Give up on a slow answer after LAB_TIMEOUT seconds (default 60) and let the retry middleware ask again, which a
    rotating gateway routes to another model. The OpenAI client's own silent retries are turned off for the same reason.
    Providers without an OpenAI client (no root_client) are left alone."""
    client = getattr(model, "root_client", None)
    if client is not None:
        client.timeout = float((os.getenv("LAB_TIMEOUT") or "").strip() or 60)
        client.max_retries = 0
    return model


def _last_text(messages):
    """The end of the lead's last message: the first clue when a run produced no report."""
    text = _field(messages[-1], "content") if messages else ""
    return (text if isinstance(text, str) else str(text))[-400:]


def unverified_sources(backend):
    """URLs of sources.json that no tool returned during this run: invented by the agent. Empty list = all verified."""
    raw = download(backend, [SOURCES_PATH]).get(SOURCES_PATH)
    try:
        sources = json.loads(_decode(raw)) if raw else []
    except ValueError:
        return []   # save_outputs reports the broken file
    return [s.get("url") for s in sources if isinstance(s, dict) and not seen_source(s.get("url", ""))]


def report_problems(backend):
    """The full validator (citations, layout, rubric floors) on the files the agent left: [] means the run is acceptable.
    The agent runs the same checks in the sandbox; this is the host's own verdict, so a lead that stopped early or ignored
    them cannot overwrite a good report with a bad one."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    try:
        report, sources = _decode(files.get(REPORT_PATH)) or "", json.loads(_decode(files.get(SOURCES_PATH)) or "null")
    except (UnicodeDecodeError, ValueError):
        return []   # save_outputs reports the unreadable files
    if not report.strip() or not isinstance(sources, list):
        return []
    return check(report, sources) + check_structure(report) + check_depth(report, sources)


def limit_request_rate(model):
    """Optional LAB_MAX_RPS (requests per second, e.g. 0.3): one limiter shared by the lead and every subagent, for
    free LLM tiers whose per-minute quota is smaller than a burst of parallel researchers."""
    value = (os.getenv("LAB_MAX_RPS") or "").strip()
    if value:
        model.rate_limiter = InMemoryRateLimiter(requests_per_second=float(value), max_bucket_size=3)
    return model


def _model_name(model):
    return getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "unknown")


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = topic.strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = set_request_timeout(limit_request_rate(make_model()))
    except (RuntimeError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    start = time.monotonic()
    reset_seen()
    with open_sandbox() as backend:  # the sandbox is always cleaned up, even on errors
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                         MERGER_PATH: MERGER_SOURCE.read_bytes()})
        agent = build_lead_agent(backend, model)
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": 1000})
        except GraphRecursionError as exc:
            print(f"FAILED: the agent hit the recursion limit ({exc})", file=sys.stderr)
            return 1
        invented = unverified_sources(backend)
        if invented:
            print(f"FAILED: {len(invented)} sources were never returned by a tool (invented?): {invented[:5]}", file=sys.stderr)
            return 1
        problems = report_problems(backend)
        if problems:
            print("FAILED: the report does not pass the validator:\n  " + "\n  ".join(problems[:8]), file=sys.stderr)
            return 1
        try:
            path = save_outputs(backend, topic, result["messages"], time.monotonic() - start, _model_name(model))
        except RuntimeError as exc:
            print(f"FAILED: {exc}\nlast message of the lead: {_last_text(result['messages'])!r}", file=sys.stderr)
            return 1
    print(f"Report saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
