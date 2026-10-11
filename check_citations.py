"""check_citations.py - validates a report against its sources.json.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It exits 0 and prints "OK: ..." when the report is consistent, else prints each problem and exits 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

FAMILIES = ("arxiv", "hf-daily", "hf-search", "web")
# a family label must match the URL shape of the tool that returned the source ("web" can be any URL)
FAMILY_PREFIX = {"arxiv": "https://arxiv.org/abs/", "hf-daily": "https://huggingface.co/papers/",
                 "hf-search": "https://huggingface.co/papers/"}

_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]; not the Markdown link [3](url)
_REF_LINE = re.compile(r"^\[(\d+)\]")
_URL = re.compile(r"https?://\S+")
_TRAILING = ".,;:)]>\"'"


def _expand(group):
    """'1, 3-5' -> [1, 3, 4, 5]."""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def _cited_numbers(body):
    """Numbers cited in the body, outside code blocks, code spans and Markdown links."""
    cited = set()
    for i, segment in enumerate(_CODE.split(body)):
        if i % 2:  # odd indexes are code
            continue
        for match in _GROUP.finditer(segment):
            cited.update(_expand(match.group(1)))
    return cited


def _check_sources(sources):
    problems, numbers, urls = [], set(), {}
    for position, entry in enumerate(sources, start=1):
        if not isinstance(entry, dict):
            problems.append(f"sources[{position}] is not an object")
            continue
        n, url, family = entry.get("n"), entry.get("url"), entry.get("source")
        label = f"source [{n}]" if isinstance(n, int) and not isinstance(n, bool) else f"sources[{position}]"
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"{label}: n must be an integer, got {n!r}")
        elif n in numbers:
            problems.append(f"{label}: number n={n} is used by more than one source")
        else:
            numbers.add(n)
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"{label}: url must start with http:// or https://, got {url!r}")
            continue
        if url in urls:
            problems.append(f"{label}: duplicate url {url} (also used by {urls[url]})")
        urls.setdefault(url, label)
        if family not in FAMILIES:
            problems.append(f"{label}: source must be one of {', '.join(FAMILIES)}, got {family!r}")
        elif family in FAMILY_PREFIX and not url.startswith(FAMILY_PREFIX[family]):
            problems.append(f"{label}: labelled {family!r} but the url does not start with {FAMILY_PREFIX[family]}")
    return problems


def _check_references(reference_text, sources_by_n):
    """The References section: exactly one '[n] ... URL' line per source, URL equal to the source's url."""
    problems, seen = [], {}
    for raw in reference_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        head = _REF_LINE.match(line)
        if not head:
            problems.append(f"References: line does not start with [n]: {line[:80]!r}")
            continue
        n = int(head.group(1))
        seen[n] = seen.get(n, 0) + 1
        if n not in sources_by_n:
            problems.append(f"References: [{n}] is not a source in sources.json")
            continue
        urls = _URL.findall(line)
        if len(urls) != 1:
            problems.append(f"References: [{n}] must contain exactly one URL, found {len(urls)} "
                            f"(one line per source, never bundle sources under one number)")
        elif urls[0] != sources_by_n[n] and urls[0].rstrip(_TRAILING) != sources_by_n[n]:
            problems.append(f"References: [{n}] URL {urls[0]} differs from sources.json ({sources_by_n[n]})")
    for n, count in sorted(seen.items()):
        if count > 1:
            problems.append(f"References: [{n}] appears on {count} lines")
    for n in sorted(sources_by_n):
        if n not in seen:
            problems.append(f"References: no line for source [{n}]")
    return problems


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]
    problems = _check_sources(sources)
    sources_by_n = {s["n"]: s["url"] for s in sources
                    if isinstance(s, dict) and isinstance(s.get("n"), int) and isinstance(s.get("url"), str)}

    headings = list(_REF_HEADING.finditer(report_text))
    if headings:
        body, reference_text = report_text[:headings[-1].start()], report_text[headings[-1].end():]
    else:
        body, reference_text = report_text, ""
        problems.append("report has no '## References' heading")

    cited = _cited_numbers(body)
    problems += [f"[{n}] cited but missing from sources.json" for n in sorted(cited - set(sources_by_n))]
    problems += [f"source [{n}] never cited in the report body" for n in sorted(set(sources_by_n) - cited)]
    if headings:
        problems += _check_references(reference_text, sources_by_n)
    return problems


def check_structure(report_text):
    """REPORT_TEMPLATE.md layout: a title, then TL;DR, Background, 3 to 6 themes, Trends and open problems."""
    body = _REF_HEADING.split(report_text)[0]
    body = _CODE.sub("", body)
    problems = [] if re.search(r"(?m)\A\s*# \S", body) else ["the report must start with a '# <title>' line"]
    headings = re.findall(r"(?m)^##[ \t]+(.+?)[ \t]*$", body)
    expected = [("TL;DR", 0), ("Background", 1), ("Trends and open problems", -1)]
    for name, position in expected:
        if name not in headings:
            problems.append(f"missing the exact heading '## {name}'")
        elif headings.index(name) != (position if position >= 0 else len(headings) - 1):
            problems.append(f"'## {name}' is in the wrong place (order: TL;DR, Background, themes, Trends and open problems)")
    themes = [h for h in headings if h not in {name for name, _ in expected}]
    if not 3 <= len(themes) <= 6:
        problems.append(f"the report needs 3 to 6 theme sections between Background and Trends, found {len(themes)}")
    return problems


MIN_FAMILIES = 3        # RUBRIC 2.2: at least 3 of the 4 source families
MIN_BODY_WORDS = 1000   # REPORT_TEMPLATE.md asks for a synthesis, not a stub


def check_depth(report_text, sources):
    """The rubric floors a report must reach (command line only, like check_structure): source families and length."""
    families = {s.get("source") for s in sources if isinstance(s, dict)} & set(FAMILIES) if isinstance(sources, list) else set()
    problems = []
    if len(families) < MIN_FAMILIES:
        problems.append(f"only {len(families)} source families ({', '.join(sorted(families)) or 'none'}): the report must cite "
                        f"sources of at least {MIN_FAMILIES} of {', '.join(FAMILIES)}")
    untitled = [s.get("n") for s in sources if isinstance(s, dict) and not str(s.get("title") or "").strip()] if isinstance(sources, list) else []
    if untitled:
        problems.append(f"sources without a title (the References would read 'Untitled'): {untitled[:10]}: copy each title from the notes")
    words = len(_REF_HEADING.split(report_text)[0].split())
    if words < MIN_BODY_WORDS:
        problems.append(f"the report body has {words} words, the minimum is {MIN_BODY_WORDS}: develop the themes")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources) + check_structure(report) + check_depth(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
