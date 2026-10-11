import copy

from check_citations import check

SOURCES = [
    {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025-01-02", "source": "arxiv"},
    {"n": 2, "id": "2503.00002", "url": "https://huggingface.co/papers/2503.00002", "title": "B", "date": "2025-03-04", "source": "hf-search"},
    {"n": 3, "id": "", "url": "https://example.org/page", "title": "C", "date": "n.d.", "source": "web"},
]
REPORT = """# Survey

## TL;DR
- First finding [1].
- Second finding [2][3].

## References
[1] A. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)
[2] B. hf-search. https://huggingface.co/papers/2503.00002 (2025-03-04)
[3] C. web. https://example.org/page (n.d.)
"""


def sources():
    return copy.deepcopy(SOURCES)


def test_valid_report_is_ok():
    assert check(REPORT, sources()) == []


def test_empty_sources():
    assert check(REPORT, []) == ["no sources in sources.json"]


def test_source_entry_rules():
    bad = sources()
    bad[0]["n"] = "1"
    bad[1]["url"] = "ftp://x/y"
    bad[2]["url"] = bad[0]["url"]
    problems = "\n".join(check(REPORT, bad))
    assert "n must be an integer" in problems
    assert "url must start with" in problems
    assert "duplicate url" in problems


def test_missing_references_heading():
    problems = check(REPORT.split("## References")[0], sources())
    assert any("no '## References' heading" in p for p in problems)


def test_cited_but_missing_and_never_cited():
    report = REPORT.replace("[2][3]", "[2][9]")
    problems = check(report, sources())
    assert "[9] cited but missing from sources.json" in problems
    assert "source [3] never cited in the report body" in problems


def test_numbers_in_references_do_not_count_as_citations():
    report = REPORT.replace("- Second finding [2][3].", "- Second finding [2].")
    assert "source [3] never cited in the report body" in check(report, sources())


def test_grouped_citations_are_expanded():
    report = REPORT.replace("[1].", "[1, 2].").replace("[2][3]", "[1-3]")
    assert check(report, sources()) == []


def test_code_and_markdown_links_are_not_citations():
    report = REPORT.replace("- First finding [1].", "- First finding [1] with `x[7]`, a [4](http://l) link and\n```\n[8]\n```")
    assert check(report, sources()) == []


def test_reference_line_bundling_several_sources():
    report = REPORT.replace("[3] C. web. https://example.org/page (n.d.)",
                            "[3] C. web. https://example.org/page; D. https://example.org/other (n.d.)")
    assert any("exactly one URL" in p for p in check(report, sources()))


def test_reference_url_must_match_sources():
    report = REPORT.replace("https://example.org/page", "https://example.org/elsewhere")
    assert any("differs from sources.json" in p for p in check(report, sources()))


def test_reference_lines_missing_duplicated_or_unknown():
    report = REPORT.replace("[3] C. web. https://example.org/page (n.d.)\n", "")
    assert "References: no line for source [3]" in check(report, sources())
    report = REPORT + "[1] A again. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)\n"
    assert "References: [1] appears on 2 lines" in check(report, sources())
    report = REPORT + "[7] X. web. https://example.org/x (n.d.)\n"
    assert "References: [7] is not a source in sources.json" in check(report, sources())


def test_family_label_must_match_url():
    bad = sources()
    bad[0]["source"] = "hf-daily"
    assert any("labelled 'hf-daily'" in p for p in check(REPORT, bad))
    bad = sources()
    bad[2]["source"] = "twitter"
    assert any("source must be one of" in p for p in check(REPORT, bad))
    bad = sources()
    bad[2]["url"], bad[2]["source"] = "https://arxiv.org/pdf/2001.08361", "web"   # an arXiv page found by web_search
    report = REPORT.replace("https://example.org/page", "https://arxiv.org/pdf/2001.08361")
    assert check(report, bad) == []


# ---- layout (command line only: check() keeps the pure citation rules) ----
from check_citations import check_structure  # noqa: E402

LAYOUT = "# T\n\n## TL;DR\n- a [1]\n\n## Background\nb\n\n## One\nx\n\n## Two\nx\n\n## Three\nx\n\n## Trends and open problems\nt\n\n## References\n[1] A. arxiv. https://arxiv.org/abs/1 (n.d.)\n"


def test_structure_ok():
    assert check_structure(LAYOUT) == []


def test_structure_problems():
    assert any("Trends and open problems" in p for p in check_structure(LAYOUT.replace("open problems", "Open Problems")))
    assert any("TL;DR" in p for p in check_structure(LAYOUT.replace("## TL;DR", "## Summary")))
    assert any("3 to 6 theme" in p for p in check_structure(LAYOUT.replace("## Three\nx\n\n", "")))
    assert any("start with" in p for p in check_structure(LAYOUT.replace("# T\n", "intro\n")))
    swapped = LAYOUT.replace("## TL;DR\n- a [1]\n\n## Background\nb\n\n", "## Background\nb\n\n## TL;DR\n- a [1]\n\n")
    assert any("wrong place" in p for p in check_structure(swapped))


def test_check_itself_ignores_layout():
    assert check("no headings [1]\n\n## References\n[1] A. arxiv. https://arxiv.org/abs/1 (n.d.)\n",
                 [{"n": 1, "url": "https://arxiv.org/abs/1", "source": "arxiv"}]) == []


# ---- floors of the rubric (command line only) ----
from check_citations import check_depth  # noqa: E402


def test_depth_floors():
    body = "word " * 1000
    three = [{"source": s, "title": "T"} for s in ("arxiv", "hf-search", "web")]
    assert check_depth(body + "\n## References\n", three) == []
    assert any("2 source families" in p for p in check_depth(body, three[1:]))
    assert any("body has 5 words" in p for p in check_depth("a b c d e\n## References\nx" * 1, three))
    assert any("0 source families" in p for p in check_depth(body, "not a list"))
    assert any("2 source families" in p for p in check_depth(body, [{"source": "hf-daily", "title": "T"}, {"source": "hf-search", "title": "T"}, {"source": "hf-search", "title": "T"}]))


def test_depth_requires_a_title_for_every_source():
    body = "word " * 1000
    full = [{"n": 1, "source": s, "title": "T"} for s in ("arxiv", "hf-search", "web")]
    assert check_depth(body, full) == []
    untitled = [dict(full[0]), {"n": 2, "source": "hf-search"}, {"n": 3, "source": "web", "title": "  "}]
    assert any("without a title" in p and "[2, 3]" in p for p in check_depth(body, untitled))
