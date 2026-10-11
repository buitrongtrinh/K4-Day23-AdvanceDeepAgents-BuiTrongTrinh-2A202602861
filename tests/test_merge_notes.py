import merge_notes
from merge_notes import merge, parse_notes

NOTES_A = """## World Models
- id: 1803.10122
- url: https://arxiv.org/abs/1803.10122
- date: 2018-03-27
- source: arxiv
- notes:
  - a VAE and an RNN learn a compact model of the environment

## 2. DreamerV3
- id: 2301.04104
- url: https://arxiv.org/abs/2301.04104
- date: 2023-01-10
- source: hf-search
- notes: - labelled wrongly: an arXiv url found by hf_search_papers cannot be hf-search
"""
NOTES_B = """## **Genie 2** blog
- **id**:
- **url**: [genie](https://deepmind.google/blog/genie-2/)
- **date**: n.d.
- **source**: web

## World Models again (repeat)
- url: https://www.arxiv.org/abs/1803.10122/
- source: arxiv

## Broken block
- source: web

## Bad family
- url: https://example.org/x
- source: twitter
"""


def test_parse_notes_is_tolerant_about_markdown():
    first = parse_notes(NOTES_B)[0]
    assert first == {"title": "Genie 2 blog", "id": "", "url": "https://deepmind.google/blog/genie-2/", "date": "n.d.", "source": "web"}
    assert parse_notes(NOTES_A)[1]["title"] == "DreamerV3"          # the "2. " numbering of a heading is dropped


def test_merge_numbers_dedups_and_relabels():
    sources, problems = merge({"a.md": NOTES_A, "b.md": NOTES_B})
    assert [s["url"] for s in sources] == ["https://arxiv.org/abs/1803.10122", "https://arxiv.org/abs/2301.04104",
                                           "https://deepmind.google/blog/genie-2/"]       # the repeated url is dropped
    assert [s["n"] for s in sources] == [1, 2, 3]
    assert sources[0]["source"] == "arxiv" and sources[2]["date"] == "n.d."
    assert sources[1]["source"] == "web"        # an arXiv abstract under an hf-* label: the label does not fit the url
    assert any("no usable url" in p for p in problems) and any("twitter" in p for p in problems)


def test_merged_sources_pass_the_validator_shape():
    from check_citations import check_depth
    sources, _ = merge({"a.md": NOTES_A, "b.md": NOTES_B})
    assert not any("title" in p for p in check_depth("w " * 1000, sources))      # every source has a title


def test_cli_writes_master_and_prints_the_list(tmp_path, capsys):
    (tmp_path / "01-x.md").write_text(NOTES_A)
    master = tmp_path / "all.json"
    assert merge_notes.main(["m", str(tmp_path), str(master)]) == 0
    out = capsys.readouterr().out
    assert "MERGED: 2 sources" in out and "[1] arxiv" in out and "missing: hf-daily, hf-search" in out
    assert master.exists()
    assert merge_notes.main(["m", str(tmp_path / "empty"), str(master)]) == 1
