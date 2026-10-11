"""merge_notes.py - builds sources_all.json from the researchers' notes.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file; the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/merge_notes.py [notes_dir] [sources_all.json]

An LLM that copies dozens of URLs by hand mistypes or invents some. This script copies them by code: it reads every notes
file (the format is NOTE_FORMAT of agents.py), drops repeated urls, numbers the sources from 1 and prints the numbered list
the lead cites from. It also repairs what is mechanical: a source labelled arxiv / hf-* whose url is not an arXiv abstract /
Hugging Face paper page is relabelled "web" (the label is the tool that returned it, the url must fit the label).
"""
import glob
import json
import os
import re
import sys

NOTES_DIR = "/tmp/work/research/notes"
MASTER = "/tmp/work/research/sources_all.json"
FAMILIES = ("arxiv", "hf-daily", "hf-search", "web")
ARXIV_PREFIX, HF_PREFIX = "https://arxiv.org/abs/", "https://huggingface.co/papers/"

_FIELD = re.compile(r"^\s*(?:[-*]\s*)?\**(id|url|date|source)\**\s*:\s*(.*?)\s*$", re.I)
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _clean(value):
    link = _LINK.search(value)
    value = link.group(1) if link else value
    return value.strip().strip("`<>*\"' ").strip()


def parse_notes(text):
    """The blocks of one notes file: [{title, id, url, date, source}]. A block starts at a '## ' heading."""
    blocks, current = [], None
    for line in text.splitlines():
        if line.startswith("## "):
            title = re.sub(r"[*`]", "", re.sub(r"^\d+[.)]\s*", "", line[3:].strip())).strip()
            current = {"title": title, "id": "", "url": "", "date": "", "source": ""}
            blocks.append(current)
            continue
        field = _FIELD.match(line)
        if current is not None and field:
            current[field.group(1).lower()] = _clean(field.group(2))
    return blocks


def _key(url):
    return re.sub(r"^https?://(www\.)?", "", url).rstrip("/").lower()


def merge(notes_texts):
    """notes_texts: {file name: text}. Returns (sources, problems)."""
    sources, problems, seen = [], [], set()
    for name in sorted(notes_texts):
        for block in parse_notes(notes_texts[name]):
            label = f"{os.path.basename(name)}: '{block['title'][:50]}'"
            if not re.match(r"https?://\S+$", block["url"]):
                problems.append(f"{label}: no usable url")
                continue
            if _key(block["url"]) in seen:
                continue
            seen.add(_key(block["url"]))
            family = block["source"].lower()
            if family not in FAMILIES:
                problems.append(f"{label}: source '{block['source']}' is not one of {', '.join(FAMILIES)}")
                continue
            if (family == "arxiv" and not block["url"].startswith(ARXIV_PREFIX)) or (
                    family.startswith("hf-") and not block["url"].startswith(HF_PREFIX)):
                family = "web"
            if not block["title"]:
                problems.append(f"{label}: no title")
                continue
            sources.append({"n": len(sources) + 1, "id": block["id"], "url": block["url"], "title": block["title"],
                            "date": block["date"] or "n.d.", "source": family})
    return sources, problems


def main(argv):
    notes_dir = argv[1] if len(argv) > 1 else NOTES_DIR
    master = argv[2] if len(argv) > 2 else MASTER
    texts = {}
    for path in glob.glob(os.path.join(notes_dir, "*.md")):
        with open(path, encoding="utf-8") as f:
            texts[path] = f.read()
    if not texts:
        print(f"no notes files in {notes_dir}")
        return 1
    sources, problems = merge(texts)
    if not sources:
        print("no usable source in the notes:\n" + "\n".join(problems))
        return 1
    with open(master, "w", encoding="utf-8") as f:
        json.dump(sources, f, ensure_ascii=False, indent=2)
    families = {}
    for s in sources:
        families[s["source"]] = families.get(s["source"], 0) + 1
    print(f"MERGED: {len(sources)} sources from {len(texts)} notes files -> {master}")
    print("families: " + ", ".join(f"{k}={v}" for k, v in sorted(families.items())) +
          (f"  (missing: {', '.join(f for f in FAMILIES if f not in families)})" if len(families) < len(FAMILIES) else ""))
    if problems:
        print("skipped blocks (fix them in the notes, or ignore):\n  " + "\n  ".join(problems))
    print("n | source | date | title")
    for s in sources:
        print(f"[{s['n']}] {s['source']} | {s['date']} | {s['title'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
