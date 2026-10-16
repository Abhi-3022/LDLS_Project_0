#!/usr/bin/env python3
"""
Stage-1 parser for the HTML domain.

Turns ONE HTML page into the project's JSON schema. It does NOT touch the LLM —
it only produces the structured JSON that the C++ driver later prefills.

TWO MODES
---------
default (full DOM):
    Keeps the whole page. Splits sequences at the direct children of <body>.
    Good for small, hand-structured pages (like sample_pages/pets.html).

--main-only  (USE THIS FOR REAL WEB PAGES, e.g. Wikipedia):
    1. Finds the main article container and throws away page chrome
       (nav bars, side bars, footers, info/nav boxes, references, scripts).
    2. Splits the article into one sequence PER SECTION, using the <h2>
       headings as boundaries (Intro / Description / Diet / ...).
    This turns a 198k-token, 96%-markup Wikipedia dump into a small,
    clean tree of real sections — the subtrees the project needs.

Schema field mapping (both modes):
    seq_ids   -> every node and segment carries one
    node_type -> nodes[].node_type   (segments reference node_id)
    parent    -> nodes[].parent
    tokens    -> segments[].text  (the C++ DRIVER tokenizes these with Llama's
                 tokenizer, so token ids always match the model)
    positions -> assigned by the DRIVER as it tokenizes segments in order
                 (pos_start_est is an inspection-only estimate)

Token counts here are a cheap PROXY (word/punct regex); real ids/positions come
from the driver.

Usage:
    python parser.py sample_pages/pets.html -o pets.json --pretty
    python parser.py sample_pages/dog.html  -o dog.json  --pretty --main-only
"""

import argparse
import json
import re
import sys

from bs4 import BeautifulSoup, Tag, NavigableString, Comment

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"}

SPLIT_TAGS = {"section", "article", "main", "aside", "div", "script", "style"}
SHARED_TAGS = {"header", "nav", "footer"}

# ---- cleaning rules for --main-only -----------------------------------------
JUNK_TAGS = {"script", "style", "nav", "footer", "aside", "form",
             "noscript", "link", "meta", "sup", "figure", "img", "table"}
# any element whose class/id contains one of these substrings is dropped
JUNK_MARKERS = ("navbox", "infobox", "metadata", "mw-editsection", "mw-jump",
                "catlinks", "printfooter", "hatnote", "toc", "reference",
                "reflist", "sistersitebox", "navigation", "vertical-navbox",
                "mw-indicators", "shortdescription", "noprint", "sidebar")
# candidate containers for the real article content; we pick whichever
# actually has the most text (a page can contain an empty decoy container).
MAIN_SELECTORS = [
    ("div", {"class": "mw-parser-output"}),   # Wikipedia / MediaWiki
    ("main", {}),
    ("div", {"id": "mw-content-text"}),
    ("div", {"id": "bodyContent"}),
    ("article", {}),
    ("div", {"id": "content"}),
]

# the meaningful content blocks we group into sections
CONTENT_TAGS = ["h2", "h3", "p", "ul", "ol", "dl", "pre", "blockquote"]

# in --main-only we keep only these structural tags (no attributes) and unwrap
# everything else (div/span/a/b/i...) down to its text — this strips the
# attribute noise that makes raw HTML 96% markup and blows the token budget.
BLOCK_KEEP = {"p", "ul", "ol", "li", "dl", "dt", "dd",
              "h2", "h3", "h4", "pre", "blockquote"}

# tail sections that are link lists / metadata, not article content -> dropped
DROP_HEADINGS = ("reference", "bibliograph", "sources", "notes", "see also",
                 "related pages", "other websites", "external link",
                 "further reading", "more reading", "photo gallery",
                 "gallery", "citations", "footnotes")


def is_drop_heading(label: str) -> bool:
    low = label.lower()
    return any(p in low for p in DROP_HEADINGS)


def est_tokens(s: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", s or ""))


def open_tag(tag: Tag) -> str:
    attrs = ""
    for k, v in tag.attrs.items():
        val = " ".join(v) if isinstance(v, list) else v
        attrs += f' {k}="{val}"'
    return f"<{tag.name}{attrs}>"


def classes_and_id(tag: Tag) -> str:
    return (" ".join(tag.get("class", [])) + " " + (tag.get("id") or "")).lower()


# =============================================================================
# FULL-DOM MODE (unchanged behaviour)
# =============================================================================
class FullDomParser:
    def __init__(self, merge_min):
        self.merge_min = merge_min
        self.nodes, self.tag_of, self.children = [], {}, {}
        self.seq_label, self.next_seq = {0: "shared / global context"}, 0

    def build(self, tag, parent_id, depth):
        nid = len(self.nodes)
        self.nodes.append({"node_id": nid, "node_type": tag.name, "parent": parent_id,
                           "depth": depth, "seq_id": None, "_tok": 0})
        self.tag_of[nid] = tag
        self.children[nid] = []
        tok = est_tokens(open_tag(tag))
        for child in tag.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                tok += est_tokens(str(child))
            elif isinstance(child, Tag):
                cid = self.build(child, nid, depth + 1)
                self.children[nid].append(cid)
                tok += self.nodes[cid]["_tok"]
        if tag.name not in VOID:
            tok += est_tokens(f"</{tag.name}>")
        self.nodes[nid]["_tok"] = tok
        return nid

    def assign_seq(self, nid):
        node = self.nodes[nid]
        parent = node["parent"]
        ptype = self.nodes[parent]["node_type"] if parent >= 0 else None
        if ptype == "body":
            name = node["node_type"]
            if name in SHARED_TAGS:
                node["seq_id"] = 0
            elif name in SPLIT_TAGS and node["_tok"] >= self.merge_min:
                self.next_seq += 1
                node["seq_id"] = self.next_seq
                t = self.tag_of[nid]
                ident = t.get("id") or (".".join(t.get("class", [])) or f"node{nid}")
                self.seq_label[self.next_seq] = f"{name}#{ident}"
            else:
                node["seq_id"] = 0
        else:
            node["seq_id"] = self.nodes[parent]["seq_id"] if parent >= 0 else 0
        for cid in self.children[nid]:
            self.assign_seq(cid)

    def serialize(self, nid, segments):
        tag, node = self.tag_of[nid], self.nodes[nid]
        seq = node["seq_id"]
        is_code = tag.name in ("script", "style")
        if tag.name != "[document]":
            segments.append({"node_id": nid, "seq_id": seq,
                             "kind": "script" if is_code else "markup", "text": open_tag(tag)})
        for child in tag.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                if str(child).strip() == "":
                    continue
                segments.append({"node_id": nid, "seq_id": seq,
                                 "kind": "script" if is_code else "content", "text": str(child)})
            elif isinstance(child, Tag):
                cid = next(c for c in self.children[nid] if self.tag_of[c] is child)
                self.serialize(cid, segments)
        if tag.name != "[document]" and tag.name not in VOID:
            segments.append({"node_id": nid, "seq_id": seq, "kind": "markup", "text": f"</{tag.name}>"})

    def run(self, soup):
        root = soup.find("html") or soup
        self.build(root, -1, 0)
        self.assign_seq(0)
        segments = []
        self.serialize(0, segments)
        for n in self.nodes:
            n.pop("_tok", None)
        return self.nodes, segments, self.seq_label


# =============================================================================
# MAIN-ONLY MODE (for real web pages)
# =============================================================================
def is_heading(tag: Tag):
    """Return the heading level 2/3 if this element is (or wraps) an h2/h3."""
    if tag.name in ("h2", "h3"):
        return int(tag.name[1])
    # MediaWiki wraps headings: <div class="mw-heading"><h2>...</h2></div>
    inner = tag.find(["h2", "h3"], recursive=True) if tag.name == "div" else None
    if inner and "mw-heading" in classes_and_id(tag):
        return int(inner.name[1])
    return None


def heading_text(tag: Tag) -> str:
    t = tag.get_text(" ", strip=True)
    return re.sub(r"\s*\[\s*edit\s*\]\s*$", "", t)[:60] or "section"


def clean_main(main: Tag):
    """Strip page chrome from the main content container, in place."""
    for el in main.find_all(True):
        if not el.name:
            continue
        if el.name in JUNK_TAGS:
            el.decompose()
            continue
        marks = classes_and_id(el)
        if any(m in marks for m in JUNK_MARKERS):
            el.decompose()


def find_main(soup: BeautifulSoup) -> Tag:
    """Return the first candidate container (in priority order) that holds
    real article text. mw-parser-output / bodyContent come before <main>
    because <main> also contains the language sidebar and page chrome."""
    for name, attrs in MAIN_SELECTORS:
        best, best_len = None, 0
        for el in (soup.find_all(name, attrs) if attrs else soup.find_all(name)):
            n = len(el.get_text(" ", strip=True))
            if n > best_len:
                best, best_len = el, n
        if best is not None and best_len > 500:
            return best
    return soup.body or soup


def top_level_blocks(main: Tag):
    """Content blocks (headings, paragraphs, lists...) in document order,
    skipping ones nested inside another content block (avoids duplicates)."""
    out = []
    for el in main.find_all(CONTENT_TAGS):
        if el.find_parent(CONTENT_TAGS) is not None:
            continue
        if el.name not in ("ul", "ol") and el.get_text(strip=True) == "":
            continue
        out.append(el)
    return out


def serialize_clean(tag, seq, node_id, segments):
    """Serialize a subtree keeping only structural block tags (no attributes)
    and unwrapping everything else down to its text."""
    keep = tag.name in BLOCK_KEEP
    if keep:
        segments.append({"node_id": node_id, "seq_id": seq, "kind": "markup",
                         "text": f"<{tag.name}>"})
    for child in tag.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            txt = re.sub(r"[ \t\n]+", " ", str(child))
            if txt.strip() == "":
                continue
            segments.append({"node_id": node_id, "seq_id": seq, "kind": "content", "text": txt})
        elif isinstance(child, Tag):
            serialize_clean(child, seq, node_id, segments)   # unwraps non-kept tags
    if keep:
        segments.append({"node_id": node_id, "seq_id": seq, "kind": "markup",
                         "text": f"</{tag.name}>"})


def run_main_only(soup, merge_min, max_tokens, drop_boilerplate):
    main = find_main(soup)
    clean_main(main)

    nodes = [{"node_id": 0, "node_type": "article", "parent": -1, "depth": 0, "seq_id": 0}]
    seq_label = {0: "intro / shared"}
    segments = []
    next_seq = 0

    # node 1 = the intro block (everything before the first <h2>), seq 0
    cur_node = 1
    cur_seq = 0
    nodes.append({"node_id": 1, "node_type": "intro", "parent": 0, "depth": 1, "seq_id": 0})
    seq_tokens = {}  # node_id -> est tokens, for the merge rule
    running = [0]    # running total across kept sections, for the budget cap

    def add_block(tag, node_id, seq):
        before = len(segments)
        serialize_clean(tag, seq, node_id, segments)
        added = sum(est_tokens(s["text"]) for s in segments[before:])
        seq_tokens[node_id] = seq_tokens.get(node_id, 0) + added
        running[0] += added

    def over_budget():
        return max_tokens > 0 and running[0] >= max_tokens

    skip_section = False
    for block in top_level_blocks(main):
        if block.name == "h2":
            label = heading_text(block)
            if drop_boilerplate and is_drop_heading(label):
                skip_section = True                 # drop this whole section
                continue
            if over_budget():                       # hard stop at a section boundary
                break
            skip_section = False
            next_seq += 1
            cur_seq = next_seq
            cur_node = len(nodes)
            nodes.append({"node_id": cur_node, "node_type": "section", "parent": 0,
                          "depth": 1, "seq_id": cur_seq, "heading": label})
            seq_label[cur_seq] = label
            add_block(block, cur_node, cur_seq)      # include the heading text
            continue
        if skip_section:
            continue
        if over_budget():                            # hard stop mid-section at a block boundary
            break
        add_block(block, cur_node, cur_seq)

    # hard trim: drop trailing segments so the proxy budget is actually met
    if max_tokens > 0:
        while segments and running[0] > max_tokens:
            running[0] -= est_tokens(segments[-1]["text"])
            segments.pop()

    # drop any section node that ended up empty (budget hit right after its heading)
    used = {s["seq_id"] for s in segments}
    nodes = [n for n in nodes if n["seq_id"] in used or n["node_id"] == 0]

    # ---- merge rule: fold tiny sections back into shared (seq 0) ----
    merged = []
    for n in nodes:
        if n["node_type"] == "section" and seq_tokens.get(n["node_id"], 0) < merge_min:
            old = n["seq_id"]
            for s in segments:
                if s["node_id"] == n["node_id"]:
                    s["seq_id"] = 0
            n["seq_id"] = 0
            merged.append(seq_label.pop(old, None))
    return nodes, segments, seq_label


# =============================================================================
def finish(nodes, segments, seq_label, doc_id, mode):
    pos = 0
    for seg in segments:
        seg["n_tokens_est"] = est_tokens(seg["text"])
        seg["pos_start_est"] = pos
        pos += seg["n_tokens_est"]
    total = sum(s["n_tokens_est"] for s in segments)
    markup = sum(s["n_tokens_est"] for s in segments if s["kind"] in ("markup", "script"))
    per_seq = {}
    for s in segments:
        per_seq[s["seq_id"]] = per_seq.get(s["seq_id"], 0) + s["n_tokens_est"]
    stats = {
        "max_depth": max(n["depth"] for n in nodes),
        "n_nodes": len(nodes),
        "n_sequences": len(set(s["seq_id"] for s in segments)),
        "est_tokens_total": total,
        "est_tokens_per_seq": {str(k): v for k, v in sorted(per_seq.items())},
        "markup_or_syntax_share": round(markup / total, 3) if total else 0.0,
    }
    return {
        "doc_id": doc_id, "domain": "html", "mode": mode,
        "note": "segments[].text are tokenized by the C++ driver with Llama's "
                "tokenizer; positions assigned there. Counts here are a proxy.",
        "seq_legend": {str(k): v for k, v in sorted(seq_label.items())},
        "nodes": nodes, "segments": segments, "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser(description="HTML -> project JSON schema")
    ap.add_argument("html_file")
    ap.add_argument("-o", "--out")
    ap.add_argument("--merge-min", type=int, default=16)
    ap.add_argument("--main-only", action="store_true",
                    help="keep only the article body and split by <h2> sections (use for real web pages)")
    ap.add_argument("--max-tokens", type=int, default=1500,
                    help="main-only: proxy-token budget to stop at (0 = no cap). "
                         "Real Llama tokens run ~1.3x higher, so 1500 proxy ~= 2000 real.")
    ap.add_argument("--keep-boilerplate", action="store_true",
                    help="main-only: keep References/Other websites/etc. (dropped by default)")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    with open(args.html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    doc_id = args.html_file.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if args.main_only:
        nodes, segments, seq_label = run_main_only(
            soup, args.merge_min, args.max_tokens, not args.keep_boilerplate)
        mode = "main-only"
    else:
        nodes, segments, seq_label = FullDomParser(args.merge_min).run(soup)
        mode = "full-dom"
    result = finish(nodes, segments, seq_label, doc_id, mode)

    text = json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        s = result["stats"]
        print(f"wrote {args.out}  [{mode}]", file=sys.stderr)
        print(f"  sequences={s['n_sequences']}  depth={s['max_depth']}  "
              f"est_tokens={s['est_tokens_total']}  markup_share={s['markup_or_syntax_share']}", file=sys.stderr)
        print(f"  seq sizes: {s['est_tokens_per_seq']}", file=sys.stderr)
        print(f"  legend: {result['seq_legend']}", file=sys.stderr)
        approx_real = int(s["est_tokens_total"] * 1.3)
        print(f"  ~{approx_real} real Llama tokens (est); the driver enforces the true count via n_ctx",
              file=sys.stderr)
        if approx_real > 2000:
            print("  WARNING: likely over 2000 real tokens — lower --max-tokens or pick a shorter page.",
                  file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
