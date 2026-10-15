#!/usr/bin/env python3
"""
Stage-1 parser for the LaTeX domain. Emits the SAME JSON schema as parser.py,
so the exact same C++ driver consumes it. Only this file changes per domain.

TWO MODES
---------
--mode sections  (default)  : clean tree. Split sequences at \\section boundaries.
                              Natural for "questions target one section".
--mode cites                : cross-reference version (the project's domain-5
                              approximation "group by cited block"). Each labeled
                              block (theorem/lemma/definition/proof/...) is its own
                              sequence, and a dependency graph `uses` records which
                              blocks each block \\ref's. A proof depends on the few
                              results it cites, not the whole paper.

Both strip the preamble (everything before \\begin{document}) and comments, keep
section/environment structure plus the math, merge tiny blocks, and cap tokens.

Schema field mapping (same as HTML):
    seq_ids   -> node/segment seq_id
    node_type -> nodes[].node_type (section / theorem / proof / intro / ...)
    parent    -> nodes[].parent
    tokens    -> segments[].text (the DRIVER tokenizes with Llama's tokenizer)
    positions -> assigned by the driver
`cites` mode additionally emits top-level `uses`: {seq_id: [cited seq_ids]}.

Usage:
    python parser_latex.py sample_pages/note.tex -o note.json --pretty
    python parser_latex.py sample_pages/note.tex -o note_cites.json --pretty --mode cites
"""

import argparse
import json
import os
import re
import sys

from pylatexenc.latexwalker import (
    LatexWalker, LatexCharsNode, LatexMacroNode, LatexEnvironmentNode,
    LatexGroupNode, LatexCommentNode, LatexMathNode, LatexSpecialsNode,
)

# environments that count as "labeled blocks" in --mode cites
BLOCK_ENVS = {"theorem", "lemma", "definition", "proposition", "corollary",
              "proof", "remark", "example", "claim", "fact", "conjecture"}
# macros that reference another labeled block
REF_MACROS = {"ref", "eqref", "cref", "Cref", "autoref", "pageref"}
# structural section commands (section boundaries in --mode sections).
# subsection is included so papers with a few very large sections still yield
# several evictable subtrees under the 2k budget.
SECTION_MACROS = {"section", "subsection"}
# macros whose text we keep but whose name we don't emit (pure metadata)
DROP_MACROS = {"label", "maketitle", "title", "author", "date", "newtheorem",
               "usepackage", "documentclass"}


def est_tokens(s: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", s or ""))


# ---- multi-file support: inline \input / \include and drop comments ---------
def strip_comments(tex: str) -> str:
    # remove LaTeX line comments ( % to end of line ), but not escaped \%
    return re.sub(r"(?<!\\)%.*", "", tex)


def flatten_inputs(tex: str, base_dir: str, seen=None) -> str:
    """Recursively replace \\input{f} / \\include{f} with the file's contents.
    arXiv papers split content across many files; this inlines them so the
    main .tex parses into the whole paper. Missing files are skipped."""
    if seen is None:
        seen = set()

    def repl(m):
        rel = m.group(1).strip()
        path = os.path.join(base_dir, rel)
        if not os.path.splitext(path)[1]:
            path += ".tex"
        rp = os.path.abspath(path)
        if rp in seen or not os.path.isfile(rp):
            return ""                                 # cycle or missing -> drop
        seen.add(rp)
        with open(rp, "r", encoding="utf-8", errors="ignore") as f:
            sub = strip_comments(f.read())
        return flatten_inputs(sub, os.path.dirname(rp), seen)

    return re.sub(r"\\(?:input|include)\s*\{([^}]+)\}", repl, tex)


# ---- small node helpers -----------------------------------------------------
def node_args(node):
    out = []
    nad = getattr(node, "nodeargd", None)
    if nad and getattr(nad, "argnlist", None):
        out = [a for a in nad.argnlist if a is not None]
    return out


def get_text(node) -> str:
    """Plain text of a node/subtree (for titles, labels, refs)."""
    if node is None:
        return ""
    if isinstance(node, LatexCharsNode):
        return node.chars
    if isinstance(node, LatexGroupNode):
        return "".join(get_text(n) for n in node.nodelist)
    if isinstance(node, LatexEnvironmentNode):
        return "".join(get_text(n) for n in node.nodelist)
    if isinstance(node, LatexMacroNode):
        return "".join(get_text(a) for a in node_args(node))
    if isinstance(node, LatexMathNode):
        return node.latex_verbatim()
    return ""


def descendants(node):
    """Yield node and all descendants (for finding labels/refs)."""
    yield node
    kids = []
    if isinstance(node, LatexGroupNode):
        kids = node.nodelist
    elif isinstance(node, LatexEnvironmentNode):
        kids = list(node.nodelist) + node_args(node)   # body AND env args, e.g. [Proof of \cref{...}]
    elif isinstance(node, LatexMacroNode):
        kids = node_args(node)
    elif isinstance(node, LatexMathNode):
        kids = getattr(node, "nodelist", []) or []
    for k in kids:
        if k is not None:
            yield from descendants(k)


def find_label(block):
    for n in descendants(block):
        if isinstance(n, LatexMacroNode) and n.macroname == "label":
            args = node_args(n)
            if args:
                return get_text(args[0]).strip()
    return None


def refs_in(block):
    out = []
    for n in descendants(block):
        if isinstance(n, LatexMacroNode) and n.macroname in REF_MACROS:
            args = node_args(n)
            if args:
                out.append(get_text(args[0]).strip())
    return out


# ---- serialization: node subtree -> clean segments --------------------------
def serialize(node, seq, nid, segments):
    if isinstance(node, LatexCommentNode):
        return
    if isinstance(node, LatexCharsNode):
        txt = re.sub(r"[ \t\n]+", " ", node.chars)
        if txt.strip():
            segments.append({"node_id": nid, "seq_id": seq, "kind": "content", "text": txt})
        return
    if isinstance(node, LatexMathNode):
        segments.append({"node_id": nid, "seq_id": seq, "kind": "content", "text": node.latex_verbatim()})
        return
    if isinstance(node, LatexGroupNode):
        for n in node.nodelist:
            serialize(n, seq, nid, segments)
        return
    if isinstance(node, LatexEnvironmentNode):
        env = node.environmentname
        segments.append({"node_id": nid, "seq_id": seq, "kind": "markup", "text": f"\\begin{{{env}}}"})
        for n in node.nodelist:
            serialize(n, seq, nid, segments)
        segments.append({"node_id": nid, "seq_id": seq, "kind": "markup", "text": f"\\end{{{env}}}"})
        return
    if isinstance(node, LatexMacroNode):
        name = node.macroname
        if name in DROP_MACROS:               # metadata: keep args' text, drop the name
            for a in node_args(node):
                serialize(a, seq, nid, segments)
            return
        segments.append({"node_id": nid, "seq_id": seq, "kind": "markup", "text": f"\\{name}"})
        for a in node_args(node):
            serialize(a, seq, nid, segments)
        return
    if isinstance(node, LatexSpecialsNode):
        segments.append({"node_id": nid, "seq_id": seq, "kind": "markup", "text": node.specials_chars})
        return


# ---- find the document body -------------------------------------------------
def document_body(nodes):
    for n in nodes:
        if isinstance(n, LatexEnvironmentNode) and n.environmentname == "document":
            return n.nodelist
    return nodes   # a fragment with no \begin{document}


# =============================================================================
# MODE: sections
# =============================================================================
def run_sections(body, merge_min, max_tokens, max_shared=0):
    nodes = [{"node_id": 0, "node_type": "article", "parent": -1, "depth": 0, "seq_id": 0}]
    nodes.append({"node_id": 1, "node_type": "intro", "parent": 0, "depth": 1, "seq_id": 0})
    seq_label = {0: "intro / shared"}
    segments = []
    next_seq, cur_seq, cur_node = 0, 0, 1
    running = [0]
    seq0 = [0]            # tokens placed in the shared/intro sequence so far
    seq_tokens = {}

    def add(node, node_id, seq):
        before = len(segments)
        serialize(node, seq, node_id, segments)
        added = sum(est_tokens(s["text"]) for s in segments[before:])
        seq_tokens[node_id] = seq_tokens.get(node_id, 0) + added
        running[0] += added
        if seq == 0:
            seq0[0] += added

    for node in body:
        if isinstance(node, LatexMacroNode) and node.macroname in SECTION_MACROS:
            if max_tokens > 0 and running[0] >= max_tokens:
                break
            title = get_text(node_args(node)[0]).strip() if node_args(node) else "section"
            next_seq += 1
            cur_seq = next_seq
            cur_node = len(nodes)
            nodes.append({"node_id": cur_node, "node_type": "section", "parent": 0,
                          "depth": 1, "seq_id": cur_seq, "heading": title})
            seq_label[cur_seq] = title
            segments.append({"node_id": cur_node, "seq_id": cur_seq, "kind": "markup", "text": "\\section"})
            segments.append({"node_id": cur_node, "seq_id": cur_seq, "kind": "content", "text": title})
            continue
        if max_tokens > 0 and running[0] >= max_tokens:
            break
        # cap the shared/intro sequence so a long abstract+intro can't eat the whole
        # budget before any \section is reached (papers otherwise come out as 1 seq).
        if cur_seq == 0 and max_shared > 0 and seq0[0] >= max_shared:
            continue
        add(node, cur_node, cur_seq)

    _trim(segments, running, max_tokens)
    nodes = _drop_empty(nodes, segments)
    return nodes, segments, seq_label, None


# =============================================================================
# MODE: cites  (group by labeled block + dependency graph)
# =============================================================================
def run_cites(body, merge_min, max_tokens):
    # pass 1: collect labeled blocks in document order, assign a seq to each
    blocks = []   # (node, seq, label)
    def collect(nodes):
        for n in nodes:
            if isinstance(n, LatexEnvironmentNode):
                if n.environmentname in BLOCK_ENVS:
                    blocks.append(n)
                else:
                    collect(n.nodelist)   # descend non-block envs
    collect(body)

    block_seq = {}     # id(node) -> seq
    label2seq = {}
    nodes = [{"node_id": 0, "node_type": "article", "parent": -1, "depth": 0, "seq_id": 0},
             {"node_id": 1, "node_type": "shared", "parent": 0, "depth": 1, "seq_id": 0}]
    seq_label = {0: "shared / prose"}
    for i, b in enumerate(blocks, start=1):
        block_seq[id(b)] = i
        lab = find_label(b) or f"{b.environmentname}{i}"
        label2seq[lab] = i
        nodes.append({"node_id": len(nodes), "node_type": b.environmentname,
                      "parent": 0, "depth": 1, "seq_id": i, "label": lab})
        seq_label[i] = f"{b.environmentname} ({lab})"
    nid_of_seq = {n["seq_id"]: n["node_id"] for n in nodes if n["node_id"] >= 1}

    segments = []
    uses = {}
    running = [0]

    def add(node, node_id, seq):
        before = len(segments)
        serialize(node, seq, node_id, segments)
        running[0] += sum(est_tokens(s["text"]) for s in segments[before:])

    # cites mode keeps only the BLOCKS (theorems/proofs) + section headings as
    # light shared context, and SKIPS the bulk prose — the dependency structure
    # is the point, and on a real paper the prose would eat the whole budget.
    def walk(nodes_):
        for n in nodes_:
            if max_tokens > 0 and running[0] >= max_tokens:
                return
            if isinstance(n, LatexEnvironmentNode) and id(n) in block_seq:
                seq = block_seq[id(n)]
                add(n, nid_of_seq[seq], seq)
                tgt = [label2seq[r] for r in refs_in(n) if r in label2seq]
                if tgt:
                    uses[seq] = sorted(set(tgt))
            elif isinstance(n, LatexEnvironmentNode):
                walk(n.nodelist)                     # descend to find nested blocks
            elif isinstance(n, LatexMacroNode) and n.macroname in SECTION_MACROS:
                title = get_text(node_args(n)[0]).strip() if node_args(n) else "section"
                segments.append({"node_id": 1, "seq_id": 0, "kind": "markup", "text": "\\section"})
                segments.append({"node_id": 1, "seq_id": 0, "kind": "content", "text": title})
                running[0] += est_tokens(title) + 1
            # else: skip body prose in cites mode
    walk(body)

    _trim(segments, running, max_tokens)
    nodes = _drop_empty(nodes, segments)
    uses = {k: v for k, v in uses.items() if any(s["seq_id"] == k for s in segments)}
    return nodes, segments, seq_label, uses


# ---- shared budget helpers --------------------------------------------------
def _trim(segments, running, max_tokens):
    if max_tokens > 0:
        while segments and running[0] > max_tokens:
            running[0] -= est_tokens(segments[-1]["text"])
            segments.pop()


def _drop_empty(nodes, segments):
    used = {s["seq_id"] for s in segments}
    return [n for n in nodes if n["seq_id"] in used or n["node_id"] == 0]


# ---- assemble + stats -------------------------------------------------------
def finish(nodes, segments, seq_label, uses, doc_id, mode):
    pos = 0
    for seg in segments:
        seg["n_tokens_est"] = est_tokens(seg["text"])
        seg["pos_start_est"] = pos
        pos += seg["n_tokens_est"]
    total = sum(s["n_tokens_est"] for s in segments)
    markup = sum(s["n_tokens_est"] for s in segments if s["kind"] == "markup")
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
    out = {
        "doc_id": doc_id, "domain": "latex", "mode": mode,
        "note": "segments[].text are tokenized by the C++ driver; positions assigned there.",
        "seq_legend": {str(k): v for k, v in sorted(seq_label.items())},
        "nodes": nodes, "segments": segments, "stats": stats,
    }
    if uses:
        out["uses"] = {str(k): v for k, v in sorted(uses.items())}
    return out


def main():
    ap = argparse.ArgumentParser(description="LaTeX -> project JSON schema")
    ap.add_argument("tex_file")
    ap.add_argument("-o", "--out")
    ap.add_argument("--mode", choices=["sections", "cites"], default="sections")
    ap.add_argument("--merge-min", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--max-shared", type=int, default=400,
                    help="sections mode: cap tokens kept in the shared/intro sequence so "
                         "a long abstract+intro doesn't eat the budget before the first section")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    with open(args.tex_file, "r", encoding="utf-8", errors="ignore") as f:
        tex = strip_comments(f.read())
    tex = flatten_inputs(tex, os.path.dirname(os.path.abspath(args.tex_file)))  # pull in \input files
    nodes_top, _, _ = LatexWalker(tex).get_latex_nodes()
    body = document_body(nodes_top)

    if args.mode == "cites":
        nodes, segments, seq_label, uses = run_cites(body, args.merge_min, args.max_tokens)
    else:
        nodes, segments, seq_label, uses = run_sections(body, args.merge_min, args.max_tokens, args.max_shared)

    doc_id = args.tex_file.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    result = finish(nodes, segments, seq_label, uses, doc_id, args.mode)

    text = json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        s = result["stats"]
        print(f"wrote {args.out}  [latex / {args.mode}]", file=sys.stderr)
        print(f"  sequences={s['n_sequences']}  est_tokens={s['est_tokens_total']}  "
              f"markup_share={s['markup_or_syntax_share']}", file=sys.stderr)
        print(f"  seq sizes: {s['est_tokens_per_seq']}", file=sys.stderr)
        print(f"  legend: {result['seq_legend']}", file=sys.stderr)
        if "uses" in result:
            print(f"  uses (dependency graph): {result['uses']}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
