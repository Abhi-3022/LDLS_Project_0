#!/usr/bin/env python3
"""
Build a per-domain CORPUS from several document JSONs: load ALL of a domain's
documents into ONE KV cache, each document becoming its own sequence (subtree).

This feeds the SAME driver. The per-document tests then run across documents:
  - isolation: change document B, ask about document A -> answer unchanged
  - eviction : evict document C from the cache -> answer holds, KV memory drops

Each input document's internal sections are collapsed into a single sequence
(one lane per DOCUMENT), which is the right granularity for cross-document tests.
A document can be capped with --per-doc-cap so the whole corpus fits the context.

Usage:
    python merge_docs.py html_corpus.json dog.json cat.json moon.json sun.json water.json
    python merge_docs.py latex_corpus.json note_cites.json analysis_note_cites.json \
        paper_attention.json paper_resnet.json paper_flashattn.json --per-doc-cap 700
"""

import argparse
import json
import os
import re
import sys


def est_tokens(s: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", s or ""))


def main():
    ap = argparse.ArgumentParser(description="Merge domain docs into one corpus JSON (doc = sequence)")
    ap.add_argument("out")
    ap.add_argument("docs", nargs="+", help="per-document JSON files (same domain)")
    ap.add_argument("--per-doc-cap", type=int, default=0,
                    help="cap proxy tokens kept per document (0 = no cap) so the corpus fits n_ctx")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    nodes = [{"node_id": 0, "node_type": "corpus", "parent": -1, "depth": 0, "seq_id": 0}]
    seq_legend = {0: "shared"}
    segments = []
    domain = None

    for i, path in enumerate(args.docs, start=1):
        d = json.load(open(path, encoding="utf-8"))
        domain = domain or d.get("domain", "mixed")
        doc_id = d.get("doc_id") or os.path.basename(path).rsplit(".", 1)[0]
        nodes.append({"node_id": i, "node_type": "document", "parent": 0,
                      "depth": 1, "seq_id": i, "heading": doc_id})
        seq_legend[i] = doc_id
        # collapse this document's sections into sequence i, in order, up to the cap
        used = 0
        for seg in d.get("segments", []):
            t = seg.get("text", "")
            if not t:
                continue
            if args.per_doc_cap and used >= args.per_doc_cap:
                break
            segments.append({"node_id": i, "seq_id": i,
                             "kind": seg.get("kind", "content"), "text": t})
            used += est_tokens(t)

    # positions / stats
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
        "max_depth": 1,
        "n_nodes": len(nodes),
        "n_sequences": len(set(s["seq_id"] for s in segments)),
        "n_documents": len(args.docs),
        "est_tokens_total": total,
        "est_tokens_per_seq": {str(k): v for k, v in sorted(per_seq.items())},
        "markup_or_syntax_share": round(markup / total, 3) if total else 0.0,
    }
    out = {
        "doc_id": os.path.basename(args.out).rsplit(".", 1)[0],
        "domain": domain, "mode": "corpus (one sequence per document)",
        "note": "each sequence is a whole document; the driver tests isolation and "
                "eviction across documents. Tokens here are a proxy; the driver tokenizes.",
        "seq_legend": {str(k): v for k, v in sorted(seq_legend.items())},
        "nodes": nodes, "segments": segments, "stats": stats,
    }
    text = json.dumps(out, indent=2 if args.pretty else None, ensure_ascii=False)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {args.out}  [{domain} corpus]", file=sys.stderr)
    print(f"  documents={stats['n_documents']}  sequences={stats['n_sequences']}  "
          f"est_tokens={total}  (~{int(total*1.3)} real)", file=sys.stderr)
    print(f"  per-doc sizes: {stats['est_tokens_per_seq']}", file=sys.stderr)
    print(f"  legend: {out['seq_legend']}", file=sys.stderr)


if __name__ == "__main__":
    main()
