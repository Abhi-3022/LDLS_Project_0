# The shared JSON schema

Every parser (one per domain) emits **this** JSON. The C++ driver reads only this
JSON and never knows which domain produced it — that is what lets one driver serve
all domains. The schema is the contract between Stage 1 (parsers) and Stage 2 (driver).

## Top-level object

```jsonc
{
  "doc_id":   "dog",                 // string: document name
  "domain":   "html",                // "html" | "latex"
  "mode":     "main-only",           // parser mode used (html: full-dom|main-only; latex: sections|cites)
  "note":     "...",                 // human note; ignored by the driver
  "seq_legend": {                    // seq_id -> human label (for printing / the report)
    "0": "intro / shared",
    "1": "Appearance and behaviour",
    "3": "Anatomy"
  },
  "nodes":    [ ...node objects... ],      // the tree  (carries node_type + parent)
  "segments": [ ...segment objects... ],   // ordered text pieces (carries tokens + seq_ids)
  "stats":    { ...tree statistics... },   // for the report slide
  "uses":     { "4": [1, 2] }        // OPTIONAL: latex cites mode only (dependency graph)
}
```

## node object  (the tree)

```jsonc
{
  "node_id":   3,            // int, unique
  "node_type": "section",    // tag/kind: "section","p","theorem","proof","intro",...
  "parent":    0,            // node_id of parent, or -1 for the root
  "depth":     1,            // distance from root
  "seq_id":    3,            // which sequence (KV-cache lane) this node belongs to
  "heading":   "Anatomy",    // optional: section/ block title
  "label":     "thm:main"    // optional: latex cites mode, the \label
}
```

## segment object  (ordered text, in document order)

```jsonc
{
  "node_id":      3,                  // which node this text belongs to
  "seq_id":       3,                  // KV-cache lane  (this is what the driver reads)
  "kind":         "content",          // "markup" | "content" | "script"
  "text":         "Dogs are loyal.",  // the text the DRIVER tokenizes  (this is "tokens")
  "n_tokens_est": 4,                  // proxy token count (stats only; not real Llama tokens)
  "pos_start_est":120                 // estimated start position (inspection only)
}
```

Concatenating every `segments[].text` in order reproduces the (cleaned) source document.

## stats object  (report slide)

```jsonc
{
  "max_depth":              1,
  "n_nodes":                8,
  "n_sequences":            7,
  "est_tokens_total":       1304,
  "est_tokens_per_seq":     { "0": 239, "1": 88, "3": 367 },  // subtree sizes
  "markup_or_syntax_share": 0.166                              // share of markup/syntax tokens
}
```

## How the five required fields map

The assignment fixes five fields: `tokens, positions, seq_ids, node_type, parent`.

| required field | where it lives |
|---|---|
| `seq_ids`   | `nodes[].seq_id` and `segments[].seq_id` |
| `node_type` | `nodes[].node_type` |
| `parent`    | `nodes[].parent` |
| `tokens`    | `segments[].text` — the driver tokenizes these with **Llama's** tokenizer, so the token ids always match the model |
| `positions` | assigned by the **driver** as it tokenizes segments in document order (a `pos_start_est` is included only for inspection) |

**Design note (deliberate).** We normalize `node_type`/`parent` into the `nodes` table
and reference them from `segments` by `node_id`, rather than repeating them on every
token. The required fields are all present; this just avoids duplicating the whole tree
on each token.

## What the driver actually consumes

Minimal contract: the driver needs only
- `segments[]` → `{ seq_id, text }` in order, and
- `seq_legend` (for printing).

Everything else (`nodes`, `stats`, `uses`) is for humans, the report, or future
extensions. A new domain is supported by writing a parser that emits this object —
no driver change.

## Merge rule (small subtrees)

llama.cpp allows only a limited number of sequences (`n_seq_max`). Subtrees whose token
count is below `--merge-min` are folded into the shared sequence (`seq_id = 0`) instead
of getting their own lane. This keeps the sequence count bounded, as the assignment
requires for very small subtrees.
