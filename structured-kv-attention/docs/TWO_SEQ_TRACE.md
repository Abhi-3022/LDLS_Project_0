# Trace: `two_seq.cpp` — how we adapted the batched example

`two_seq.cpp` is the Project-0 foundation sanity check. It takes the exact
machinery of `examples/batched/batched.cpp` and bends it to **prove sequence
isolation**: run two prompts as two sequences in one batch, and show the greedy
output of each is **identical** to running that prompt alone. If isolation
holds, they must match. This is the claim the whole structured-attention project
rests on.

Two deliberate differences from the batched example, both so the result is a
clean pass/fail:
1. **Greedy argmax, no sampler** → deterministic, so "same output" is exact.
2. **Raw `llama.h` API** (`llama_batch_init` / `llama_decode`), not the
   `common_*` / `llama_process` helpers → nothing hidden, every field is set by
   hand. This is the same path `driver.cpp` uses.

---

## Helpers (lines 35–62)

- `tokenize()` (35) — the **two-call `llama_tokenize` idiom**: first call with a
  null buffer returns the negated token count, then allocate and call again to
  fill. (The batched example hides this inside `common_tokenize`.)
- `token_to_text()` (47) — wraps **`llama_token_to_piece`** into a `std::string`.
- `argmax_at()` (54) — **greedy decode**: `llama_get_logits_ith(ctx, idx)` then a
  linear scan for the max. This replaces the batched example's
  `llama_sampler_sample`.

## `run_prompts()` — the core (lines 67–144)

Runs a set of prompts as parallel sequences, one `seq_id` each, and greedily
generates `n_predict` tokens per sequence.

- `n_vocab = llama_vocab_n_tokens(vocab)` (72).
- **`llama_memory_clear(llama_get_memory(ctx), true)`** (75) — start from an
  empty KV cache so one `run_prompts` call can't contaminate the next. (This is
  the memory API — the modern replacement for `llama_kv_cache_clear`.)
- Tokenize every prompt, BOS at the start of each sequence (78–83).
- **`llama_batch_init(max(total_prompt_tokens, n_seq), 0, 1)`** (87) — the raw
  batch. Third arg = **max seq_ids per token = 1**: here every token belongs to
  exactly one sequence. (Contrast the batched example, where prompt tokens
  belonged to *all* sequences at once — that needed the coupled/unified path.)

### PREFILL (lines 92–111) — this is where isolation is set up

```cpp
batch.token[k]     = toks[s][i];
batch.pos[k]       = pos[s]++;     // positions are PER SEQUENCE, from 0
batch.n_seq_id[k]  = 1;
batch.seq_id[k][0] = s;            // <<< this single line is the isolation
batch.logits[k]    = is_last ? 1 : 0;   // only need logits at each seq's last token
```

Every field the batched example set through `batch.add(...)` is set here by
hand. Tagging token `k` with `seq_id = s` is what confines it to sequence `s`'s
attention mask. `last_idx[s]` records the row holding sequence `s`'s final
logits — the equivalent of the batched example's `i_batch[i]`.

- **`llama_decode(ctx, batch)`** (107) — the prefill forward pass (what
  `llama_process(...DECODE...)` wraps in the example).
- First generated token per sequence: `argmax_at(ctx, n_vocab, last_idx[s])`
  (117–120).

### GENERATE (lines 122–140)

One new token per sequence per step: rebuild the batch with `n_seq` tokens (each
at its own `pos[s]++`, tagged with its own `seq_id`, logits on), `llama_decode`,
then argmax each sequence's new logits. Structurally identical to the batched
loop, minus the sampler.

- `llama_batch_free(batch)` (142).

## `main()` — the experiment (lines 152–206)

- Model load **forced onto CPU**: `mparams.n_gpu_layers = 0` (166) — the project
  runs CPU-only. `llama_model_load_from_file` (167), `llama_model_get_vocab`
  (170).
- Context (173–177): `n_ctx = 4096`, `n_batch = 4096` (must hold the whole
  prefill in one decode), `n_seq_max = 8`. `llama_init_from_model`.
- **The actual test** (180–184):
  ```cpp
  together = run_prompts(ctx, vocab, {A, B}, n_predict);  // two sequences, one batch
  only_a   = run_prompts(ctx, vocab, {A},    n_predict);  // A alone
  only_b   = run_prompts(ctx, vocab, {B},    n_predict);  // B alone
  ```
- Compare (191–200): `together[0] == only_a[0]` and `together[1] == only_b[0]`.
  Both must match → `PASS — sequences are isolated by seq_id`. Any mismatch →
  `FAIL — sequences leaked into each other`. Exit code 0 on pass, 2 on fail (so a
  script can check it).
- Cleanup: `llama_free`, `llama_model_free`, `llama_backend_free` (202–204).

---

## The line of descent

```
batched.cpp                     two_seq.cpp                     driver.cpp
-----------                     -----------                     ----------
many continuations of 1 prompt  prove seq_id isolates           many subtrees of 1 document
sampler chain                   greedy argmax (deterministic)   greedy argmax
common_batch/llama_process      raw llama_batch_init/decode     raw llama_batch_init/decode
prompt in all seqs (coupled)    one seq per prompt (clean)      one seq per subtree + seq_cp prefix
i_batch[]                       last_idx[]                      last_idx
                                + memory API clear              + memory API seq_rm (eviction)
```

`two_seq.cpp` is the bridge: it keeps the batched example's sequence machinery
but strips it to a deterministic yes/no isolation test, in the raw API the
driver then builds on. Once "seq_id isolates a sequence" is proven here, the
driver's two tests (isolation under corruption, safe eviction) are just that same
guarantee applied to document subtrees.

## `llama.h` symbols touched

`llama_backend_init` / `llama_backend_free`, `llama_model_default_params`,
`llama_model_load_from_file`, `llama_model_get_vocab`, `llama_model_free`,
`llama_context_default_params`, `llama_init_from_model`, `llama_free`,
`llama_memory_clear` + `llama_get_memory`, `llama_tokenize`,
`llama_token_to_piece`, `llama_vocab_n_tokens`, `llama_batch_init` /
`llama_batch_free`, `llama_decode`, `llama_get_logits_ith`.
