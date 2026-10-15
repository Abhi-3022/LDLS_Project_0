# Trace: `examples/batched/batched.cpp`

Assignment step: *"Read `llama.h` and trace the batched example."* This is the
foundation exercise — understand how llama.cpp runs **several sequences in one
KV cache** before building the structured driver that depends on exactly that.

**What the example does in one line:** take *one* prompt and generate
`n_parallel` **independent continuations of it at once**, all sharing the model
and the prompt's KV cache. That "many sequences in one cache" idea is what the
project's driver generalizes to "one sequence per document subtree."

> Note on API: this checkout's copy uses the newer helper layer —
> `common_batch`, `common_tokenize`, `llama_process(...)` — which wrap the raw
> `llama.h` calls (`llama_batch`, `llama_tokenize`, `llama_decode`). The driver
> in this repo uses the raw calls directly; both hit the same machinery.

---

## Phase 1 — Setup & model load (lines 19–55)

| line | call | `llama.h` role |
|---|---|---|
| 27 | `common_init()` | logging/CLI init |
| 29 | `common_params_parse(..., LLAMA_EXAMPLE_BATCHED, ...)` | parse `-m -p -n -np` |
| 41 | **`llama_backend_init()`** | start the runtime (§Backend) |
| 46 | `common_model_params_to_llama(params)` | fill `llama_model_params` |
| 48 | **`llama_model_load_from_file(path, mparams)`** | load the GGUF (§Model loading) |
| 55 | **`llama_model_get_vocab(model)`** | vocab handle (§Model) |

## Phase 2 — Tokenize & size the cache (lines 59–69)

- `common_tokenize(vocab, prompt, true)` (60) → token ids; `true` prepends BOS.
  Wraps **`llama_tokenize`**.
- **`n_kv_req = prompt_len + (n_predict − prompt_len) * n_parallel`** (62) — the
  key sizing line. The prompt is stored **once** (shared across all streams);
  each of the `n_parallel` streams then adds its own generated tokens on top.
- `ctx_params.n_ctx = n_kv_req`, `n_batch = max(n_predict, n_parallel)` (68–69).

## Phase 3 — Context & per-sequence samplers (lines 71–99)

- A **sampler chain per stream** (76–85): `llama_sampler_chain_init` +
  `top_k / top_p / temp / dist`. Each parallel sequence samples independently,
  which is why the continuations differ.
- `backend_sampling` (87–90) attaches the samplers to the context via
  `llama_sampler_seq_config` (the newer in-backend sampling path).
- **`llama_init_from_model(model, ctx_params)`** (92) — create the context.
- `llama_n_ctx(ctx)` (99) + the `n_kv_req > n_ctx` guard (104).

## Phase 4 — Prefill the prompt into ALL sequences (lines 120–154) ← the crux

```cpp
common_batch batch(ctx);                       // 120
std::vector<llama_seq_id> seq_ids(n_parallel);  // {0,1,...,n_parallel-1}  122-125
for (i = 0; i < tokens_list.size(); ++i)
    batch.add(tokens_list[i], i, seq_ids, false);   // 128-130
```

Each prompt token is added at position `i` and tagged with **every** sequence id
at once. So the prompt occupies one set of cells that **belong to all parallel
sequences simultaneously** — one physical copy, shared. (This multi-seq-per-token
case is the "coupled sequences" situation that needs a unified KV cache — the same
`kv_unified` issue the driver hit.)

- `batch.set_output(batch.size()-1, true)` (149) — ask for logits only on the
  **last** prompt token (all you need to start generating).
- **`llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())`** (151) — the
  forward pass = **prefill**. `llama_process` is the newer wrapper over
  **`llama_decode`**.
- **Lines 156–160 (commented out)** — the alternative: prefill the prompt in
  seq 0 only, then `llama_kv_cache_seq_cp(0, i, -1, -1)` into the others.
  **This is exactly the `seq_cp` shared-prefix trick the driver uses** — the
  example just chose the "tag with all seq_ids" route instead.
- Lines 133–146 handle encoder-decoder models (not used by Llama; `has_encoder`
  is false, so this branch is skipped).

## Phase 5 — The generation loop (lines 166–231)

- `i_batch[i]` (173) — remembers **which batch row holds each stream's last
  logits** (the driver's `last_idx`, one per sequence).
- Each iteration (`while n_cur <= n_predict`):
  1. `batch.clear()` (182).
  2. For each stream `i`: **`llama_sampler_sample(sampler, ctx, i_batch[i])`**
     (191) — sample its next token from its own logits row.
  3. **`llama_vocab_is_eog(vocab, new_token_id)`** (194) — stop that stream at
     end-of-generation (mark `i_batch[i] = -1`).
  4. `batch.add(new_token_id, n_cur, i, true)` (214) — append the new token at
     position `n_cur` in **that one sequence**, logits on.
  5. **`llama_process(...DECODE...)`** (227) decodes the whole
     `n_parallel`-token batch in one shot — every live stream advances one step.
- Loop ends when every stream hits EOG or `n_predict` (`batch.size()==0`, 220).

## Phase 6 — Report & cleanup (lines 233–262)

- Print each `streams[i]`, timing, `llama_perf_sampler_print` /
  `llama_perf_context_print`.
- `llama_sampler_free` (×n), **`llama_free(ctx)`**, **`llama_model_free(model)`**,
  **`llama_backend_free()`**.

---

## How it maps to this project's driver

| batched example | `driver.cpp` (this repo) |
|---|---|
| one prompt → N parallel **continuations** | one document → N **subtree** sequences |
| prompt tagged with all seq_ids (shared once) | shared prefix copied via `llama_memory_seq_cp` into each section |
| `i_batch[i]` = per-stream logits row | `last_idx` in `ask()` |
| `llama_sampler_sample` (sampler chain) | hand-rolled argmax (greedy, deterministic) |
| `common_batch` / `common_tokenize` / `llama_process` (helpers) | raw `llama_batch_init` / `llama_tokenize` / `llama_decode` |
| commented-out `llama_kv_cache_seq_cp` | the live shared-prefix mechanism |

Same machinery, opposite intent: the batched example proves *multiple sequences
live in one KV cache and decode together*; the driver repurposes those sequences
to mean *parts of a document*, so a question can be isolated to one subtree and
unneeded subtrees can be evicted.

## `llama.h` symbols touched

`llama_backend_init` / `llama_backend_free`, `llama_model_load_from_file`,
`llama_model_get_vocab`, `llama_model_free`, `llama_init_from_model`,
`llama_free`, `llama_n_ctx`, `llama_model_has_encoder`,
`llama_model_decoder_start_token`, `llama_vocab_bos`, `llama_tokenize` (via
`common_tokenize`), `llama_batch` + add/clear/set_output, `llama_decode` (via
`llama_process`), `llama_sampler_*`, `llama_vocab_is_eog`, `llama_token_to_piece`
(via `common_token_to_piece`), and the commented `llama_kv_cache_seq_cp`.
