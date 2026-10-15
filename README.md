# Project 0 — Common Foundation (CPU only)

Goal of this stage (everyone on every team does it, before the domain-specific work):

1. Build **llama.cpp** on CPU.
2. Run **Llama-3.2-1B Instruct, Q4_K_M**.
3. Read `llama.h` and trace the **batched** example.
4. Understand how the **attention mask is built from sequence IDs**.
5. Prove that **two prompts run as two sequences in one batch == two separate runs.**

Step 5 is the real deliverable. The provided `two_seq.cpp` does it and prints `PASS`/`FAIL`.

> Why step 5 matters: it is the experiment that proves sequence IDs *isolate* sequences.
> The entire project later relies on this — you will assign one sequence per document
> subtree, and you need to trust that subtree B can never leak into subtree A.

---

## 0. Prerequisites

```bash
# Linux/Mac. You need: git, cmake (>=3.14), a C++17 compiler, python3 (for the model download).
sudo apt-get install -y git cmake build-essential   # Debian/Ubuntu
# mac: xcode-select --install ; brew install cmake
```

---

## 1. Build llama.cpp on CPU

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# CPU-only build (no CUDA). On an x86 box this auto-enables AVX for speed.
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build build -j
```

This produces tools in `build/bin/`, e.g. `llama-cli`, `llama-batched`, `llama-simple`.

---

## 2. Get the model (Llama-3.2-1B Q4_K_M, GGUF)

`Q4_K_M` = weights quantized to ~4 bits (the K-quant medium variant). ~0.8 GB.

```bash
pip install -U "huggingface_hub[cli]"

# Community GGUF repos are usually ungated. If one 404s, try another (unsloth, lmstudio-community).
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --local-dir models
```

If a repo is gated, log in first with `huggingface-cli login` (needs a HF token and, for
Meta's own repo, accepting the license). The `bartowski` conversion is typically open.

---

## 3. Sanity run (does the model generate at all?)

```bash
# --temp 0 = greedy = deterministic. -ngl 0 = 0 GPU layers (CPU); it's the default here anyway.
./build/bin/llama-cli -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    -p "The capital of France is" -n 16 --temp 0 -no-cnv
```

You should see it complete the sentence. If this works, your toolchain and model are good.

---

## 4. Read `llama.h` and trace the batched example

Open two files side by side:

- `include/llama.h` — the public C API.
- `examples/batched/batched.cpp` — generates several sequences **in parallel**, sharing the model.

Walk the batched example and connect each call to `llama.h`. The spine is:

| Call | What it does |
|---|---|
| `llama_backend_init()` | start the runtime |
| `llama_model_load_from_file()` | load weights (set `n_gpu_layers = 0` for CPU) |
| `llama_init_from_model()` | create a context (holds the KV cache); set `n_ctx`, `n_batch`, `n_seq_max` |
| `llama_tokenize()` | text → token ids |
| `llama_batch_init()` / fill `batch.token/pos/seq_id/logits` | assemble a decode step |
| `llama_decode()` | run the forward pass over the batch |
| `llama_get_logits_ith()` | read logits for a chosen row → sample next token |
| `llama_kv_self_clear()` / `llama_kv_self_seq_rm()` | clear / **evict** KV cache (you'll use `seq_rm` later) |

> The exact names drift between versions (see the `// [API]` notes in `two_seq.cpp`).
> Always trust the `llama.h` in **your** checkout over any tutorial.

**The four fields of `llama_batch` are the whole game:**

- `token[i]` — the token id at row *i*
- `pos[i]` — its position **within its own sequence** (each sequence counts from 0)
- `seq_id[i][..]` — which sequence(s) this token belongs to  ← the structural handle
- `logits[i]` — whether to compute output logits for this row (set for the last token you want to predict from)

---

## 5. How the mask is built from sequence IDs (trace this)

The rule llama.cpp enforces when it builds the attention mask:

> A query token at **(sequence `s`, position `p`)** may attend to a cached key token only if
> **(a)** that key shares sequence id `s`, **and (b)** its position ≤ `p` (causal).

So two tokens with **different** seq_ids are masked apart — they cannot attend to each other,
even in the same batch. That is exactly why batching prompt A (seq 0) and prompt B (seq 1)
together gives the same result as running them separately.

Where to see it in the source (names vary by version — grep for these):

- The KV-cache classes in `src/llama-kv-cache*.cpp`. Look for where the **KQ mask** input is
  filled (e.g. a function like `set_input_kq_mask`). You'll find a loop over cached cells that
  sets the mask to `-INF` unless the cell **belongs to the query's sequence** (a `seq_has(...)`
  / seq-id check) **and** its position is ≤ the query position.
- `llama_set_causal_attn()` in `llama.h` toggles the causal part.

Write ~5 lines in your report describing that check — that is the "trace mask construction
from sequence IDs" deliverable.

---

## 6. Run the isolation experiment (the deliverable)

Add the custom program to the build:

```bash
# from the llama.cpp repo root
mkdir -p examples/two-seq
cp /path/to/two_seq.cpp     examples/two-seq/
cp /path/to/CMakeLists.txt  examples/two-seq/
echo 'add_subdirectory(two-seq)' >> examples/CMakeLists.txt

cmake --build build -j
```

Run it:

```bash
./build/bin/llama-two-seq \
    models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    "The capital of France is" \
    "Two plus two equals" \
    32
```

What it does: runs both prompts **together** as seq 0 and seq 1 in one batch (greedy), then runs
each **alone**, and compares the generated token streams. Expected tail of the output:

```
seq 0 vs A alone : MATCH ✓
seq 1 vs B alone : MATCH ✓
RESULT: PASS — sequences are isolated by seq_id
```

Greedy decoding (argmax, no sampling) makes the comparison exact and deterministic — if isolation
holds, the streams are identical token-for-token.

### If it says FAIL / DIFFER
- Make sure `n_ctx` is large enough for **all** sequences at once (cells are shared: total ≈ sum
  of every sequence's prompt+generated length). Bump `cparams.n_ctx` if your prompts are long.
- Make sure you cleared the KV cache between runs (the code calls `llama_kv_self_clear`; if your
  version renamed it, fix that line — otherwise the "alone" runs see stale keys and diverge).
- Keep decoding greedy; any temperature/sampling makes outputs non-deterministic and the compare
  meaningless.

---

## 7. What to put in the repo / report for this stage

- The build working and the sanity run (step 3) — a screenshot or log line.
- Your 5-line description of the seq_id → mask check (step 5).
- The `llama-two-seq` `PASS` output (step 6).
- CPU run time (wrap the run in `time ./build/bin/llama-two-seq ...`).

This foundation is shared by all teams; the domain-specific parser, JSON schema, and the
subtree-eviction driver build directly on top of `two_seq.cpp`'s batch-filling pattern.
