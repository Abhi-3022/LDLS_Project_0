# Structure-aware KV-cache attention (HTML + LaTeX)

Project 0 — efficient / structure-aware attention on CPU with `llama.cpp`.

A document is parsed into a **tree of subtrees**, and each subtree is prefilled
into the KV cache under its **own sequence id**. Because a question attends only
to keys sharing its sequence, we can (1) **isolate** a question to one subtree and
(2) **evict** an unrelated subtree from the cache to reclaim memory — without
changing the answer.

One **parser per domain** turns a document into a shared JSON schema; one
**shared driver** reads that JSON for every domain. This repo has two domains —
**HTML** and **LaTeX** — feeding the exact same `driver.cpp`. Adding a domain
means adding a parser, not touching the driver.

```
document (HTML)   ──parser.py────────►  JSON ┐
                                              ├─(SCHEMA.md)─► driver.cpp ─► 2 tests
document (LaTeX)  ──parser_latex.py──►  JSON ┘
```

## Repo layout

```
structured-kv-attention/
  README.md              <- this file (how to build & run)
  SCHEMA.md              <- the JSON contract between parser and driver (shared)
  TESTS.md               <- what the two tests prove (shared)
  run_tests.ps1          <- runs the driver and records CPU time (Windows)
  driver/
    driver.cpp           <- load JSON -> prefill(seq ids) -> ask -> evict   (THE driver)
    two_seq.cpp          <- foundation sanity check: prove seq_id isolates
    CMakeLists.txt       <- build integration for llama.cpp
    DRIVER_README.md     <- driver internals
  parsers/html/
    parser.py            <- HTML -> JSON parser
    requirements.txt     <- beautifulsoup4
  parsers/latex/
    parser_latex.py      <- LaTeX -> JSON parser  (--mode sections | cites)
    requirements.txt     <- pylatexenc
  samples/
    dog.html  dog.json   <- HTML worked example (Wikipedia "Dog", cleaned)
    pets.html pets.json  <- HTML tiny example for a quick smoke test
    note.tex             <- LaTeX worked example (a short math note)
    note.json            <- note.tex parsed in --mode sections
    note_cites.json      <- note.tex parsed in --mode cites (with `uses` graph)
  docs/
    BATCHED_TRACE.md     <- trace of llama.cpp's batched example
    TWO_SEQ_TRACE.md     <- trace of two_seq.cpp (how we adapted it)
```

## Prerequisites

- A C++17 compiler + **CMake** (`winget install Kitware.CMake` on Windows).
- **Python 3.9+** for the parser.
- **`llama.cpp`** source (cloned next to this repo — *not* committed here).
- A **GGUF model**. Tested with `Llama-3.2-1B-Instruct-Q4_K_M.gguf` on CPU.

> `llama.cpp/`, `build/`, `models/` and `*.gguf` are git-ignored — everyone
> clones/builds/downloads those locally.

## 1. Build the driver inside llama.cpp

The driver links against `llama`, so it builds as a llama.cpp example.

```bash
# clone llama.cpp somewhere (here: next to this repo)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# wire our programs in as an example
mkdir -p examples/project0
cp ../structured-kv-attention/driver/driver.cpp      examples/project0/
cp ../structured-kv-attention/driver/two_seq.cpp     examples/project0/
cp ../structured-kv-attention/driver/CMakeLists.txt  examples/project0/
echo "add_subdirectory(project0)" >> examples/CMakeLists.txt

# build (CPU)
cmake -B build
cmake --build build --config Release -j
```

Binaries:
- `build/bin/Release/llama-doc-driver.exe` (Windows) · `build/bin/llama-doc-driver` (Linux/macOS)
- `build/bin/Release/llama-two-seq.exe`    · `build/bin/llama-two-seq`

## 2. Get the model (CPU)

```bash
mkdir -p models
# e.g. with the huggingface CLI (pick any GGUF you have):
hf download bartowski/Llama-3.2-1B-Instruct-GGUF \
   Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models
```

(`-ngl 0` / `n_gpu_layers = 0` in the code already forces CPU — no GPU needed.)

## 3. Parse an HTML document → JSON

```bash
cd ../structured-kv-attention
pip install -r parsers/html/requirements.txt

# a real Wikipedia page (strips chrome, splits by <h2>, cleans, caps tokens):
python parsers/html/parser.py samples/dog.html -o samples/dog.json --pretty --main-only

# a tiny hand-made page (full DOM, no cleaning):
python parsers/html/parser.py samples/pets.html -o samples/pets.json --pretty
```

Useful flags: `--main-only` (for real web pages), `--max-tokens 1500`,
`--merge-min 16` (fold subtrees smaller than this into the shared sequence).
The parser prints a `seq_legend` (seq_id → section name) — you pick `--ask`/
`--change`/`--evict` from it.

### LaTeX (same schema, same driver)

```bash
pip install -r parsers/latex/requirements.txt

# sections mode: one sequence per \section (natural "ask one section")
python parsers/latex/parser_latex.py samples/note.tex -o samples/note.json --pretty --mode sections

# cites mode: one sequence per labeled block (theorem/lemma/proof/...) PLUS a
# `uses` dependency graph of which block \ref's which  (the structure-aware bit)
python parsers/latex/parser_latex.py samples/note.tex -o samples/note_cites.json --pretty --mode cites
```

`parser_latex.py` strips the preamble and comments, inlines `\input`/`\include`
(for multi-file arXiv papers), keeps the math, and emits the identical JSON. In
`cites` mode it also writes `uses`, e.g. `{"4": [1, 2]}` = "the proof (seq 4)
depends on seqs 1 and 2". Point it at any arXiv paper's `.tex` source.

## 4. Run the two tests (the driver)

```
llama-doc-driver  <model.gguf>  <doc.json>  --ask A --change B --evict C
```
`A B C` are three **distinct** sequence ids from the doc's `seq_legend`.

```bash
# from the llama.cpp folder, using the dog.json we just made:
build/bin/Release/llama-doc-driver.exe \
    models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    ../structured-kv-attention/samples/dog.json \
    --ask 3 --change 6 --evict 5
```

On Windows you can instead use the wrapper (it also prints CPU run time):

```powershell
powershell -ExecutionPolicy Bypass -File run_tests.ps1 -Doc samples\dog.json -Ask 3 -Change 6 -Evict 5
```

Same driver, LaTeX input — ask about the theorem (seq 3), corrupt the lemma
(seq 2), evict the definition (seq 1):

```bash
build/bin/Release/llama-doc-driver.exe \
    models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    ../structured-kv-attention/samples/note_cites.json \
    --ask 3 --change 2 --evict 1
```

### Expected output (tail)

```
TEST 1: PASS   (changing seq B did NOT affect the answer about seq A)
TEST 2: PASS   KV cells: before=N after=M (freed K tokens of seq C)
RESULT: BOTH TESTS PASS
```

- **Test 1 — isolation:** ask about subtree A; corrupt subtree B (same token
  count) and ask again → answer unchanged, because A's sequence can't attend to B.
- **Test 2 — eviction:** evict subtree C by position range
  (`llama_memory_seq_rm`) → A's answer unchanged and KV cell count drops.

See `TESTS.md` for the full reasoning (incl. why B is corrupted at the token
level to keep `n_kv` fixed) and `SCHEMA.md` for the JSON contract.

## 5. (Optional) Foundation sanity check

`two_seq.cpp` proves the premise everything rests on — that a `seq_id` isolates a
sequence in the KV cache:

```bash
llama-two-seq  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf  "The capital of France is"  "Water boils at"
# -> seq 0 vs A-alone : MATCH ;  seq 1 vs B-alone : MATCH ;  RESULT: PASS
```

## How it works (one screen)

1. **Parse** (`parser.py` for HTML, `parser_latex.py` for LaTeX): DFS over the
   document tree → one `seq_id` per subtree (section, or labeled block), emit
   `{nodes, segments, seq_legend, stats}` per `SCHEMA.md` (LaTeX `cites` mode also
   emits a `uses` dependency graph). The driver never sees HTML or LaTeX — only
   this JSON.
2. **Prefill** (`driver.cpp`): decode a **shared prefix** (intro) into seq 0, copy
   it into each section's sequence (`llama_memory_seq_cp`), then append each
   section's tokens. Document tokens are decoded with logits **off** (KV only).
3. **Ask**: append the question to the target sequence (logits **on** at the last
   token), greedily decode the answer, then remove the question tokens.
4. **Evict**: `llama_memory_seq_rm(seq, p0, p1)` frees a subtree's cells.

Decoding is **greedy** so test comparisons are exact. Context uses
`kv_unified = true` (tokens may belong to the shared prefix across sequences).

## Notes

- CPU-only, dependency-free driver (its own tiny JSON reader — no libraries).
- Tested with Llama-3.2-1B Q4_K_M; any GGUF the installed `llama.cpp` supports
  should work.
- `llama.cpp`'s C API is renamed periodically; lines likely to differ across
  versions are marked `// [API]` in the sources.
