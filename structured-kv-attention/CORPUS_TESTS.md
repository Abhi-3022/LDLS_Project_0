# Per-domain corpus tests (all documents in one KV cache)

The two tests in `TESTS.md` run on a **single document**, where the subtrees are
its sections. This file runs the **same two tests at the document level**: every
document of a domain is prefilled into **one KV cache**, each document becoming
its own sequence (subtree), and the tests run **separately for each domain**.

```
5 HTML docs  ──merge_docs.py──►  html_corpus.json  ──driver──► tests across documents
5 LaTeX docs ──merge_docs.py──►  latex_corpus.json ──driver──► tests across documents
```

## Build the corpora

`merge_docs.py` collapses each document's sections into one sequence (one lane
per **document**) and concatenates the documents into a single corpus JSON — the
same schema the driver already reads. `--per-doc-cap` bounds each document so the
whole corpus fits the context window.

```bash
# HTML: 5 Wikipedia pages
python merge_docs.py samples/html_corpus.json \
    samples/dog.json samples/cat.json samples/moon.json samples/sun.json samples/water.json \
    --pretty --per-doc-cap 700

# LaTeX: 2 math notes + 3 arXiv papers
python merge_docs.py samples/latex_corpus.json \
    samples/note.json samples/analysis_note.json \
    samples/paper_attention.json samples/paper_resnet.json samples/paper_flashattn.json \
    --pretty --per-doc-cap 800
```

Resulting corpora (one sequence per document, seq 0 = shared):

```
html_corpus.json   5 docs -> seq 1..5 = dog, cat, moon, sun, water          (~4.6k real tok)
latex_corpus.json  5 docs -> seq 1..5 = note, analysis_note, attention,
                                         resnet, flashattention              (~4.0k real tok)
```

## Run the tests (same driver, bigger context)

The whole corpus must fit in the KV cache, so pass `--n-ctx` large enough (the
driver defaults to 4096; use 8192 for these corpora). The `--ask/--change/--evict`
sequence ids are now **documents**, read from the corpus's `seq_legend`.

```powershell
# rebuild the driver first (it now accepts --n-ctx)
copy "...\driver\driver.cpp" "...\llama.cpp\examples\project0\driver.cpp"
cmake --build build --config Release --target llama-doc-driver -j

# HTML corpus: ask about doc 1 (dog), corrupt doc 2 (cat), evict doc 3 (moon)
build\bin\Release\llama-doc-driver.exe models\Llama-3.2-1B-Instruct-Q4_K_M.gguf ^
    samples\html_corpus.json  --ask 1 --change 2 --evict 3 --n-ctx 8192

# LaTeX corpus: ask about doc 1 (note), corrupt doc 3 (attention), evict doc 4 (resnet)
build\bin\Release\llama-doc-driver.exe models\Llama-3.2-1B-Instruct-Q4_K_M.gguf ^
    samples\latex_corpus.json --ask 1 --change 3 --evict 4 --n-ctx 8192
```

## What each test proves at the corpus level

- **Test 1 — isolation across documents.** The question about document A lives in
  A's sequence and can only attend to A (plus the shared prefix). Corrupting a
  *different* document B cannot change A's answer — the whole domain corpus sits in
  one cache, yet documents don't leak into each other.
- **Test 2 — eviction of a whole document.** Evicting document C
  (`llama_memory_seq_rm` over its position range) frees that document's KV cells
  while A's answer is unchanged — reclaiming the memory of an entire unneeded
  document from a shared cache. This is the agent-context use case: old documents
  (or tool results) can be dropped as whole units.

Run per domain and record pass/fail, `KV cells before/after`, and CPU time — the
same report numbers as the single-document tests, now at document granularity.
