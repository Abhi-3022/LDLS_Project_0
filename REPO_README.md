# Structure-aware KV cache — shared repo

Prefill a structured document into the KV cache with **one sequence id per subtree**,
so a question attends only to the relevant subtree and unneeded subtrees can be evicted.
One shared driver serves every domain; each domain adds only a parser.

## Deliverables (assignment)

| Deliverable | File(s) | Scope |
|---|---|---|
| **Driver** | `driver.cpp`, `CMakeLists.txt`, `two_seq.cpp`, `DRIVER_README.md` | shared (all domains) |
| **Tests** | `TESTS.md`, `run_tests.ps1` (logic is inside `driver.cpp`) | shared |
| **Schema** | `SCHEMA.md` | shared |
| **Parser — HTML** | `parser.py` | per-domain |
| **Parser — LaTeX** | `parser_latex.py` | per-domain |

## Suggested repo layout
```
efficient-attention/
  REPO_README.md          <- this file
  SCHEMA.md               <- the JSON contract (shared)
  TESTS.md                <- the two tests (shared)
  run_tests.ps1           <- test runner + CPU timing (shared)
  driver/
    driver.cpp            <- load JSON -> prefill(seq ids) -> ask -> evict
    two_seq.cpp           <- Project-0 foundation: sequence-isolation sanity check
    CMakeLists.txt
    DRIVER_README.md
  parsers/
    html/   parser.py           (run with --main-only on real pages)
    latex/  parser_latex.py     (--mode sections | cites ; follows \input)
  samples/                <- sample_pages/ and generated *.json
```
(You can keep the flat layout you have now; this is just how to organize the repo.)

## Pipeline
```
document  ->  parser (per domain)  ->  JSON (SCHEMA.md)  ->  driver  ->  2 tests
   HTML        parser.py                                     driver.cpp
   LaTeX       parser_latex.py
```

## Build & run
```powershell
# build (from llama.cpp root, with examples/project0/ wired in)
cmake --build build --config Release --target llama-doc-driver -j

# HTML
python parser.py sample_pages\dog.html -o dog.json --pretty --main-only
build\bin\Release\llama-doc-driver.exe models\Llama-3.2-1B-Instruct-Q4_K_M.gguf ..\dog.json --ask 3 --change 6 --evict 5

# LaTeX
python parser_latex.py sample_pages\note.tex -o note_cites.json --pretty --mode cites
build\bin\Release\llama-doc-driver.exe models\Llama-3.2-1B-Instruct-Q4_K_M.gguf ..\note_cites.json --ask 3 --change 1 --evict 2
```

## Status
- Foundation (CPU llama.cpp, Llama-3.2-1B Q4_K_M): working.
- HTML + LaTeX parsers: working on real data (Wikipedia / arXiv).
- Driver, both tests: PASS on HTML and LaTeX.
