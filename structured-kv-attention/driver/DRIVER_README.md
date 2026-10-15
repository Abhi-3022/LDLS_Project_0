# The C++ driver — running the two tests

`driver.cpp` is the shared driver. It reads a JSON file from `parser.py`, prefills
the document into the KV cache with one **sequence id per section**, and runs the
two required tests.

## How the structure reaches the model
- `seq 0` holds the shared / intro context.
- `seq 1..K` hold one section each (the evictable subtrees).
- A **question about section S** is tagged with sequence ids `{0, S}`, so it attends
  to the shared context **and** section S only — never the other sections.
  That is the whole trick: it is why changing another section can't affect the answer,
  and why evicting another section is safe.

## Build
```
# from the llama.cpp repo root
mkdir -p examples/project0
# copy two_seq.cpp, driver.cpp, CMakeLists.txt into examples/project0/
echo 'add_subdirectory(project0)' >> examples/CMakeLists.txt
cmake --build build --config Release -j
```
On Windows the binary is `build\bin\Release\llama-doc-driver.exe`.

## Run
```
build\bin\Release\llama-doc-driver.exe <model.gguf> dog.json
```
Options (all optional — sensible defaults are chosen and printed):
- `--ask S`      section to ask about (default: first section)
- `--change B`   section to corrupt in Test 1 (default: last section)
- `--evict C`    section to evict in Test 2 (default: a different section)
- `--q "..."`    custom question (default: asks what the target section says)
- `-n N`         max answer tokens (default 48)

Example on the dog page (ask about Anatomy, change Health, evict Dog breeds):
```
build\bin\Release\llama-doc-driver.exe models\Llama-3.2-1B-Instruct-Q4_K_M.gguf dog.json --ask 3 --change 6 --evict 5
```

## What you should see
```
================ TEST 1: isolation ================
answer (original)    : ...
answer (after change): ...            <- identical to the first
TEST 1: PASS ✓  (changing seq 6 did NOT affect the answer about seq 3)

================ TEST 2: eviction =================
KV cells: before=NNN  after=MMM  (freed K tokens of seq 5)
KV memory (approx): before=X MB  after=Y MB     <- Y < X
answer (before evict): ...
answer (after evict) : ...            <- identical
TEST 2: PASS ✓
```
These two PASS lines plus the KV-cells numbers are your report-slide test results.
Wrap the command in `Measure-Command { ... }` (PowerShell) to get the CPU run time.

## If a test FAILS
- Answers differ when they shouldn't: make sure `--ask` and `--change`/`--evict`
  are **different** sections, and that the question targets the `--ask` section.
- Build errors on a `// [API]` line: open your `include/llama.h` and swap in the
  name it has (the alternatives are written next to each marked line). The most
  likely ones are the KV-cache functions: `llama_kv_self_seq_rm` /
  `llama_kv_self_clear` may be `llama_kv_cache_seq_rm` / `llama_kv_cache_clear`,
  or the newer memory API `llama_memory_seq_rm(llama_get_memory(ctx), ...)`.

## Notes / honesty
- The KV-memory figure is an approximate upper bound (`2·layers·n_embd·fp16` per
  token); it ignores the GQA reduction, so treat it as "tokens in cache × a
  constant." The **cells freed** number is exact.
- Greedy decoding makes the answers deterministic so equality is a valid test.
- Positions are global document order; eviction removes a section's position range
  and leaves holes, which llama.cpp handles fine.
