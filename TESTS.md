# The tests

The project requires two behavioural tests. Both are implemented inside
`driver.cpp` and run automatically on every invocation; `run_tests.ps1` wraps the
driver to also record CPU run time.

## Test 1 — Isolation
> change subtree B, ask about subtree A, and check that the output is unchanged.

**What it does.** Prefill the document; ask a question about section A (`--ask`).
Then prefill again with section B (`--change`) **corrupted**, and ask the same
question about A. The two answers must be identical.

**Why it proves isolation.** A's question lives in sequence `{A}` and can only
attend to keys sharing that sequence (A's own tokens + the shared prefix). It can
never attend to B's tokens (sequence `{B}`). So corrupting B cannot change A's
answer — unless the sequence isolation is broken.

**Implementation detail that matters.** B is corrupted at the **token level**: its
tokens are replaced by a dummy token **keeping the exact same count**. This keeps the
KV-cache size (`n_kv`) identical between the two runs. If B's *length* changed, the
masked attention over A would sum a different number of terms and floating-point
non-associativity would perturb greedy decoding — a false failure, not a real leak.
Holding the token count fixed makes the comparison exact.

## Test 2 — Eviction
> remove an unrelated subtree; the answer holds, and KV memory drops.

**What it does.** Prefill the document; ask about A; record KV cells. Evict section C
(`--evict`) from the cache by its position range (`llama_memory_seq_rm`); record KV
cells again; ask about A once more. The answer must be unchanged and the cell count
must drop.

**Why it proves safe eviction.** C's cells belong only to sequence `{C}`; the shared
prefix cells survive (they still belong to the other sections' sequences). A never
attended to C, so freeing C's cells changes nothing for A — while reclaiming memory.
This is "subtrees are contiguous intervals → evict the interval" (Lecture 11).

## Running

```powershell
cd "C:\Users\roop1\Desktop\project 0\llama.cpp"
build\bin\Release\llama-doc-driver.exe <model.gguf> <doc.json> --ask A --change B --evict C
```
`--ask/--change/--evict` are sequence ids (three distinct ones); the driver prints the
`seq_legend` at startup so you can choose. Decoding is greedy, so the comparisons are
deterministic.

Expected tail:
```
TEST 1: PASS   (changing seq B did NOT affect the answer about seq A)
TEST 2: PASS   KV cells: before=N after=M (freed K tokens of seq C)
RESULT: BOTH TESTS PASS
```

## Report numbers these tests produce
- Test 1: pass/fail.
- Test 2: pass/fail, `KV cells before/after`, approx `KV memory before/after`.
- CPU run time: from `run_tests.ps1` (uses `Measure-Command`).
