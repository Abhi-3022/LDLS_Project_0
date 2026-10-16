// two_seq.cpp — Project 0 (Common Foundation) experiment
// ---------------------------------------------------------------------------
// Goal: prove that two prompts run as TWO SEQUENCES in ONE batch produce the
// SAME greedy output as running each prompt ALONE. This demonstrates that
// sequence IDs isolate sequences in the KV cache / attention mask — the whole
// premise of the structured-attention project.
//
// Decoding is GREEDY (argmax) so the comparison is deterministic. No sampler,
// no randomness — if isolation holds, the token streams must be identical.
//
// Build: see README.md (drop into llama.cpp/examples/two-seq/ and rebuild).
// Run:   ./build/bin/llama-two-seq <model.gguf> "prompt A" "prompt B"
//
// NOTE ON API VERSIONS: llama.cpp's C API is renamed fairly often. The lines
// most likely to differ in your checkout are marked  // [API]  — if the build
// fails on one, open include/llama.h and grep for the nearby name. Common swaps:
//   llama_model_load_from_file   <-> llama_load_model_from_file
//   llama_init_from_model        <-> llama_new_context_with_model
//   llama_model_get_vocab        (older code used the model directly)
//   llama_kv_self_clear(ctx)     <-> llama_kv_cache_clear(ctx)
//                                 <-> llama_memory_clear(llama_get_memory(ctx), true)
//   llama_vocab_n_tokens         <-> llama_n_vocab
// ---------------------------------------------------------------------------

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---- small helpers ---------------------------------------------------------

static std::vector<llama_token> tokenize(const llama_vocab * vocab,
                                         const std::string & text,
                                         bool add_bos) {
    // First call with a null buffer returns the (negative) token count.
    int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(),
                            nullptr, 0, add_bos, /*parse_special=*/true);
    std::vector<llama_token> out(n);
    llama_tokenize(vocab, text.c_str(), (int) text.size(),
                   out.data(), (int) out.size(), add_bos, /*parse_special=*/true);
    return out;
}

static std::string token_to_text(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, /*special=*/true);
    return std::string(buf, n > 0 ? n : 0);
}

// Greedy: pick the highest-logit token for the given batch row index.
static llama_token argmax_at(llama_context * ctx, int n_vocab, int idx) {
    const float * logits = llama_get_logits_ith(ctx, idx);
    llama_token best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
}

// ---- the core routine ------------------------------------------------------
// Run `prompts` as parallel sequences (one seq_id each) and greedily generate
// n_predict tokens for each. Returns the generated token ids per prompt.
static std::vector<std::vector<llama_token>>
run_prompts(llama_context * ctx, const llama_vocab * vocab,
            const std::vector<std::string> & prompts, int n_predict) {

    const int n_seq   = (int) prompts.size();
    const int n_vocab = llama_vocab_n_tokens(vocab);                  // [API]

    // Start from a clean KV cache so runs don't contaminate each other.
    llama_memory_clear(llama_get_memory(ctx), true);                 // [API]

    // Tokenize every prompt (add BOS once at the start of each sequence).
    std::vector<std::vector<llama_token>> toks(n_seq);
    int total_prompt_tokens = 0;
    for (int s = 0; s < n_seq; s++) {
        toks[s] = tokenize(vocab, prompts[s], /*add_bos=*/true);
        total_prompt_tokens += (int) toks[s].size();
    }

    // One batch big enough for the prefill (all prompt tokens at once).
    // third arg = max seq_ids PER TOKEN; here each token belongs to exactly 1.
    llama_batch batch = llama_batch_init(std::max(total_prompt_tokens, n_seq), 0, 1);

    std::vector<llama_pos> pos(n_seq, 0);      // per-sequence position counter
    std::vector<int>       last_idx(n_seq, -1); // batch row holding each seq's last logits

    // ---- PREFILL: load every prompt, each token tagged with its seq_id ----
    batch.n_tokens = 0;
    for (int s = 0; s < n_seq; s++) {
        for (size_t i = 0; i < toks[s].size(); i++) {
            const int k = batch.n_tokens;
            batch.token[k]     = toks[s][i];
            batch.pos[k]       = pos[s]++;          // positions are PER SEQUENCE
            batch.n_seq_id[k]  = 1;
            batch.seq_id[k][0] = s;                 // <<< this is the isolation
            const bool is_last = (i + 1 == toks[s].size());
            batch.logits[k]    = is_last ? 1 : 0;   // only need logits at the end
            if (is_last) last_idx[s] = k;
            batch.n_tokens++;
        }
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed during prefill\n");
        llama_batch_free(batch);
        return {};
    }

    std::vector<std::vector<llama_token>> out(n_seq);
    std::vector<llama_token> cur(n_seq);

    // First generated token for each sequence.
    for (int s = 0; s < n_seq; s++) {
        cur[s] = argmax_at(ctx, n_vocab, last_idx[s]);
        out[s].push_back(cur[s]);
    }

    // ---- GENERATE: one new token per sequence per step --------------------
    for (int step = 1; step < n_predict; step++) {
        batch.n_tokens = 0;
        for (int s = 0; s < n_seq; s++) {
            const int k = batch.n_tokens;
            batch.token[k]     = cur[s];
            batch.pos[k]       = pos[s]++;
            batch.n_seq_id[k]  = 1;
            batch.seq_id[k][0] = s;
            batch.logits[k]    = 1;
            last_idx[s]        = k;
            batch.n_tokens++;
        }
        if (llama_decode(ctx, batch) != 0) break;
        for (int s = 0; s < n_seq; s++) {
            cur[s] = argmax_at(ctx, n_vocab, last_idx[s]);
            out[s].push_back(cur[s]);
        }
    }

    llama_batch_free(batch);
    return out;
}

static void print_stream(const llama_vocab * vocab,
                         const std::vector<llama_token> & toks) {
    for (llama_token t : toks) printf("%s", token_to_text(vocab, t).c_str());
    printf("\n");
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> \"prompt A\" \"prompt B\" [n_predict]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string prompt_a   = argv[2];
    const std::string prompt_b   = argv[3];
    const int n_predict          = (argc > 4) ? atoi(argv[4]) : 32;

    llama_backend_init();

    // --- load the model on CPU ---
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;                                        // CPU only
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams); // [API]
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);         // [API]

    // --- create a context that allows at least 2 sequences ---
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 4096;   // total KV cells across ALL sequences
    cparams.n_batch   = 4096;   // must hold the whole prefill in one decode
    cparams.n_seq_max = 8;      // [API] max parallel sequences (drop if absent)
    llama_context * ctx = llama_init_from_model(model, cparams);      // [API]
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // --- 1) both prompts together, as two sequences in one batch ---
    auto together = run_prompts(ctx, vocab, {prompt_a, prompt_b}, n_predict);
    // --- 2) each prompt alone (separate runs) ---
    auto only_a   = run_prompts(ctx, vocab, {prompt_a}, n_predict);
    auto only_b   = run_prompts(ctx, vocab, {prompt_b}, n_predict);

    printf("\n===== TOGETHER (seq 0) =====\n"); print_stream(vocab, together[0]);
    printf("===== ALONE    (A)     =====\n");   print_stream(vocab, only_a[0]);
    printf("\n===== TOGETHER (seq 1) =====\n"); print_stream(vocab, together[1]);
    printf("===== ALONE    (B)     =====\n");   print_stream(vocab, only_b[0]);

    const bool match_a = (together[0] == only_a[0]);
    const bool match_b = (together[1] == only_b[0]);

    printf("\n---------------------------------------------\n");
    printf("seq 0 vs A alone : %s\n", match_a ? "MATCH ✓" : "DIFFER ✗");
    printf("seq 1 vs B alone : %s\n", match_b ? "MATCH ✓" : "DIFFER ✗");
    printf("RESULT: %s\n", (match_a && match_b)
           ? "PASS — sequences are isolated by seq_id"
           : "FAIL — sequences leaked into each other");
    printf("---------------------------------------------\n");

    llama_free(ctx);
    llama_model_free(model);                                         // [API] (older: llama_free_model)
    llama_backend_free();
    return (match_a && match_b) ? 0 : 2;
}
