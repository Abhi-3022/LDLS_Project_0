// driver.cpp — Project 0 shared driver (HTML / structured KV-cache)
// ---------------------------------------------------------------------------
// Reads a JSON document produced by parser.py, prefills its segments into the
// KV cache tagged with per-subtree sequence ids, answers a question about one
// section, and runs the two required tests:
//
//   TEST 1 (isolation): change a DIFFERENT section, ask the same question about
//                       the target section -> the answer must be UNCHANGED.
//   TEST 2 (eviction) : evict an unrelated section from the KV cache by its
//                       position range -> KV memory drops, answer UNCHANGED.
//
// Decoding is GREEDY so comparisons are deterministic.
//
// How sequence ids encode the tree (see parser.py):
//   seq 0        = shared / intro context
//   seq 1..K     = one per document section (the evictable subtrees)
//   a QUESTION about section S is tagged with seq ids {0, S}, so it attends to
//   the shared context + section S only, never the other sections.
//
// Build: drop into llama.cpp/examples/doc-driver/ with the CMakeLists.txt,
//        add `add_subdirectory(doc-driver)` to examples/CMakeLists.txt, rebuild.
// Run:   ./build/bin/llama-doc-driver <model.gguf> dog.json
//        ./build/bin/llama-doc-driver <model.gguf> dog.json --ask 3 --change 6 --evict 5
//
// API NOTE: lines marked // [API] use llama.cpp names that change between
// versions. If the build fails on one, grep your include/llama.h for the
// nearby name. Common swaps are listed next to each one.
// ---------------------------------------------------------------------------

#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// Minimal JSON reader (enough for parser.py's output; no dependencies)
// ===========================================================================
struct Json {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    double num = 0; bool bl = false; std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;   // preserves order
    const Json* find(const std::string& k) const {
        for (auto& kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

struct JsonParser {
    const std::string& s; size_t i = 0;
    explicit JsonParser(const std::string& in) : s(in) {}
    void ws() { while (i < s.size() && (s[i]==' '||s[i]=='\t'||s[i]=='\n'||s[i]=='\r')) i++; }

    Json parse() { ws(); return value(); }

    Json value() {
        ws();
        char c = s[i];
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') { Json j; j.t = Json::STR; j.str = string(); return j; }
        if (c == 't' || c == 'f') { Json j; j.t = Json::BOOL; j.bl = (c=='t'); i += (c=='t'?4:5); return j; }
        if (c == 'n') { i += 4; return Json(); }
        return number();
    }
    Json object() {
        Json j; j.t = Json::OBJ; i++; ws();
        if (s[i] == '}') { i++; return j; }
        while (true) {
            ws(); std::string k = string(); ws(); i++; /* ':' */
            j.obj.emplace_back(k, value()); ws();
            if (s[i] == ',') { i++; continue; }
            i++; break; /* '}' */
        }
        return j;
    }
    Json array() {
        Json j; j.t = Json::ARR; i++; ws();
        if (s[i] == ']') { i++; return j; }
        while (true) {
            j.arr.push_back(value()); ws();
            if (s[i] == ',') { i++; continue; }
            i++; break; /* ']' */
        }
        return j;
    }
    std::string string() {
        std::string out; i++; /* opening quote */
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') break;
            if (c == '\\') {
                char e = s[i++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case '/': out += '/';  break;
                    case '"': out += '"';  break;
                    case '\\': out += '\\'; break;
                    case 'u': {              // \uXXXX -> UTF-8 (BMP only)
                        unsigned v = (unsigned) std::stoul(s.substr(i, 4), nullptr, 16); i += 4;
                        if (v < 0x80) out += (char) v;
                        else if (v < 0x800) { out += (char)(0xC0|(v>>6)); out += (char)(0x80|(v&0x3F)); }
                        else { out += (char)(0xE0|(v>>12)); out += (char)(0x80|((v>>6)&0x3F)); out += (char)(0x80|(v&0x3F)); }
                        break;
                    }
                    default: out += e;
                }
            } else out += c;
        }
        return out;
    }
    Json number() {
        size_t j0 = i;
        while (i < s.size() && (isdigit((unsigned char)s[i])||s[i]=='-'||s[i]=='+'||s[i]=='.'||s[i]=='e'||s[i]=='E')) i++;
        Json j; j.t = Json::NUM; j.num = std::stod(s.substr(j0, i - j0)); return j;
    }
};

// ===========================================================================
// llama helpers
// ===========================================================================
static std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& t, bool bos) {
    int n = -llama_tokenize(vocab, t.c_str(), (int)t.size(), nullptr, 0, bos, true);
    std::vector<llama_token> out(n);
    llama_tokenize(vocab, t.c_str(), (int)t.size(), out.data(), (int)out.size(), bos, true);
    return out;
}
static std::string piece(const llama_vocab* vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    return std::string(buf, n > 0 ? n : 0);
}

// a document segment
struct Seg { int seq; std::string text; };

// load segments + legend from the JSON file
static bool load_doc(const std::string& path, std::vector<Seg>& segs,
                     std::map<int,std::string>& legend) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    std::stringstream ss; ss << f.rdbuf();
    std::string data = ss.str();
    Json root = JsonParser(data).parse();
    const Json* S = root.find("segments");
    if (!S || S->t != Json::ARR) { fprintf(stderr, "no segments array\n"); return false; }
    for (auto& e : S->arr) {
        const Json* sq = e.find("seq_id"); const Json* tx = e.find("text");
        if (sq && tx) segs.push_back({ (int)sq->num, tx->str });
    }
    const Json* L = root.find("seq_legend");
    if (L && L->t == Json::OBJ)
        for (auto& kv : L->obj) legend[std::stoi(kv.first)] = kv.second.str;
    return !segs.empty();
}

// ---------------------------------------------------------------------------
// a thin batch helper
struct Batch {
    llama_batch b;
    explicit Batch(int cap) { b = llama_batch_init(cap, 0, 2); b.n_tokens = 0; }
    ~Batch() { llama_batch_free(b); }
    void clear() { b.n_tokens = 0; }
    void add(llama_token tok, llama_pos pos, std::vector<int> seqs, bool logits) {
        int k = b.n_tokens;
        b.token[k] = tok; b.pos[k] = pos; b.n_seq_id[k] = (int)seqs.size();
        for (size_t s = 0; s < seqs.size(); s++) b.seq_id[k][s] = seqs[s];
        b.logits[k] = logits ? 1 : 0;
        b.n_tokens++;
    }
};

// The document layout after prefill, using the SHARED-PREFIX pattern:
//   - the shared intro (seq-0 segments + BOS) is decoded once, positions 0..prefix_len-1
//   - for each section k: the prefix is COPIED into seq k (llama_memory_seq_cp), then the
//     section's tokens are appended in seq k at positions prefix_len.. (consecutive)
// So each section's sequence contains exactly: shared prefix + that section. Positions stay
// consecutive from 0 within every sequence (llama.cpp requires this), and the prefix cells
// are physically shared across sequences in the KV cache.
struct Doc {
    int prefix_len = 0;                        // shared prefix tokens (incl BOS)
    std::map<int,int> content_len;             // section seq -> its own token count
    std::map<int,llama_pos> seq_end;           // section seq -> next free position
};

// corrupt_seq / dummy: if set, section `corrupt_seq` is prefilled with the SAME
// NUMBER of tokens but all replaced by `dummy`. This destroys that section's
// content while keeping the KV-cache size (n_kv) identical, so the isolation
// test is not disturbed by floating-point non-determinism from a size change.
static Doc prefill(llama_context* ctx, const llama_vocab* vocab,
                   const std::vector<Seg>& segs, int n_batch,
                   int corrupt_seq = -1, llama_token dummy = -1) {
    Doc d;
    llama_memory_t mem = llama_get_memory(ctx);
    Batch batch(n_batch);
    auto flush = [&]() { if (batch.b.n_tokens > 0) { llama_decode(ctx, batch.b); batch.clear(); } };

    // 1) shared prefix: BOS + every seq-0 segment, in sequence 0
    llama_pos pos = 0;
    llama_token bos = llama_vocab_bos(vocab);               // [API] older: llama_token_bos
    if (bos >= 0) batch.add(bos, pos++, {0}, false);
    for (const auto& sg : segs) {
        if (sg.seq != 0) continue;
        for (llama_token t : tokenize(vocab, sg.text, false)) {
            if (batch.b.n_tokens >= n_batch) flush();
            batch.add(t, pos++, {0}, false);
        }
    }
    flush();
    d.prefix_len = pos;

    // 2) each section: copy the prefix into its sequence, then append the section
    std::set<int> secs;
    for (const auto& s : segs) if (s.seq > 0) secs.insert(s.seq);
    for (int k : secs) {
        llama_memory_seq_cp(mem, 0, k, 0, d.prefix_len);    // share the prefix with seq k
        llama_pos p = d.prefix_len;
        for (const auto& sg : segs) {
            if (sg.seq != k) continue;
            std::vector<llama_token> tt = tokenize(vocab, sg.text, false);
            if (k == corrupt_seq && dummy >= 0)
                for (auto& t : tt) t = dummy;           // same count, garbage content
            for (llama_token t : tt) {
                if (batch.b.n_tokens >= n_batch) flush();
                batch.add(t, p++, {k}, false);
            }
        }
        flush();
        d.content_len[k] = (int)(p - d.prefix_len);
        d.seq_end[k] = p;
    }
    return d;
}

// Ask a question about section `target`. The question goes in seq `target` ONLY, appended
// after that section (so it attends to the shared prefix + that section, nothing else).
// The question/answer tokens are removed afterwards so the cache is clean for the next ask.
static std::vector<llama_token> ask(llama_context* ctx, const llama_vocab* vocab,
                                    const Doc& d, int target, const std::string& q,
                                    int n_predict) {
    const int n_vocab = llama_vocab_n_tokens(vocab);        // [API] older: llama_n_vocab
    llama_memory_t mem = llama_get_memory(ctx);
    Batch batch(512);
    llama_pos pos = d.seq_end.count(target) ? d.seq_end.at(target) : d.prefix_len;
    const llama_pos qstart = pos;

    std::vector<llama_token> qtok = tokenize(vocab, q, false);
    for (size_t j = 0; j < qtok.size(); j++)
        batch.add(qtok[j], pos++, {target}, j + 1 == qtok.size());
    llama_decode(ctx, batch.b);

    std::vector<llama_token> out;
    int last_idx = (int)qtok.size() - 1;
    for (int step = 0; step < n_predict; step++) {
        const float* logits = llama_get_logits_ith(ctx, last_idx);
        llama_token best = 0; float bv = logits[0];
        for (int v = 1; v < n_vocab; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
        if (llama_vocab_is_eog(vocab, best)) break;         // [API] older: llama_token_is_eog
        out.push_back(best);
        batch.clear();
        batch.add(best, pos++, {target}, true);
        llama_decode(ctx, batch.b);
        last_idx = 0;
    }
    llama_memory_seq_rm(mem, target, qstart, pos);          // clean up this ask
    return out;
}

static std::string detok(const llama_vocab* vocab, const std::vector<llama_token>& t) {
    std::string s; for (llama_token x : t) s += piece(vocab, x); return s;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <doc.json> "
                        "[--ask S] [--change B] [--evict C] [--q \"question\"] [-n N]\n", argv[0]);
        return 1;
    }
    std::string model_path = argv[1], json_path = argv[2];
    int ask_seq = -1, change_seq = -1, evict_seq = -1, n_predict = 48;
    std::string question;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--ask" && i+1<argc) ask_seq = atoi(argv[++i]);
        else if (a == "--change" && i+1<argc) change_seq = atoi(argv[++i]);
        else if (a == "--evict" && i+1<argc) evict_seq = atoi(argv[++i]);
        else if (a == "--q" && i+1<argc) question = argv[++i];
        else if (a == "-n" && i+1<argc) n_predict = atoi(argv[++i]);
    }

    std::vector<Seg> segs; std::map<int,std::string> legend;
    if (!load_doc(json_path, segs, legend)) return 1;

    // content sections = seq ids > 0, in order
    std::set<int> seqset; for (auto& s : segs) seqset.insert(s.seq);
    std::vector<int> sections; for (int s : seqset) if (s > 0) sections.push_back(s);
    if (sections.empty()) { fprintf(stderr, "document has no sections > 0\n"); return 1; }
    if (ask_seq    < 0) ask_seq    = sections.front();
    if (change_seq < 0) change_seq = sections.back();
    if (evict_seq  < 0) evict_seq  = sections[sections.size() > 1 ? sections.size()-2 : 0];

    printf("Document sections:\n");
    for (int s : seqset) printf("  seq %d : %s\n", s, legend.count(s) ? legend[s].c_str() : "(section)");
    printf("\nask about seq %d (%s) | change seq %d (%s) | evict seq %d (%s)\n\n",
           ask_seq, legend[ask_seq].c_str(), change_seq, legend[change_seq].c_str(),
           evict_seq, legend[evict_seq].c_str());

    if (question.empty())
        question = "\n\nQuestion: In one short sentence, what does the section \"" +
                   (legend.count(ask_seq) ? legend[ask_seq] : std::string("this")) +
                   "\" say?\nAnswer:";

    // ---- load model (CPU) ----
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);   // [API]
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);                   // [API]

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 4096; cp.n_seq_max = 64;                     // [API] n_seq_max
    cp.kv_unified = true;  // REQUIRED: our question tokens belong to 2 seqs ({0,target}),
                           // which "couples" sequences — only a unified KV cache supports that.
    llama_context* ctx = llama_init_from_model(model, cp);                     // [API]
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const int n_layer = llama_model_n_layer(model);                           // [API] older: llama_n_layer
    const int n_embd  = llama_model_n_embd(model);                            // [API] older: llama_n_embd

    // =======================================================================
    // TEST 1 — isolation: change a different section, ask about the target
    // =======================================================================
    printf("================ TEST 1: isolation ================\n");
    llama_memory_clear(llama_get_memory(ctx), true);
    Doc d1 = prefill(ctx, vocab, segs, cp.n_batch);
    std::vector<llama_token> a1 = ask(ctx, vocab, d1, ask_seq, question, n_predict);

    // corrupt the 'change' section's tokens (same count), re-prefill, ask again
    llama_token dummy = tokenize(vocab, "banana", false).front();
    llama_memory_clear(llama_get_memory(ctx), true);
    Doc d2 = prefill(ctx, vocab, segs, cp.n_batch, change_seq, dummy);
    std::vector<llama_token> a2 = ask(ctx, vocab, d2, ask_seq, question, n_predict);

    printf("answer (original) : %s\n", detok(vocab, a1).c_str());
    printf("answer (after change): %s\n", detok(vocab, a2).c_str());
    bool t1 = (a1 == a2);
    printf("TEST 1: %s  (changing seq %d did %saffect the answer about seq %d)\n\n",
           t1 ? "PASS ✓" : "FAIL ✗", change_seq, t1 ? "NOT " : "", ask_seq);

    // =======================================================================
    // TEST 2 — eviction: drop an unrelated section, memory falls, answer holds
    // =======================================================================
    printf("================ TEST 2: eviction =================\n");
    llama_memory_clear(llama_get_memory(ctx), true);
    Doc d3 = prefill(ctx, vocab, segs, cp.n_batch);

    // distinct KV cells = shared prefix (counted once) + each section's own tokens
    auto cells = [&](const Doc& d, int removed)->long {
        long c = d.prefix_len;
        for (auto& kv : d.content_len) if (kv.first != removed) c += kv.second;
        return c;
    };
    auto kv_bytes = [&](long c)->double {            // approx upper bound (ignores GQA): 2(K+V)*layers*embd*fp16
        return (double)c * 2.0 * n_layer * n_embd * 2.0;
    };

    long before = cells(d3, -1);
    std::vector<llama_token> a3 = ask(ctx, vocab, d3, ask_seq, question, n_predict);

    // evict the unrelated section: remove its tokens (positions >= prefix_len) from its sequence.
    // The shared prefix cells stay (they still belong to the other sections' sequences).
    llama_memory_seq_rm(llama_get_memory(ctx), evict_seq, d3.prefix_len, -1);
    long after = cells(d3, evict_seq);
    std::vector<llama_token> a4 = ask(ctx, vocab, d3, ask_seq, question, n_predict);

    printf("KV cells: before=%ld  after=%ld  (freed %ld tokens of seq %d)\n",
           before, after, before - after, evict_seq);
    printf("KV memory (approx): before=%.2f MB  after=%.2f MB\n",
           kv_bytes(before)/1e6, kv_bytes(after)/1e6);
    printf("answer (before evict): %s\n", detok(vocab, a3).c_str());
    printf("answer (after evict) : %s\n", detok(vocab, a4).c_str());
    bool t2 = (a3 == a4) && (after < before);
    printf("TEST 2: %s  (evicting seq %d freed memory and left the answer %s)\n\n",
           t2 ? "PASS ✓" : "FAIL ✗", evict_seq, (a3==a4) ? "unchanged" : "CHANGED");

    printf("================ RESULT: %s ================\n",
           (t1 && t2) ? "BOTH TESTS PASS" : "SEE FAILURES ABOVE");

    llama_free(ctx);
    llama_model_free(model);                                                  // [API] older: llama_free_model
    llama_backend_free();
    return (t1 && t2) ? 0 : 2;
}
