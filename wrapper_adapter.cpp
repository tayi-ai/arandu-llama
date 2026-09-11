// Adapter control and teacher-forced scoring.
//
// These two capabilities are what training needs and inference does not, which
// is why upstream has no reason to carry them: an inference binding scores what
// the model wrote, and never changes a weight after load.
//
// Zeroth-order optimisation needs the opposite of both. It scores a completion
// somebody else wrote -- the loss has to be of the same quantity a
// backpropagated trainer reports, or the two cannot be compared -- and it moves
// the adapter between one forward pass and the next, with the base weights
// staying on the cards.
//
// They live in a file of their own so this fork's diff against upstream is this
// file plus a few declarations. A fork that edits the middle of wrapper.cpp is a
// fork that conflicts on every upstream release, and this one has to follow
// llama.cpp closely enough to keep loading the same models.

#include "wrapper.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// The handle layouts wrapper.cpp allocates. They are repeated rather than
// shared through a header because upstream keeps them private to that
// translation unit, and adding a header would be a second point of conflict.
// If either gains a field, the change is caught by the compiler here.
typedef struct {
    llama_model* model;
} llama_wrapper_model_t;

typedef struct {
    llama_context* ctx;
    llama_model* model;
    std::vector<int> cached_tokens;
} llama_wrapper_context_t;

// The error slot wrapper.cpp owns. Declared, not defined: one string, one
// meaning, whichever file wrote last.
extern std::string g_last_error;

extern "C" {

void* llama_wrapper_adapter_load(void* model, const char* path) {
    if (!model || !path) {
        g_last_error = "Model and adapter path cannot be null";
        return nullptr;
    }
    try {
        auto wrapper = static_cast<llama_wrapper_model_t*>(model);
        // The adapter is read against the model it will be applied to, so a
        // mismatch in architecture or tensor shape is refused here rather than
        // surfacing later as a wrong number.
        llama_adapter_lora* adapter = llama_adapter_lora_init(wrapper->model, path);
        if (!adapter) {
            g_last_error = "Failed to load adapter from: " + std::string(path);
            return nullptr;
        }
        return adapter;
    } catch (const std::exception& e) {
        g_last_error = "Exception loading adapter: " + std::string(e.what());
        return nullptr;
    }
}

void llama_wrapper_adapter_free(void* adapter) {
    if (!adapter) return;
    llama_adapter_lora_free(static_cast<llama_adapter_lora*>(adapter));
}

// llama_wrapper_adapter_meta reads one GGUF metadata value from a loaded
// adapter.
//
// It exists because of how the scale is interpreted. llama.cpp applies
// adapter_scale * alpha / rank when the file declares a non-zero alpha, and the
// bare adapter_scale when it does not. PEFT applies alpha / r. So a converted
// adapter that lost its alpha key is applied at a different magnitude than the
// trainer applied it, every weight is off by one constant factor, and the loss
// that comes out belongs to a different model -- not to a different quantisation
// of the same one.
//
// That failure is silent. It produces a plausible number, and a plausible number
// is what gets compared against a measured reference and believed. The keys worth
// reading are "adapter.lora.alpha" and "adapter.type".
//
// Returns the length written, or -1 when the key is absent.
int llama_wrapper_adapter_meta(void* adapter, const char* key, char* buffer, int buffer_size) {
    if (!adapter || !key || !buffer || buffer_size <= 0) {
        g_last_error = "Adapter, key and a non-empty buffer are required";
        return -1;
    }
    return static_cast<int>(llama_adapter_meta_val_str(
        static_cast<llama_adapter_lora*>(adapter), key, buffer, static_cast<size_t>(buffer_size)));
}

int llama_wrapper_model_describe(void* model, char* buffer, int buffer_size) {
    if (!model || !buffer || buffer_size <= 0) {
        g_last_error = "Model and a non-empty buffer are required";
        return -1;
    }
    auto wrapper = static_cast<llama_wrapper_model_t*>(model);
    return static_cast<int>(llama_model_desc(wrapper->model, buffer, static_cast<size_t>(buffer_size)));
}

int llama_wrapper_adapters_set(void* ctx, void** adapters, float* scales, int n_adapters) {
    if (!ctx) {
        g_last_error = "Context cannot be null";
        return -1;
    }
    if (n_adapters < 0 || (n_adapters > 0 && (!adapters || !scales))) {
        g_last_error = "Adapter and scale arrays are required when n_adapters is positive";
        return -1;
    }
    try {
        auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
        if (!wrapper->ctx) {
            g_last_error = "Context has been freed";
            return -1;
        }
        // Passing zero adapters is how every adapter is removed, which is what
        // a caller wants after a probe: an engine left perturbed is a model the
        // next measurement reads without anybody meaning to.
        std::vector<llama_adapter_lora*> handles(static_cast<size_t>(n_adapters));
        for (int i = 0; i < n_adapters; i++) {
            handles[static_cast<size_t>(i)] = static_cast<llama_adapter_lora*>(adapters[i]);
            if (!handles[static_cast<size_t>(i)]) {
                g_last_error = "Adapter " + std::to_string(i) + " is null";
                return -1;
            }
        }
        int32_t applied = llama_set_adapters_lora(
            wrapper->ctx,
            n_adapters > 0 ? handles.data() : nullptr,
            static_cast<size_t>(n_adapters),
            scales);
        if (applied < 0) {
            g_last_error = "llama_set_adapters_lora refused the request";
            return -1;
        }
        return static_cast<int>(applied);
    } catch (const std::exception& e) {
        g_last_error = "Exception setting adapters: " + std::string(e.what());
        return -1;
    }
}

// llama_wrapper_score computes the mean negative log likelihood the model
// assigns to a token sequence it did not produce.
//
// This is teacher forcing. Every position is decoded with its logits requested,
// and the log probability read at position i is the one the model gave to the
// token that actually followed at i+1. Position zero is never scored: nothing
// preceded it to predict it.
//
// n_skip drops the leading tokens of the prompt from the average, so the loss is
// over the completion alone. A loss that included the prompt would reward a
// policy for predicting text it was handed.
//
// The log softmax is computed with the max subtracted before exponentiating. The
// naive form overflows on a vocabulary this size, and the overflow arrives as
// inf, then as nan, and a nan loss in a zeroth-order step is an estimate that
// silently poisons every parameter it touches.
int llama_wrapper_score(void* ctx, const int* tokens, int n_tokens, int n_skip,
                        double* out_sum, int* out_count) {
    if (!ctx || !tokens || !out_sum || !out_count) {
        g_last_error = "Context, tokens and both outputs are required";
        return -1;
    }
    *out_sum = 0.0;
    *out_count = 0;
    if (n_tokens < 2) {
        g_last_error = "Scoring needs at least two tokens: one to condition on and one to predict";
        return -1;
    }
    if (n_skip < 0 || n_skip >= n_tokens) {
        g_last_error = "n_skip has to leave at least one token to score";
        return -1;
    }
    try {
        auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
        if (!wrapper->ctx) {
            g_last_error = "Context has been freed";
            return -1;
        }
        const int n_ctx = static_cast<int>(llama_n_ctx(wrapper->ctx));
        if (n_tokens > n_ctx) {
            g_last_error = "Sequence of " + std::to_string(n_tokens) +
                           " tokens exceeds the context of " + std::to_string(n_ctx);
            return -1;
        }
        const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
        const int n_vocab = static_cast<int>(llama_vocab_n_tokens(vocab));
        if (n_vocab <= 0) {
            g_last_error = "The model reports an empty vocabulary";
            return -1;
        }

        // The KV cache is cleared rather than reused. Prefix reuse is what makes
        // generation fast, and it is exactly wrong here: the cache was filled
        // under whatever adapter was applied when it was written, and reading it
        // after the adapter moved would score a mixture of two models.
        llama_memory_clear(llama_get_memory(wrapper->ctx), true);
        // The prefix bookkeeping has to go with the cache it describes. A
        // generate call reads cached_tokens to decide how many leading tokens it
        // may skip; left populated over an emptied cache, it skips tokens that
        // are no longer there and decodes on top of nothing.
        wrapper->cached_tokens.clear();

        // The sequence is decoded in windows, and this is not an optimisation.
        //
        // A flagged position costs n_vocab floats of output. This model's
        // vocabulary is 248320 tokens, so one position is 993280 bytes and a
        // 2477-token example flagged whole would ask for 2.46 GB of logits on a
        // card holding 15.32 GiB of weights.
        //
        // The second reason is harder to recover from. llama_context_params
        // carries n_outputs_max, which defaults to n_batch, and exceeding it is
        // a GGML_ASSERT inside output_reserve -- the process aborts rather than
        // returning an error a caller could handle. So the window is bounded by
        // n_batch as well as by memory.
        const int n_batch = static_cast<int>(llama_n_batch(wrapper->ctx));
        int window = n_batch > 0 ? n_batch : 512;
        // 128 MB of logits per window, which is the same order as the KV cache
        // this context already holds and small beside the weights.
        const int by_memory = static_cast<int>((128LL << 20) / (static_cast<long long>(n_vocab) * 4));
        if (by_memory > 0 && by_memory < window) window = by_memory;
        if (window < 1) window = 1;

        double sum = 0.0;
        int counted = 0;
        // Position i predicts token i+1, so the last position of the sequence
        // predicts nothing and is never flagged. The first scored position is
        // n_skip-1, which is the last token of the prompt: it is what predicts
        // the first token of the completion.
        const int first_scored = n_skip > 1 ? n_skip - 1 : 0;

        for (int start = 0; start < n_tokens; start += window) {
            const int count = std::min(window, n_tokens - start);
            llama_batch batch = llama_batch_init(count, 0, 1);
            for (int i = 0; i < count; i++) {
                const int position = start + i;
                batch.token[i] = tokens[position];
                batch.pos[i] = position;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                // Only the positions whose logits are actually read are flagged.
                // Flagging the rest would buy nothing and cost n_vocab floats
                // each.
                batch.logits[i] =
                    (position >= first_scored && position < n_tokens - 1) ? 1 : 0;
            }
            batch.n_tokens = count;

            // The KV cache carries across windows, which is what makes this a
            // single forward pass split into pieces rather than several passes.
            const int32_t decoded = llama_decode(wrapper->ctx, batch);
            if (decoded != 0) {
                llama_batch_free(batch);
                // 1 means no KV slot and 2 means aborted with ubatches already
                // in memory; neither leaves a state worth reading.
                g_last_error = "llama_decode failed with " + std::to_string(decoded) +
                               " on the window starting at " + std::to_string(start);
                return -1;
            }

            // Collect what has to be scored, then compute it in parallel.
            //
            // This is CPU work, not GPU work, and at this vocabulary it is not
            // small: one position is a pass to find the maximum and a pass of
            // 248320 exponentials, so a 2477-token example is on the order of a
            // billion operations. Serial, that costs about what the forward pass
            // on the cards costs, and doubles the price of every probe.
            // tools/perplexity/perplexity.cpp threads it for the same reason.
            std::vector<std::pair<int, int>> work;  // logit row, target token
            work.reserve(static_cast<size_t>(count));
            for (int i = 0; i < count; i++) {
                if (!batch.logits[i]) continue;
                const int position = start + i;
                const int target = tokens[position + 1];
                if (target < 0 || target >= n_vocab) {
                    llama_batch_free(batch);
                    g_last_error = "Token " + std::to_string(target) +
                                   " is outside the vocabulary";
                    return -1;
                }
                work.emplace_back(i, target);
            }

            std::vector<double> results(work.size(), 0.0);
            std::atomic<size_t> next(0);
            std::atomic<bool> failed(false);
            std::string failure;
            std::mutex failure_lock;

            auto compute = [&]() {
                for (;;) {
                    const size_t index = next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= work.size() || failed.load(std::memory_order_relaxed)) return;
                    const float* logits = llama_get_logits_ith(wrapper->ctx, work[index].first);
                    if (!logits) {
                        std::lock_guard<std::mutex> guard(failure_lock);
                        failure = "No logits at batch index " + std::to_string(work[index].first);
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                    // Log softmax with the maximum subtracted. The naive form
                    // overflows on a vocabulary of 248320, and the overflow
                    // arrives as inf, then as nan; a nan estimate poisons every
                    // parameter the step touches without failing anything.
                    //
                    // The sum accumulates in double where the reference uses
                    // float: over 248320 terms the float loses the tail, and this
                    // number is compared against a reference measured to sixteen
                    // digits.
                    float max_logit = logits[0];
                    for (int v = 1; v < n_vocab; v++) {
                        if (logits[v] > max_logit) max_logit = logits[v];
                    }
                    double denominator = 0.0;
                    for (int v = 0; v < n_vocab; v++) {
                        denominator += std::exp(static_cast<double>(logits[v] - max_logit));
                    }
                    const double log_prob =
                        static_cast<double>(logits[work[index].second] - max_logit) -
                        std::log(denominator);
                    if (!std::isfinite(log_prob)) {
                        std::lock_guard<std::mutex> guard(failure_lock);
                        failure = "The log probability at batch index " +
                                  std::to_string(work[index].first) + " is not finite";
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                    results[index] = log_prob;
                }
            };

            unsigned threads = std::thread::hardware_concurrency();
            if (threads == 0) threads = 1;
            if (threads > work.size()) threads = static_cast<unsigned>(work.size());
            if (threads > 1) {
                std::vector<std::thread> workers;
                workers.reserve(threads);
                // A thread that cannot be created throws, and a vector of
                // joinable threads destroyed by the unwinding calls
                // std::terminate -- the process aborts, in the middle of a
                // measurement, with no error a caller could report. Under memory
                // pressure on a shared card that is a real failure and not a
                // theoretical one.
                //
                // Whatever was created is joined, and the remainder is computed
                // on this thread. Fewer workers is slower; aborting is not.
                try {
                    for (unsigned t = 0; t < threads; t++) workers.emplace_back(compute);
                } catch (const std::exception&) {
                }
                if (workers.empty()) {
                    compute();
                } else {
                    compute();
                    for (auto& worker : workers) worker.join();
                }
            } else if (!work.empty()) {
                compute();
            }

            llama_batch_free(batch);
            if (failed.load(std::memory_order_relaxed)) {
                g_last_error = failure;
                return -1;
            }
            // Summed after the threads join, in a fixed order, so the total does
            // not depend on which worker finished first. Floating point addition
            // is not associative, and a loss that changes with thread scheduling
            // is a loss two runs cannot be compared on.
            for (double value : results) {
                sum += value;
                counted++;
            }
        }

        if (counted == 0) {
            g_last_error = "No position was scored";
            return -1;
        }

        // A mean of exactly log(n_vocab) is a model that produced nothing.
        //
        // That is the uniform distribution: every token equally likely, which is
        // what the graph returns when the forward pass did not actually run. It
        // is not a bad model -- a bad model is wrong in some direction, and this
        // is wrong in none.
        //
        // Measured on 2026-09-10, splitting a model across two Tesla T10s: the
        // score came back 11.76178354556442 for SmolLM3 in F16, in Q4_K_M and in
        // Q2_K alike, and log(128256) is 11.7617835455. Identical to ten places
        // across three different quantisations is not a coincidence, and the
        // failure was intermittent -- two runs in eight with the same command.
        // A quarter of the time it returned a number that reads as a loss.
        //
        // The tolerance is tight on purpose. A real model landing this close to
        // uniform by accident would be reporting the same thing anyway.
        const double uniform = std::log(static_cast<double>(n_vocab));
        const double mean = -sum / static_cast<double>(counted);
        if (std::fabs(mean - uniform) < 1e-9) {
            g_last_error = "The score is log(n_vocab) = " + std::to_string(uniform) +
                           ", which is the uniform distribution: the forward pass produced no "
                           "information. On CUDA this is what a model split across two devices "
                           "returns, intermittently. Run on one device.";
            return -1;
        }
        // The caller receives the sum and the count rather than the mean, so a
        // caller averaging over several examples weights them by length instead
        // of averaging averages.
        *out_sum = -sum;
        *out_count = counted;
        return 0;
    } catch (const std::exception& e) {
        g_last_error = "Exception scoring: " + std::string(e.what());
        return -1;
    }
}

}  // extern "C"

// Restores the embeddings flag on every exit path, exceptions included.
//
// A capture switches the flag on for its decode. A context left in embeddings
// mode would flag every row of the next score and reserve an embeddings buffer
// nobody reads, and a context created for embeddings would stop producing them
// if the flag were left off. So the creation value is put back before anything
// else happens on the context, whichever way the capture ends.
namespace {
struct embeddings_guard {
    llama_context* ctx;
    bool after;
    ~embeddings_guard() { llama_set_embeddings(ctx, after); }
};
}  // namespace

extern "C" {

int llama_wrapper_model_n_embd_out(void* model) {
    if (!model) {
        g_last_error = "Model cannot be null";
        return -1;
    }
    // Reads only ->model, the first member of both handle definitions.
    auto wrapper = static_cast<llama_wrapper_model_t*>(model);
    return static_cast<int>(llama_model_n_embd_out(wrapper->model));
}

// llama_wrapper_capture_final reads the per-token final representation of a
// token sequence: the hidden state after the final RMS normalisation and before
// the output projection.
//
// At the pinned llama.cpp commit that is the graph node named "result_norm"
// (src/models/smollm3.cpp, `cb(cur, "result_norm", -1); res->t_embd = cur;`).
// It is not observed through a scheduler callback. llama.cpp already copies
// exactly that tensor into the context's embeddings buffer when
// cparams.embeddings is on and the pooling type is NONE, so the flag is
// switched on for this decode and read back with llama_get_embeddings_ith.
// Measured bitwise equal to the tensor seen through cb_eval, 14/14 rows,
// max|diff| = 0, on SmolLM3-3B Q4_K_M on CPU. A callback would have had to be
// installed at context creation, and once installed it synchronises every
// split of every decode on that context for the rest of its life.
//
// Every row of every window is flagged as an output. In embeddings mode the
// batch allocator overrides a partial selection to all rows anyway
// (src/llama-batch.cpp, "embeddings required but some input tokens were not
// marked as outputs -> overriding"), and the choice is not free of consequence:
// a decode that flags a subset of rows runs the last layer's FFN over a
// different number of rows, and its values differ from the full-window rows by
// up to 1.9e-6 (measured, rows 3 and 7 of a 14-token sequence). The rule is
// therefore part of the capture version, and the positions decide only which
// rows are copied out.
//
// The window is bounded exactly as the scoring window is, because logits are
// reserved per output row in embeddings mode too (has_logits is unconditional
// in output_reserve), and the second bound is the same GGML_ASSERT.
int llama_wrapper_capture_final(void* ctx, const int* tokens, int n_tokens,
                                const int* positions, int n_positions,
                                bool embeddings_after,
                                float* out, long long out_floats) {
    if (!ctx || !tokens || !positions || !out) {
        g_last_error = "Context, tokens, positions and an output buffer are required";
        return -1;
    }
    if (n_tokens < 1) {
        g_last_error = "Capturing needs at least one token";
        return -1;
    }
    if (n_positions < 1) {
        g_last_error = "Capturing needs at least one position";
        return -1;
    }
    try {
        auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
        if (!wrapper->ctx || !wrapper->model) {
            g_last_error = "Context has been freed";
            return -1;
        }
        const int n_ctx = static_cast<int>(llama_n_ctx(wrapper->ctx));
        if (n_tokens > n_ctx) {
            g_last_error = "Sequence of " + std::to_string(n_tokens) +
                           " tokens exceeds the context of " + std::to_string(n_ctx);
            return -1;
        }
        // Strictly increasing, so a row is copied at most once and the single
        // pass over the windows below consumes the list in order.
        for (int k = 0; k < n_positions; k++) {
            if (positions[k] < 0 || positions[k] >= n_tokens) {
                g_last_error = "Position " + std::to_string(positions[k]) + " at index " +
                               std::to_string(k) + " is outside the sequence of " +
                               std::to_string(n_tokens) + " tokens";
                return -1;
            }
            if (k > 0 && positions[k] <= positions[k - 1]) {
                g_last_error = "Positions have to be strictly increasing; index " +
                               std::to_string(k) + " is " + std::to_string(positions[k]) +
                               " after " + std::to_string(positions[k - 1]);
                return -1;
            }
        }
        const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
        const int n_vocab = static_cast<int>(llama_vocab_n_tokens(vocab));
        if (n_vocab <= 0) {
            g_last_error = "The model reports an empty vocabulary";
            return -1;
        }
        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i] < 0 || tokens[i] >= n_vocab) {
                g_last_error = "Token " + std::to_string(tokens[i]) + " at position " +
                               std::to_string(i) + " is outside the vocabulary";
                return -1;
            }
        }
        // With pooling the tensor llama.cpp copies is result_embd_pooled, one
        // row per sequence. That is an embedding of the text, and the objective
        // this exists for needs the representation of each token.
        if (llama_pooling_type(wrapper->ctx) != LLAMA_POOLING_TYPE_NONE) {
            g_last_error = "The context pools its embeddings; a per-token capture needs pooling type NONE";
            return -1;
        }
        const int n_embd_out = static_cast<int>(llama_model_n_embd_out(wrapper->model));
        if (n_embd_out <= 0) {
            g_last_error = "The model reports no output embedding width";
            return -1;
        }
        // Exact, not at least: a buffer sized for another width or another
        // count is a buffer sized for another capture.
        const long long need = static_cast<long long>(n_positions) * n_embd_out;
        if (out_floats != need) {
            g_last_error = "Output buffer holds " + std::to_string(out_floats) +
                           " floats; the capture needs exactly " + std::to_string(need);
            return -1;
        }

        // The KV cache is cleared rather than reused. Prefix reuse is what makes
        // generation fast, and it is exactly wrong here: the cache was filled
        // under whatever adapter was applied when it was written, and reading it
        // after the adapter moved would capture a mixture of two models.
        llama_memory_clear(llama_get_memory(wrapper->ctx), true);
        // The prefix bookkeeping has to go with the cache it describes. A
        // generate call reads cached_tokens to decide how many leading tokens it
        // may skip; left populated over an emptied cache, it skips tokens that
        // are no longer there and decodes on top of nothing.
        wrapper->cached_tokens.clear();

        // The same two bounds as llama_wrapper_score, duplicated on purpose so
        // that this capability does not touch the scoring path: n_batch, because
        // n_outputs_max defaults to it and exceeding it is a GGML_ASSERT that
        // aborts the process; and 128 MB of logits per window, which are
        // reserved per output row in embeddings mode as well.
        const int n_batch = static_cast<int>(llama_n_batch(wrapper->ctx));
        int window = n_batch > 0 ? n_batch : 512;
        const int by_memory = static_cast<int>((128LL << 20) / (static_cast<long long>(n_vocab) * 4));
        if (by_memory > 0 && by_memory < window) window = by_memory;
        if (window < 1) window = 1;

        embeddings_guard guard{wrapper->ctx, embeddings_after};
        llama_set_embeddings(wrapper->ctx, true);

        int k = 0;  // next position not yet copied out
        for (int start = 0; start < n_tokens; start += window) {
            const int count = std::min(window, n_tokens - start);
            llama_batch batch = llama_batch_init(count, 0, 1);
            for (int i = 0; i < count; i++) {
                batch.token[i] = tokens[start + i];
                batch.pos[i] = start + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                // Every row is an output row, so batch row i is sequence
                // position start + i in the buffer that comes back.
                batch.logits[i] = 1;
            }
            batch.n_tokens = count;

            // The KV cache carries across windows, which is what makes this a
            // single forward pass split into pieces rather than several passes.
            const int32_t decoded = llama_decode(wrapper->ctx, batch);
            if (decoded != 0) {
                llama_batch_free(batch);
                g_last_error = "llama_decode failed with " + std::to_string(decoded) +
                               " on the window starting at " + std::to_string(start);
                return -1;
            }

            // The pointer returned points into the context's own output buffer
            // and is valid until the next llama_decode on this context, so each
            // row is copied here, before the next window is decoded, and never
            // retained.
            while (k < n_positions && positions[k] < start + count) {
                const int position = positions[k];
                const float* row = llama_get_embeddings_ith(wrapper->ctx, position - start);
                if (!row) {
                    llama_batch_free(batch);
                    g_last_error = "No embeddings row for position " + std::to_string(position);
                    return -1;
                }
                // A non-finite value is a failure, not a result: a NaN target
                // enters the objective and poisons the update without failing
                // anything.
                for (int j = 0; j < n_embd_out; j++) {
                    if (!std::isfinite(row[j])) {
                        llama_batch_free(batch);
                        g_last_error = "Non-finite value at position " + std::to_string(position) +
                                       ", column " + std::to_string(j);
                        return -1;
                    }
                }
                std::memcpy(out + static_cast<size_t>(k) * static_cast<size_t>(n_embd_out), row,
                            static_cast<size_t>(n_embd_out) * sizeof(float));
                k++;
            }
            llama_batch_free(batch);
        }

        if (k != n_positions) {
            g_last_error = "Captured " + std::to_string(k) + " of " +
                           std::to_string(n_positions) + " rows";
            return -1;
        }
        return 0;
    } catch (const std::exception& e) {
        g_last_error = "Exception capturing: " + std::string(e.what());
        return -1;
    }
}

}  // extern "C"

// ---------------------------------------------------------------------------
// The LoRA arithmetic of one projection, observed.
//
// Every control that leaves B at zero measures a contribution of zero, and an
// engine that applied a hundred times the right value would pass all of them.
// So would a set of adapters compared against each other: a uniform factor k
// survives every equality between two routes to the same contribution. The only
// thing that settles it is reading the numbers the graph actually produced and
// comparing them with arithmetic done somewhere else.
//
// The state is a file-scope object rather than user data because the callback is
// a C function pointer with no room for a this, which is how tools/imatrix does
// it too. The mutex is not decoration: the scheduler may reach the callback from
// a backend thread.
// ---------------------------------------------------------------------------
namespace {

struct lora_collector {
    std::mutex mu;
    bool active = false;
    std::string w_name, a_name, b_name;

    // Node identities, remembered as the graph is walked, because the scale and
    // the sum can only be recognised by which nodes they consume.
    const ggml_tensor* base_node = nullptr;
    const ggml_tensor* b_node = nullptr;
    const ggml_tensor* scale_node = nullptr;

    std::vector<float> x, h, u_pre, u, y_base, y;
    bool got_x = false, got_h = false, got_u_pre = false;
    bool got_u = false, got_y_base = false, got_y = false;
    std::string failure;

    void reset() {
        active = false;
        base_node = nullptr;
        b_node = nullptr;
        scale_node = nullptr;
        x.clear(); h.clear(); u_pre.clear(); u.clear(); y_base.clear(); y.clear();
        got_x = got_h = got_u_pre = got_u = got_y_base = got_y = false;
        failure.clear();
    }
};

lora_collector g_lora;

// undecorated strips the backend prefix the scheduler adds to a copied operand.
//
// When a node runs on a backend that does not own its operand's buffer,
// ggml_backend_sched_split_graph makes a copy, names it "<backend>#<name>#<id>"
// and -- the part that matters -- replaces node->src[j] with the copy. So the
// callback sees "CUDA0#blk.35.ffn_down.weight#0" where the GGUF says
// "blk.35.ffn_down.weight", and an exact comparison matches nothing.
//
// tools/imatrix carries the same function for the same reason, with that exact
// example in its comment. Without it the capture fails whenever the projection
// asked for sits on a layer that stayed on the CPU -- which is the expected
// shape of a 27B on cards with about fourteen gigabytes free.
std::string undecorated(const char* raw) {
    std::string name(raw ? raw : "");
    const size_t first = name.find('#');
    if (first == std::string::npos) return name;
    const size_t second = name.find('#', first + 1);
    if (second == std::string::npos) return name.substr(first + 1);
    return name.substr(first + 1, second - first - 1);
}

bool named_is(const ggml_tensor* t, const std::string& want) {
    return t && !want.empty() && want == undecorated(ggml_get_name(t));
}

// read_into copies a tensor out of whatever buffer it lives in.
//
// ggml_backend_tensor_get rather than t->data, and not as a precaution: on Metal
// the pointer is not host readable at all, and on CUDA it addresses device
// memory. imatrix and cvector-generator both read this way.
bool read_into(const ggml_tensor* t, std::vector<float>& into) {
    if (!t) {
        g_lora.failure = "a tensor the capture needs is absent from the graph";
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        g_lora.failure = std::string("tensor ") + ggml_get_name(t) + " is type " +
                         std::to_string(static_cast<int>(t->type)) + ", not F32";
        return false;
    }
    const int64_t n = ggml_nelements(t);
    if (n <= 0) {
        g_lora.failure = std::string("tensor ") + ggml_get_name(t) + " holds no element";
        return false;
    }
    into.resize(static_cast<size_t>(n));
    ggml_backend_tensor_get(t, into.data(), 0, static_cast<size_t>(n) * sizeof(float));
    for (int64_t i = 0; i < n; i++) {
        if (!std::isfinite(into[static_cast<size_t>(i)])) {
            // A non-finite value is a failure, not a result: it would enter the
            // comparison and read as a difference nobody could attribute.
            g_lora.failure = std::string("non-finite value in ") + ggml_get_name(t) +
                             " at index " + std::to_string(i);
            return false;
        }
    }
    return true;
}

// wanted decides, from the operation and the name of the weight operand, whether
// this node is one of the six. It has to give the same answer on the ask call
// and on the call that follows it, which is why it reads state that only the
// collecting call writes: a node is recognised after the node it consumes.
bool wanted(const ggml_tensor* t) {
    if (t->op == GGML_OP_MUL_MAT) {
        return named_is(t->src[0], g_lora.w_name) ||
               named_is(t->src[0], g_lora.a_name) ||
               named_is(t->src[0], g_lora.b_name);
    }
    if (t->op == GGML_OP_SCALE) {
        return g_lora.b_node != nullptr && t->src[0] == g_lora.b_node;
    }
    if (t->op == GGML_OP_ADD) {
        // Both operands, not just the scaled contribution. build_lora_mm does
        //
        //   res = W*x ; if (w_s) res = res * w_s ; res = res + ab
        //
        // so with a per-tensor scale present the sum's left operand is a
        // GGML_OP_MUL, not the matmul that was captured as y_base. Matching on
        // src[1] alone would fill all six buffers, return success, and hand back
        // a set where y != y_base + u by a uniform factor -- which is exactly
        // the kind of defect this whole capture exists to expose. Requiring both
        // makes that case fail loudly instead.
        return g_lora.scale_node != nullptr && g_lora.base_node != nullptr &&
               t->src[1] == g_lora.scale_node && t->src[0] == g_lora.base_node;
    }
    return false;
}

bool collect(ggml_tensor* t) {
    if (t->op == GGML_OP_MUL_MAT) {
        if (named_is(t->src[0], g_lora.w_name)) {
            // The base projection. Its right operand is the input the whole
            // chain is a function of, and it is still live: it is a source of
            // the node the callback is holding.
            if (!read_into(t, g_lora.y_base)) return false;
            if (!read_into(t->src[1], g_lora.x)) return false;
            g_lora.got_y_base = g_lora.got_x = true;
            // Remembered so the sum can be required to consume this very node,
            // and not a per-tensor rescaling of it.
            g_lora.base_node = t;
            return true;
        }
        if (named_is(t->src[0], g_lora.a_name)) {
            if (!read_into(t, g_lora.h)) return false;
            g_lora.got_h = true;
            return true;
        }
        if (named_is(t->src[0], g_lora.b_name)) {
            if (!read_into(t, g_lora.u_pre)) return false;
            g_lora.got_u_pre = true;
            // Remembered so the scale that consumes it can be recognised: the
            // scale node carries no name of its own.
            g_lora.b_node = t;
            return true;
        }
        return true;
    }
    if (t->op == GGML_OP_SCALE) {
        if (!read_into(t, g_lora.u)) return false;
        g_lora.got_u = true;
        g_lora.scale_node = t;
        return true;
    }
    if (t->op == GGML_OP_ADD) {
        if (!read_into(t, g_lora.y)) return false;
        g_lora.got_y = true;
        return true;
    }
    return true;
}

bool lora_eval_callback(ggml_tensor* t, bool ask, void* user_data) {
    (void)user_data;
    std::lock_guard<std::mutex> lock(g_lora.mu);
    if (!g_lora.active) return false;
    if (!g_lora.failure.empty()) return false;
    if (ask) return wanted(t);
    if (!wanted(t)) return true;
    // Returning false here stops the rest of the split, which is what a failure
    // should do: carrying on would fill some buffers and not others.
    return collect(t);
}

}  // namespace

extern "C" {

int llama_wrapper_capture_lora(void* model, const char* module,
                               void* adapter, float adapter_scale,
                               const int* tokens, int n_tokens, int n_ctx,
                               llama_wrapper_lora_capture* out) {
    if (!model || !module || !tokens || !out) {
        g_last_error = "Model, module, tokens and an output struct are required";
        return -1;
    }
    if (n_tokens < 1) {
        g_last_error = "Capturing needs at least one token";
        return -1;
    }
    if (n_ctx < n_tokens) {
        g_last_error = "The context of " + std::to_string(n_ctx) + " is shorter than the " +
                       std::to_string(n_tokens) + " tokens to capture";
        return -1;
    }
    if (!adapter && (out->h || out->u_pre || out->u || out->y)) {
        g_last_error = "Without an adapter only x and y_base are captured; the other buffers must be null";
        return -1;
    }

    llama_context* ctx = nullptr;
    try {
        auto wrapper = static_cast<llama_wrapper_model_t*>(model);
        const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
        const int n_vocab = static_cast<int>(llama_vocab_n_tokens(vocab));
        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i] < 0 || tokens[i] >= n_vocab) {
                g_last_error = "Token " + std::to_string(tokens[i]) + " at position " +
                               std::to_string(i) + " is outside the vocabulary";
                return -1;
            }
        }

        {
            std::lock_guard<std::mutex> lock(g_lora.mu);
            if (g_lora.active) {
                // One collector, one capture. Two at once would interleave their
                // nodes and fill each other's buffers.
                g_last_error = "Another LoRA capture is in progress on this process";
                return -1;
            }
            g_lora.reset();
            g_lora.w_name = module;
            if (adapter) {
                g_lora.a_name = std::string(module) + ".lora_a";
                g_lora.b_name = std::string(module) + ".lora_b";
            }
            g_lora.active = true;
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = static_cast<uint32_t>(n_ctx);
        cparams.n_batch = static_cast<uint32_t>(n_ctx);
        cparams.n_ubatch = static_cast<uint32_t>(n_ctx);
        // The whole reason this function owns a context: cb_eval is reachable
        // only through llama_context_params, and llama.h has no setter for it
        // afterwards. The context is destroyed before this returns, so the
        // per-split synchronisation the callback forces lasts one decode.
        cparams.cb_eval = lora_eval_callback;
        cparams.cb_eval_user_data = nullptr;

        ctx = llama_init_from_model(wrapper->model, cparams);
        if (!ctx) {
            std::lock_guard<std::mutex> lock(g_lora.mu);
            g_lora.active = false;
            g_last_error = "Failed to create the diagnostic context";
            return -1;
        }

        if (adapter) {
            llama_adapter_lora* lora = static_cast<llama_adapter_lora*>(adapter);
            if (llama_set_adapters_lora(ctx, &lora, 1, &adapter_scale) != 0) {
                llama_free(ctx);
                std::lock_guard<std::mutex> lock(g_lora.mu);
                g_lora.active = false;
                g_last_error = "Failed to apply the adapter to the diagnostic context";
                return -1;
            }
        }

        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            // Every row an output row, the same rule llama_wrapper_capture_final
            // follows, and here it is not a nicety. When fewer rows are flagged
            // than the batch holds, build_inp_out_ids emits a get_rows of that
            // length and every builder applies it at the last layer, BEFORE the
            // final FFN. So flagging only the last token would narrow the last
            // block's ffn_norm and its three projections to a single column --
            // and blk.35.ffn_down.weight, the module this function documents as
            // its example, is in that block.
            batch.logits[i] = 1;
        }
        batch.n_tokens = n_tokens;
        const int32_t decoded = llama_decode(ctx, batch);
        llama_batch_free(batch);

        std::string failure;
        bool ok = false;
        {
            std::lock_guard<std::mutex> lock(g_lora.mu);
            g_lora.active = false;
            failure = g_lora.failure;
            ok = failure.empty() && decoded == 0;
        }
        if (decoded != 0 && failure.empty()) {
            failure = "llama_decode failed with " + std::to_string(decoded);
        }
        if (!ok) {
            llama_free(ctx);
            g_last_error = failure;
            return -1;
        }

        // Copied out under the lock, and the lengths are checked exactly: a
        // buffer sized for another width is a buffer for another capture.
        std::lock_guard<std::mutex> lock(g_lora.mu);
        struct slot {
            const char* name;
            bool got;
            const std::vector<float>* from;
            float* into;
            long long floats;
        };
        const slot slots[] = {
            {"x", g_lora.got_x, &g_lora.x, out->x, out->x_floats},
            {"y_base", g_lora.got_y_base, &g_lora.y_base, out->y_base, out->y_base_floats},
            {"h", g_lora.got_h, &g_lora.h, out->h, out->h_floats},
            {"u_pre", g_lora.got_u_pre, &g_lora.u_pre, out->u_pre, out->u_pre_floats},
            {"u", g_lora.got_u, &g_lora.u, out->u, out->u_floats},
            {"y", g_lora.got_y, &g_lora.y, out->y, out->y_floats},
        };
        for (const slot& s : slots) {
            if (!s.into) continue;
            if (!s.got) {
                llama_free(ctx);
                g_last_error = std::string("the graph produced no ") + s.name + " for " + module;
                return -1;
            }
            if (s.floats != static_cast<long long>(s.from->size())) {
                llama_free(ctx);
                g_last_error = std::string("buffer for ") + s.name + " holds " +
                               std::to_string(s.floats) + " floats; the capture produced " +
                               std::to_string(s.from->size());
                return -1;
            }
            std::memcpy(s.into, s.from->data(), s.from->size() * sizeof(float));
        }
        llama_free(ctx);
        return 0;
    } catch (const std::exception& e) {
        if (ctx) llama_free(ctx);
        std::lock_guard<std::mutex> lock(g_lora.mu);
        g_lora.active = false;
        g_last_error = "Exception capturing the LoRA arithmetic: " + std::string(e.what());
        return -1;
    }
}

}  // extern "C"
