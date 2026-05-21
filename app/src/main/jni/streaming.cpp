#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <functional>
#include <thread>

#include "llama.h"
#include "common.h"

#define TAG "AIOPE-INF"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ============================================================
// Streaming callback state
// ============================================================
struct StreamState {
    JNIEnv *env;
    jobject callback;
    jmethodID onToken;
    jmethodID onComplete;
    jmethodID onError;
    std::atomic<bool> *abort;
};

extern llama_model *g_model;
extern llama_context *g_ctx;
extern std::mutex g_mutex;
extern std::atomic<bool> g_abort;

// ============================================================
// JNI: Streaming generation
// ============================================================
extern "C" JNIEXPORT void JNICALL
Java_com_aiope_inf_LlamaJNI_generateStreaming(
    JNIEnv *env, jobject /* this */,
    jstring prompt,
    jint maxTokens,
    jfloat temperature,
    jfloat topP,
    jfloat repeatPenalty,
    jobject callback) {

    std::lock_guard<std::mutex> lock(g_mutex);
    g_abort.store(false);

    if (!g_model || !g_ctx) {
        jclass cls = env->GetObjectClass(callback);
        jmethodID onError = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");
        env->CallVoidMethod(callback, onError, env->NewStringUTF("Model not loaded"));
        return;
    }

    // Get callback methods
    jclass cls = env->GetObjectClass(callback);
    jmethodID onToken = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)Z");
    jmethodID onComplete = env->GetMethodID(cls, "onComplete", "(Ljava/lang/String;)V");
    jmethodID onError = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");

    std::string prompt_str;
    const char *chars = env->GetStringUTFChars(prompt, nullptr);
    prompt_str = chars;
    env->ReleaseStringUTFChars(prompt, chars);

    const llama_vocab *vocab = llama_model_get_vocab(g_model);

    // Tokenize
    std::vector<llama_token> tokens(prompt_str.size() + 128);
    int n_tokens = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(),
                                  tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(),
                                  tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    // Clear KV cache
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Decode prompt
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, {0}, (i == n_tokens - 1));
    }

    if (llama_decode(g_ctx, batch) != 0) {
        llama_batch_free(batch);
        env->CallVoidMethod(callback, onError, env->NewStringUTF("Prompt decode failed"));
        return;
    }
    llama_batch_free(batch);

    // Sampling setup
    auto *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature > 0 ? temperature : 0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP > 0 ? topP : 0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        64, repeatPenalty > 0 ? repeatPenalty : 1.1f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

    // Stream tokens
    std::string full_response;
    int n_cur = n_tokens;
    int n_max = maxTokens > 0 ? maxTokens : 512;

    for (int i = 0; i < n_max; i++) {
        if (g_abort.load()) {
            LOGI("Streaming aborted by user");
            break;
        }

        llama_token new_token = llama_sampler_sample(smpl, g_ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) break;

        // Token to text
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            full_response += piece;

            // Callback: onToken returns false to stop
            jstring jpiece = env->NewStringUTF(piece.c_str());
            jboolean cont = env->CallBooleanMethod(callback, onToken, jpiece);
            env->DeleteLocalRef(jpiece);

            if (!cont) {
                LOGI("Streaming stopped by callback");
                break;
            }
        }

        // Next token
        llama_batch next_batch = llama_batch_init(1, 0, 1);
        common_batch_add(next_batch, new_token, n_cur, {0}, true);
        n_cur++;

        if (llama_decode(g_ctx, next_batch) != 0) {
            llama_batch_free(next_batch);
            env->CallVoidMethod(callback, onError, env->NewStringUTF("Decode failed during streaming"));
            llama_sampler_free(smpl);
            return;
        }
        llama_batch_free(next_batch);
    }

    llama_sampler_free(smpl);

    // Signal completion
    jstring jfull = env->NewStringUTF(full_response.c_str());
    env->CallVoidMethod(callback, onComplete, jfull);
    env->DeleteLocalRef(jfull);
}

// ============================================================
// JNI: SSE-compatible streaming for OpenAI API
// Writes Server-Sent Events format to a pipe/buffer
// ============================================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_aiope_inf_LlamaJNI_generateSSE(
    JNIEnv *env, jobject /* this */,
    jstring prompt,
    jstring model_name,
    jint maxTokens,
    jfloat temperature,
    jfloat topP,
    jobject tokenCallback) {

    std::lock_guard<std::mutex> lock(g_mutex);
    g_abort.store(false);

    if (!g_model || !g_ctx) {
        return env->NewStringUTF("{\"error\":\"model not loaded\"}");
    }

    std::string prompt_str;
    const char *chars = env->GetStringUTFChars(prompt, nullptr);
    prompt_str = chars;
    env->ReleaseStringUTFChars(prompt, chars);

    std::string model_str;
    const char *mchars = env->GetStringUTFChars(model_name, nullptr);
    model_str = mchars;
    env->ReleaseStringUTFChars(model_name, mchars);

    const llama_vocab *vocab = llama_model_get_vocab(g_model);

    // Tokenize
    std::vector<llama_token> tokens(prompt_str.size() + 128);
    int n_tokens = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(),
                                  tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(),
                                  tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Decode prompt
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, {0}, (i == n_tokens - 1));
    }
    if (llama_decode(g_ctx, batch) != 0) {
        llama_batch_free(batch);
        return env->NewStringUTF("{\"error\":\"prompt decode failed\"}");
    }
    llama_batch_free(batch);

    // Sampling
    auto *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature > 0 ? temperature : 0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP > 0 ? topP : 0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

    // Get callback method if provided
    jclass cbCls = nullptr;
    jmethodID cbMethod = nullptr;
    if (tokenCallback) {
        cbCls = env->GetObjectClass(tokenCallback);
        cbMethod = env->GetMethodID(cbCls, "onToken", "(Ljava/lang/String;)Z");
    }

    std::string sse_output;
    std::string completion_id = "chatcmpl-aiope-" + std::to_string(time(nullptr));
    int n_cur = n_tokens;
    int n_max = maxTokens > 0 ? maxTokens : 512;

    for (int i = 0; i < n_max; i++) {
        if (g_abort.load()) break;

        llama_token new_token = llama_sampler_sample(smpl, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);

            // SSE chunk format
            std::string chunk = "data: {\"id\":\"" + completion_id + "\","
                "\"object\":\"chat.completion.chunk\","
                "\"created\":" + std::to_string(time(nullptr)) + ","
                "\"model\":\"" + model_str + "\","
                "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"";

            // Escape JSON special chars
            for (char c : piece) {
                switch (c) {
                    case '"': chunk += "\\\""; break;
                    case '\\': chunk += "\\\\"; break;
                    case '\n': chunk += "\\n"; break;
                    case '\r': chunk += "\\r"; break;
                    case '\t': chunk += "\\t"; break;
                    default: chunk += c;
                }
            }
            chunk += "\"},\"finish_reason\":null}]}\n\n";
            sse_output += chunk;

            // Stream callback
            if (cbMethod) {
                jstring jpiece = env->NewStringUTF(piece.c_str());
                jboolean cont = env->CallBooleanMethod(tokenCallback, cbMethod, jpiece);
                env->DeleteLocalRef(jpiece);
                if (!cont) break;
            }
        }

        // Next token
        llama_batch next_batch = llama_batch_init(1, 0, 1);
        common_batch_add(next_batch, new_token, n_cur, {0}, true);
        n_cur++;
        if (llama_decode(g_ctx, next_batch) != 0) {
            llama_batch_free(next_batch);
            break;
        }
        llama_batch_free(next_batch);
    }

    // Final SSE message
    sse_output += "data: {\"id\":\"" + completion_id + "\","
        "\"object\":\"chat.completion.chunk\","
        "\"created\":" + std::to_string(time(nullptr)) + ","
        "\"model\":\"" + model_str + "\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
    sse_output += "data: [DONE]\n\n";

    llama_sampler_free(smpl);
    return env->NewStringUTF(sse_output.c_str());
}
