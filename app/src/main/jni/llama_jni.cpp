#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <functional>

#include "llama.h"
#include "common.h"

#define TAG "AIOPE-INF"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ============================================================
// Global state
// ============================================================
static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static std::mutex g_mutex;
static std::atomic<bool> g_abort{false};

// GPU info
static bool g_vulkan_available = false;
static int g_gpu_layers = 0;

// ============================================================
// Helpers
// ============================================================
static std::string jstring_to_string(JNIEnv *env, jstring jstr) {
    if (!jstr) return "";
    const char *chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

static jstring string_to_jstring(JNIEnv *env, const std::string &str) {
    return env->NewStringUTF(str.c_str());
}

// ============================================================
// JNI: Model Loading
// ============================================================
extern "C" JNIEXPORT jboolean JNICALL
Java_com_aiope_inf_LlamaJNI_loadModel(
    JNIEnv *env, jobject /* this */,
    jstring modelPath,
    jint nGpuLayers,
    jint contextSize,
    jint batchSize,
    jboolean useVulkan) {

    std::lock_guard<std::mutex> lock(g_mutex);

    // Unload existing model
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }

    std::string path = jstring_to_string(env, modelPath);
    LOGI("Loading model: %s (gpu_layers=%d, ctx=%d, batch=%d, vulkan=%d)",
         path.c_str(), nGpuLayers, contextSize, batchSize, useVulkan);

    // Model params
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = nGpuLayers;

#ifdef AIOPE_VULKAN
    if (useVulkan) {
        LOGI("Vulkan GPU acceleration enabled");
        g_vulkan_available = true;
    }
#endif

    g_gpu_layers = nGpuLayers;

    // Load model
    g_model = llama_model_load_from_file(path.c_str(), model_params);
    if (!g_model) {
        LOGE("Failed to load model: %s", path.c_str());
        return JNI_FALSE;
    }

    // Context params
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = contextSize > 0 ? contextSize : 2048;
    ctx_params.n_batch = batchSize > 0 ? batchSize : 512;
    ctx_params.n_threads = std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = std::thread::hardware_concurrency();

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        LOGE("Failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    LOGI("Model loaded successfully. Threads: %d", ctx_params.n_threads);
    return JNI_TRUE;
}

// ============================================================
// JNI: Unload Model
// ============================================================
extern "C" JNIEXPORT void JNICALL
Java_com_aiope_inf_LlamaJNI_unloadModel(JNIEnv *env, jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    LOGI("Model unloaded");
}

// ============================================================
// JNI: Generate (blocking, full response)
// ============================================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_aiope_inf_LlamaJNI_generate(
    JNIEnv *env, jobject /* this */,
    jstring prompt,
    jint maxTokens,
    jfloat temperature,
    jfloat topP,
    jfloat repeatPenalty) {

    std::lock_guard<std::mutex> lock(g_mutex);
    g_abort.store(false);

    if (!g_model || !g_ctx) {
        LOGE("Model not loaded");
        return string_to_jstring(env, "");
    }

    std::string prompt_str = jstring_to_string(env, prompt);
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
    llama_kv_cache_clear(g_ctx);

    // Decode prompt
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, {0}, (i == n_tokens - 1));
    }

    if (llama_decode(g_ctx, batch) != 0) {
        LOGE("Failed to decode prompt");
        llama_batch_free(batch);
        return string_to_jstring(env, "");
    }
    llama_batch_free(batch);

    // Sampling
    std::string result;
    int n_cur = n_tokens;
    int n_max = maxTokens > 0 ? maxTokens : 512;

    auto *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature > 0 ? temperature : 0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP > 0 ? topP : 0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        64, repeatPenalty > 0 ? repeatPenalty : 1.1f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

    for (int i = 0; i < n_max; i++) {
        if (g_abort.load()) break;

        llama_token new_token = llama_sampler_sample(smpl, g_ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) break;

        // Convert token to text
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.append(buf, n);
        }

        // Prepare next batch
        llama_batch next_batch = llama_batch_init(1, 0, 1);
        llama_batch_add(next_batch, new_token, n_cur, {0}, true);
        n_cur++;

        if (llama_decode(g_ctx, next_batch) != 0) {
            LOGE("Decode failed at token %d", i);
            llama_batch_free(next_batch);
            break;
        }
        llama_batch_free(next_batch);
    }

    llama_sampler_free(smpl);
    return string_to_jstring(env, result);
}

// ============================================================
// JNI: Abort generation
// ============================================================
extern "C" JNIEXPORT void JNICALL
Java_com_aiope_inf_LlamaJNI_abort(JNIEnv *env, jobject /* this */) {
    g_abort.store(true);
    LOGI("Generation aborted");
}

// ============================================================
// JNI: Model info
// ============================================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_aiope_inf_LlamaJNI_getModelInfo(JNIEnv *env, jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model) return string_to_jstring(env, "{}");

    char desc[256];
    llama_model_desc(g_model, desc, sizeof(desc));

    std::string info = "{";
    info += "\"description\":\"" + std::string(desc) + "\",";
    info += "\"n_params\":" + std::to_string(llama_model_n_params(g_model)) + ",";
    info += "\"size\":" + std::to_string(llama_model_size(g_model)) + ",";
    info += "\"gpu_layers\":" + std::to_string(g_gpu_layers) + ",";
    info += "\"vulkan\":" + std::string(g_vulkan_available ? "true" : "false");
    info += "}";

    return string_to_jstring(env, info);
}

// ============================================================
// JNI: Check Vulkan support
// ============================================================
extern "C" JNIEXPORT jboolean JNICALL
Java_com_aiope_inf_LlamaJNI_isVulkanAvailable(JNIEnv *env, jobject /* this */) {
#ifdef AIOPE_VULKAN
    return JNI_TRUE;
#else
    return JNI_FALSE;
#endif
}

// ============================================================
// JNI: Backend info
// ============================================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_aiope_inf_LlamaJNI_getBackendInfo(JNIEnv *env, jobject /* this */) {
    std::string info = "{";
    info += "\"cpu_threads\":" + std::to_string(std::thread::hardware_concurrency()) + ",";
#ifdef AIOPE_VULKAN
    info += "\"vulkan\":true,";
#else
    info += "\"vulkan\":false,";
#endif
    info += "\"neon\":true";  // ARM64-v8a always has NEON
    info += "}";
    return string_to_jstring(env, info);
}
