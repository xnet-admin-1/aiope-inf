#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <functional>
#include <algorithm>
#include <cstdio>

#include "llama.h"
#include "common.h"
#include "chat.h"
#include "nlohmann/json.hpp"

#define TAG "AIOPE-INF"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ============================================================
// Global state
// ============================================================
llama_model *g_model = nullptr;
llama_context *g_ctx = nullptr;
llama_adapter_lora *g_lora = nullptr;
std::mutex g_mutex;
std::atomic<bool> g_abort{false};

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
    model_params.use_mmap = true;  // Memory-map to avoid loading entire model into RAM

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
    ctx_params.n_ctx = contextSize > 0 ? contextSize : 4096;
    ctx_params.n_batch = batchSize > 0 ? batchSize : 512;
    ctx_params.n_ubatch = 256; // smaller micro-batches reduce memory pressure

    // big.LITTLE-aware thread scheduling
    // Read CPU topology to find performance cores
    int n_cores = std::thread::hardware_concurrency();
    int perf_cores = 0;
    long max_freq = 0;
    for (int i = 0; i < n_cores; i++) {
        char path_buf[128];
        snprintf(path_buf, sizeof(path_buf),
            "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE *f = fopen(path_buf, "r");
        if (f) {
            long freq = 0;
            fscanf(f, "%ld", &freq);
            fclose(f);
            if (freq > max_freq) max_freq = freq;
        }
    }
    // Count cores within 80% of max frequency as "performance" cores
    for (int i = 0; i < n_cores; i++) {
        char path_buf[128];
        snprintf(path_buf, sizeof(path_buf),
            "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE *f = fopen(path_buf, "r");
        if (f) {
            long freq = 0;
            fscanf(f, "%ld", &freq);
            fclose(f);
            if (freq >= max_freq * 80 / 100) perf_cores++;
        }
    }
    if (perf_cores == 0) perf_cores = std::max(1, n_cores / 2);

    // Check thermal state — throttle if hot
    int thermal_temp = 0;
    FILE *tf = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
    if (tf) { fscanf(tf, "%d", &thermal_temp); fclose(tf); }
    // thermal_temp is in millidegrees on most devices
    if (thermal_temp > 1000) thermal_temp /= 1000; // normalize to celsius

    int gen_threads = perf_cores;
    if (thermal_temp > 75) gen_threads = std::max(2, perf_cores / 2); // HOT: halve
    if (thermal_temp > 85) gen_threads = 2; // CRITICAL: minimum

    ctx_params.n_threads = gen_threads;
    ctx_params.n_threads_batch = n_cores; // batch uses ALL cores for fast prompt eval

    LOGI("Creating context: ctx=%d, batch=%d, threads=%d (perf_cores=%d, thermal=%d°C)",
         ctx_params.n_ctx, ctx_params.n_batch, ctx_params.n_threads, perf_cores, thermal_temp);

    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

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
    if (g_lora) { llama_adapter_lora_free(g_lora); g_lora = nullptr; }
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
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Decode prompt in batches to avoid memory spike
    int batch_size = 512;
    for (int i = 0; i < n_tokens; i += batch_size) {
        if (g_abort.load()) {
            return string_to_jstring(env, "");
        }
        int n_batch = std::min(batch_size, n_tokens - i);
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        for (int j = 0; j < n_batch; j++) {
            common_batch_add(batch, tokens[i + j], i + j, {0}, (i + j == n_tokens - 1));
        }
        if (llama_decode(g_ctx, batch) != 0) {
            LOGE("Failed to decode prompt at pos %d", i);
            llama_batch_free(batch);
            return string_to_jstring(env, "");
        }
        llama_batch_free(batch);
    }

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
        llama_sampler_accept(smpl, new_token);

        // Guard: if first token is EOS, skip it (model confused by prompt)
        if (i == 0 && llama_vocab_is_eog(vocab, new_token)) continue;

        if (llama_vocab_is_eog(vocab, new_token)) break;

        // Convert token to text
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.append(buf, n);
        }

        // Prepare next batch
        llama_batch next_batch = llama_batch_init(1, 0, 1);
        common_batch_add(next_batch, new_token, n_cur, {0}, true);
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

// ============================================================
// JNI: Load LoRA adapter
// ============================================================
extern "C" JNIEXPORT jboolean JNICALL
Java_com_aiope_inf_LlamaJNI_loadLoraAdapter(
    JNIEnv *env, jobject /* this */,
    jstring loraPath,
    jfloat scale) {

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx) {
        LOGE("Cannot load LoRA: model not loaded");
        return JNI_FALSE;
    }

    // Free existing adapter
    if (g_lora) {
        llama_adapter_lora_free(g_lora);
        g_lora = nullptr;
    }

    std::string path = jstring_to_string(env, loraPath);
    LOGI("Loading LoRA adapter: %s (scale=%.2f)", path.c_str(), scale);

    g_lora = llama_adapter_lora_init(g_model, path.c_str());
    if (!g_lora) {
        LOGE("Failed to load LoRA adapter: %s", path.c_str());
        return JNI_FALSE;
    }

    // Apply adapter to context
    float scales[] = { scale };
    llama_adapter_lora *adapters[] = { g_lora };
    int32_t result = llama_set_adapters_lora(g_ctx, adapters, 1, scales);
    if (result != 0) {
        LOGE("Failed to set LoRA adapter");
        llama_adapter_lora_free(g_lora);
        g_lora = nullptr;
        return JNI_FALSE;
    }

    LOGI("LoRA adapter loaded successfully");
    return JNI_TRUE;
}

// ============================================================
// JNI: Unload LoRA adapter
// ============================================================
extern "C" JNIEXPORT void JNICALL
Java_com_aiope_inf_LlamaJNI_unloadLoraAdapter(JNIEnv *env, jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_lora) {
        llama_set_adapters_lora(g_ctx, nullptr, 0, nullptr);
        llama_adapter_lora_free(g_lora);
        g_lora = nullptr;
        LOGI("LoRA adapter unloaded");
    }
}

// ============================================================
// JNI: Apply chat template from model metadata
// ============================================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_aiope_inf_LlamaJNI_applyChatTemplate(
    JNIEnv *env, jobject /* this */,
    jstring messagesJson, jstring toolsJson) {

    // No mutex needed — only reads g_model which is stable during generation
    if (!g_model) return string_to_jstring(env, "");

    std::string msgs_str = jstring_to_string(env, messagesJson);
    std::string tools_str = toolsJson ? jstring_to_string(env, toolsJson) : "";

    try {
        // Initialize chat templates from model
        auto tmpls = common_chat_templates_init(g_model, "");
        if (!tmpls) return string_to_jstring(env, "");

        // Parse messages JSON into common_chat_msg vector
        auto msgs_json = nlohmann::ordered_json::parse(msgs_str);
        std::vector<common_chat_msg> messages;
        for (auto &m : msgs_json) {
            common_chat_msg msg;
            msg.role = m.value("role", "");
            if (m.contains("content") && m["content"].is_string()) {
                msg.content = m["content"].get<std::string>();
            } else if (m.contains("content") && m["content"].is_array()) {
                // Multimodal content parts
                for (auto &p : m["content"]) {
                    common_chat_msg_content_part part;
                    part.type = p.value("type", "text");
                    if (part.type == "text") {
                        part.text = p.value("text", "");
                    }
                    msg.content_parts.push_back(part);
                }
            }
            // Tool calls from assistant
            if (m.contains("tool_calls")) {
                for (auto &tc : m["tool_calls"]) {
                    common_chat_tool_call call;
                    if (tc.contains("function")) {
                        call.name = tc["function"].value("name", "");
                        call.arguments = tc["function"].value("arguments", "");
                        if (tc["function"].contains("arguments") && tc["function"]["arguments"].is_object()) {
                            call.arguments = tc["function"]["arguments"].dump();
                        }
                    }
                    if (tc.contains("id")) call.id = tc["id"].get<std::string>();
                    msg.tool_calls.push_back(call);
                }
            }
            // Tool response
            if (m.contains("tool_call_id")) {
                msg.tool_call_id = m["tool_call_id"].get<std::string>();
            }
            if (msg.role == "tool" && m.contains("name")) {
                msg.tool_name = m["name"].get<std::string>();
            }
            // Tool responses (Gemma 4 format)
            if (m.contains("tool_responses")) {
                // Encode tool responses into content for the template
                for (auto &tr : m["tool_responses"]) {
                    common_chat_tool_call tc;
                    tc.name = tr.value("name", "");
                    tc.arguments = tr.contains("response") ? tr["response"].dump() : "";
                    // Store as tool_call with response data for template rendering
                }
            }
            messages.push_back(msg);
        }

        // Parse tools JSON
        std::vector<common_chat_tool> tools;
        if (!tools_str.empty()) {
            auto tools_json = nlohmann::ordered_json::parse(tools_str);
            tools = common_chat_tools_parse_oaicompat(tools_json);
        }

        // Build inputs
        common_chat_templates_inputs inputs;
        inputs.messages = messages;
        inputs.tools = tools;
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        inputs.enable_thinking = false;

        // Apply template
        auto result = common_chat_templates_apply(tmpls.get(), inputs);
        return string_to_jstring(env, result.prompt);
    } catch (const std::exception &e) {
        LOGE("applyChatTemplate error: %s", e.what());
        return string_to_jstring(env, "");
    }
}
