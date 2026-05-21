#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>

#include "llama.h"
#include "common.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define TAG "AIOPE-MTMD"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern llama_model *g_model;
extern llama_context *g_ctx;
extern std::mutex g_mutex;
extern std::atomic<bool> g_abort;

static mtmd_context *g_mtmd = nullptr;

// ============================================================
// JNI: Initialize multimodal context
// ============================================================
extern "C" JNIEXPORT jboolean JNICALL
Java_com_aiope_inf_LlamaJNI_initMultimodal(
    JNIEnv *env, jobject /* this */,
    jstring mmProjPath) {

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model) {
        LOGE("Cannot init multimodal: model not loaded");
        return JNI_FALSE;
    }

    if (g_mtmd) { mtmd_free(g_mtmd); g_mtmd = nullptr; }

    const char *path = env->GetStringUTFChars(mmProjPath, nullptr);
    LOGI("Initializing multimodal: %s", path);

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = true;
    params.n_threads = std::thread::hardware_concurrency();

    g_mtmd = mtmd_init_from_file(path, g_model, params);
    env->ReleaseStringUTFChars(mmProjPath, path);

    if (!g_mtmd) {
        LOGE("Failed to init multimodal context");
        return JNI_FALSE;
    }

    LOGI("Multimodal initialized (vision=%d, audio=%d)",
         mtmd_support_vision(g_mtmd), mtmd_support_audio(g_mtmd));
    return JNI_TRUE;
}

// ============================================================
// JNI: Generate with image input
// ============================================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_aiope_inf_LlamaJNI_generateWithImage(
    JNIEnv *env, jobject /* this */,
    jstring prompt,
    jbyteArray imageData,
    jint width,
    jint height,
    jint maxTokens,
    jfloat temperature,
    jfloat topP) {

    std::lock_guard<std::mutex> lock(g_mutex);
    g_abort.store(false);

    if (!g_model || !g_ctx || !g_mtmd) {
        LOGE("Model or multimodal context not loaded");
        return env->NewStringUTF("");
    }

    // Get image bytes (RGB)
    jsize img_len = env->GetArrayLength(imageData);
    jbyte *img_bytes = env->GetByteArrayElements(imageData, nullptr);

    // Create bitmap from raw RGB data
    mtmd_bitmap *bitmap = mtmd_bitmap_init(width, height, (const unsigned char *)img_bytes);
    env->ReleaseByteArrayElements(imageData, img_bytes, JNI_ABORT);

    if (!bitmap) {
        LOGE("Failed to create bitmap");
        return env->NewStringUTF("");
    }

    // Build prompt with image marker
    std::string prompt_str;
    const char *chars = env->GetStringUTFChars(prompt, nullptr);
    prompt_str = chars;
    env->ReleaseStringUTFChars(prompt, chars);

    const char *marker = mtmd_default_marker();
    std::string full_prompt;
    if (prompt_str.find(marker) == std::string::npos) {
        full_prompt = std::string(marker) + "\n" + prompt_str;
    } else {
        full_prompt = prompt_str;
    }

    // Tokenize with multimodal chunks
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    mtmd_input_text text_input = { full_prompt.c_str(), true, true };
    const mtmd_bitmap *bitmaps[] = { bitmap };

    int32_t tokenize_result = mtmd_tokenize(g_mtmd, chunks, &text_input, bitmaps, 1);
    if (tokenize_result != 0) {
        LOGE("Multimodal tokenization failed: %d", tokenize_result);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        return env->NewStringUTF("");
    }

    // Eval all chunks using helper
    llama_memory_clear(llama_get_memory(g_ctx), true);
    llama_pos n_past = 0;
    int32_t eval_result = mtmd_helper_eval_chunks(g_mtmd, g_ctx, chunks, 0, 0,
                                                   512, true, &n_past);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);

    if (eval_result != 0) {
        LOGE("Multimodal eval failed: %d", eval_result);
        return env->NewStringUTF("");
    }

    // Generate response
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    auto *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature > 0 ? temperature : 0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP > 0 ? topP : 0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

    std::string result;
    int n_max = maxTokens > 0 ? maxTokens : 512;

    for (int i = 0; i < n_max; i++) {
        if (g_abort.load()) break;

        llama_token new_token = llama_sampler_sample(smpl, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) result.append(buf, n);

        llama_batch next_batch = llama_batch_init(1, 0, 1);
        common_batch_add(next_batch, new_token, n_past++, {0}, true);
        if (llama_decode(g_ctx, next_batch) != 0) {
            llama_batch_free(next_batch);
            break;
        }
        llama_batch_free(next_batch);
    }

    llama_sampler_free(smpl);
    return env->NewStringUTF(result.c_str());
}

// ============================================================
// JNI: Free multimodal context
// ============================================================
extern "C" JNIEXPORT void JNICALL
Java_com_aiope_inf_LlamaJNI_freeMultimodal(JNIEnv *env, jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_mtmd) { mtmd_free(g_mtmd); g_mtmd = nullptr; }
    LOGI("Multimodal context freed");
}
