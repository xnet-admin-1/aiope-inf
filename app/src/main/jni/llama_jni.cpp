#include <jni.h>
#include <android/log.h>
#include <string>
#include "llama.h"
#include "common.h"

#define LOG_TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global model and context pointers
static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static gpt_params g_params;

// Convert jstring to std::string
static std::string jstring_to_string(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string str(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return str;
}

extern "C" {

JNIEXPORT jint JNICALL
Java_com_example_llama_LlamaJNI_initialize(
        JNIEnv* env, jobject /* this */, jstring model_path) {
    
    LOGI("Initializing llama.cpp JNI");
    
    std::string model_path_str = jstring_to_string(env, model_path);
    LOGI("Loading model from: %s", model_path_str.c_str());
    
    // Initialize parameters
    g_params.model = model_path_str;
    g_params.n_ctx = 2048;
    g_params.n_threads = 4;
    g_params.n_threads_batch = 4;
    g_params.n_predict = 512;
    g_params.temp = 0.7f;
    g_params.top_k = 40;
    g_params.top_p = 0.95f;
    
    // Initialize llama backend
    llama_backend_init();
    llama_numa_init(g_params.numa);
    
    // Load the model
    LOGI("Loading model...");
    g_model = llama_load_model_from_file(g_params.model.c_str(), g_params.model_params);
    if (g_model == nullptr) {
        LOGE("Failed to load model from %s", g_params.model.c_str());
        return -1;
    }
    
    LOGI("Model loaded successfully");
    
    // Create context
    g_ctx = llama_new_context_with_model(g_model, g_params.ctx_params);
    if (g_ctx == nullptr) {
        LOGE("Failed to create context");
        llama_free_model(g_model);
        g_model = nullptr;
        return -2;
    }
    
    LOGI("Context created successfully");
    
    return 0; // Success
}

JNIEXPORT jstring JNICALL
Java_com_example_llama_LlamaJNI_generateText(
        JNIEnv* env, jobject /* this */, jstring prompt) {
    
    if (!g_model || !g_ctx) {
        return env->NewStringUTF("Error: Model not initialized");
    }
    
    std::string prompt_str = jstring_to_string(env, prompt);
    LOGI("Generating text for prompt: %s", prompt_str.c_str());
    
    // Tokenize the prompt
    std::vector<llama_token> tokens_list;
    tokens_list = llama_tokenize(g_ctx, prompt_str, false);
    
    if (tokens_list.empty()) {
        return env->NewStringUTF("Error: Failed to tokenize prompt");
    }
    
    int n_ctx = llama_n_ctx(g_ctx);
    int n_past = 0;
    
    // Evaluate prompt
    if (llama_decode(g_ctx, llama_batch_get_one(tokens_list.data(), tokens_list.size(), n_past, 0))) {
        LOGE("Failed to evaluate prompt");
        return env->NewStringUTF("Error: Failed to evaluate prompt");
    }
    
    n_past += tokens_list.size();
    
    // Generate response
    std::string response;
    for (int i = 0; i < g_params.n_predict; i++) {
        // Sample next token
        llama_token id = 0;
        {
            auto logits = llama_get_logits(g_ctx);
            auto n_vocab = llama_n_vocab(g_model);
            
            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }
            
            llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};
            
            // Apply sampling
            llama_sample_top_k(g_ctx, candidates_p.data(), g_params.top_k, 1);
            llama_sample_top_p(g_ctx, candidates_p.data(), g_params.top_p, 1);
            llama_sample_temp(g_ctx, candidates_p.data(), g_params.temp);
            
            id = llama_sample_token(g_ctx, candidates_p.data());
        }
        
        // Check for EOS
        if (id == llama_token_eos(g_model) || n_past >= n_ctx) {
            break;
        }
        
        // Convert token to string
        std::string token_str = llama_token_to_piece(g_ctx, id);
        response += token_str;
        
        // Evaluate the token
        if (llama_decode(g_ctx, llama_batch_get_one(&id, 1, n_past, 0))) {
            LOGW("Failed to evaluate token");
            break;
        }
        
        n_past++;
    }
    
    LOGI("Generation complete, response length: %zu", response.length());
    return env->NewStringUTF(response.c_str());
}

JNIEXPORT void JNICALL
Java_com_example_llama_LlamaJNI_cleanup(
        JNIEnv* /* env */, jobject /* this */) {
    
    LOGI("Cleaning up llama.cpp resources");
    
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }
    
    llama_backend_free();
}

JNIEXPORT jint JNICALL
Java_com_example_llama_LlamaJNI_getContextSize(
        JNIEnv* /* env */, jobject /* this */) {
    
    if (!g_ctx) return 0;
    return llama_n_ctx(g_ctx);
}

JNIEXPORT jfloatArray JNICALL
Java_com_example_llama_LlamaJNI_getEmbedding(
        JNIEnv* env, jobject /* this */, jstring text) {
    
    if (!g_model || !g_ctx) {
        return nullptr;
    }
    
    std::string text_str = jstring_to_string(env, text);
    
    // Tokenize
    std::vector<llama_token> tokens = llama_tokenize(g_ctx, text_str, false);
    if (tokens.empty()) {
        return nullptr;
    }
    
    // Get embedding dimension
    int n_embd = llama_n_embd(g_model);
    
    // Create output array
    jfloatArray result = env->NewFloatArray(n_embd);
    if (!result) {
        return nullptr;
    }
    
    // Evaluate tokens
    llama_decode(g_ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0));
    
    // Get embeddings (for simplicity, using last token's embedding)
    const float* embeddings = llama_get_embeddings(g_ctx);
    if (embeddings) {
        env->SetFloatArrayRegion(result, 0, n_embd, embeddings);
    }
    
    return result;
}

}