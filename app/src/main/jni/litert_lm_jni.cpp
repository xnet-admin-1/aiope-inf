#include <jni.h>
#include <android/log.h>
#include <string>
#include <dlfcn.h>
#include "litert_lm/engine.h"

#define TAG "LiteRT-LM-JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static LiteRtLmEngine* g_engine = nullptr;
static LiteRtLmSession* g_session = nullptr;
static LiteRtLmConversation* g_conversation = nullptr;

// Helper: throw Java exception
static void throwException(JNIEnv* env, const char* msg) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), msg);
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_aiope_inf_LiteRTEngine_nativeLoad(JNIEnv* env, jobject,
    jstring modelPath, jstring backend, jint maxTokens, jstring cacheDir, jstring nativeLibDir) {

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    const char* be = env->GetStringUTFChars(backend, nullptr);
    const char* cache = env->GetStringUTFChars(cacheDir, nullptr);
    const char* libDir = env->GetStringUTFChars(nativeLibDir, nullptr);

    // Cleanup previous
    if (g_conversation) { litert_lm_conversation_delete(g_conversation); g_conversation = nullptr; }
    if (g_session) { litert_lm_session_delete(g_session); g_session = nullptr; }
    if (g_engine) { litert_lm_engine_delete(g_engine); g_engine = nullptr; }

    LiteRtLmEngineSettings* settings = litert_lm_engine_settings_create(path, be, nullptr, nullptr);
    if (!settings) {
        LOGE("Failed to create engine settings for %s", path);
        env->ReleaseStringUTFChars(modelPath, path);
        env->ReleaseStringUTFChars(backend, be);
        env->ReleaseStringUTFChars(cacheDir, cache);
        env->ReleaseStringUTFChars(nativeLibDir, libDir);
        return JNI_FALSE;
    }

    litert_lm_engine_settings_set_max_num_tokens(settings, (int)maxTokens);
    litert_lm_engine_settings_set_cache_dir(settings, cache);
    litert_lm_engine_settings_set_litert_dispatch_lib_dir(settings, libDir);

    g_engine = litert_lm_engine_create(settings);
    litert_lm_engine_settings_delete(settings);

    env->ReleaseStringUTFChars(modelPath, path);
    env->ReleaseStringUTFChars(backend, be);
    env->ReleaseStringUTFChars(cacheDir, cache);
    env->ReleaseStringUTFChars(nativeLibDir, libDir);

    if (!g_engine) {
        LOGE("Engine creation failed");
        return JNI_FALSE;
    }
    LOGI("Engine created successfully");
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_aiope_inf_LiteRTEngine_nativeCreateConversation(JNIEnv* env, jobject,
    jstring systemPrompt, jint topK, jdouble topP, jdouble temperature, jint maxOutputTokens, jstring loraPath) {

    if (!g_engine) { throwException(env, "Engine not loaded"); return; }

    if (g_conversation) { litert_lm_conversation_delete(g_conversation); g_conversation = nullptr; }

    LiteRtLmSessionConfig* sessionConfig = litert_lm_session_config_create();
    LiteRtLmSamplerParams sampler = {kLiteRtLmSamplerTypeTopK, (int32_t)topK, (float)topP, (float)temperature, 0};
    litert_lm_session_config_set_sampler_params(sessionConfig, &sampler);
    litert_lm_session_config_set_max_output_tokens(sessionConfig, (int)maxOutputTokens);

    if (loraPath) {
        const char* lora = env->GetStringUTFChars(loraPath, nullptr);
        if (lora && lora[0] != '\0') {
            litert_lm_session_config_set_lora_path(sessionConfig, lora);
        }
        env->ReleaseStringUTFChars(loraPath, lora);
    }

    LiteRtLmConversationConfig* convConfig = litert_lm_conversation_config_create();
    litert_lm_conversation_config_set_session_config(convConfig, sessionConfig);

    if (systemPrompt) {
        const char* sys = env->GetStringUTFChars(systemPrompt, nullptr);
        if (sys && sys[0] != '\0') {
            litert_lm_conversation_config_set_system_message(convConfig, sys);
        }
        env->ReleaseStringUTFChars(systemPrompt, sys);
    }

    g_conversation = litert_lm_conversation_create(g_engine, convConfig);
    litert_lm_conversation_config_delete(convConfig);
    litert_lm_session_config_delete(sessionConfig);

    if (!g_conversation) {
        LOGE("Conversation creation failed");
        throwException(env, "Failed to create conversation");
    }
}

JNIEXPORT jstring JNICALL
Java_com_aiope_inf_LiteRTEngine_nativeGenerate(JNIEnv* env, jobject, jstring messageJson) {
    if (!g_conversation) { throwException(env, "No conversation"); return nullptr; }

    const char* msg = env->GetStringUTFChars(messageJson, nullptr);
    LiteRtLmJsonResponse* response = litert_lm_conversation_send_message(g_conversation, msg, nullptr, nullptr);
    env->ReleaseStringUTFChars(messageJson, msg);

    if (!response) return env->NewStringUTF("");

    const char* text = litert_lm_json_response_get_string(response);
    jstring result = env->NewStringUTF(text ? text : "");
    litert_lm_json_response_delete(response);
    return result;
}

struct StreamContext {
    JNIEnv* env;
    jobject callback;
    jmethodID onToken;
    jmethodID onDone;
    jmethodID onError;
    JavaVM* jvm;
};

static void streamCallback(void* data, const char* chunk, bool is_final, const char* error_msg) {
    auto* ctx = reinterpret_cast<StreamContext*>(data);
    JNIEnv* env;
    bool attached = false;
    if (ctx->jvm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        ctx->jvm->AttachCurrentThread(&env, nullptr);
        attached = true;
    }

    if (error_msg) {
        jstring err = env->NewStringUTF(error_msg);
        env->CallVoidMethod(ctx->callback, ctx->onError, err);
        env->DeleteLocalRef(err);
    } else if (is_final) {
        env->CallVoidMethod(ctx->callback, ctx->onDone);
    } else if (chunk) {
        jstring token = env->NewStringUTF(chunk);
        env->CallVoidMethod(ctx->callback, ctx->onToken, token);
        env->DeleteLocalRef(token);
    }

    if (is_final || error_msg) {
        env->DeleteGlobalRef(ctx->callback);
        delete ctx;
    }
    if (attached) ctx->jvm->DetachCurrentThread();
}

JNIEXPORT void JNICALL
Java_com_aiope_inf_LiteRTEngine_nativeGenerateStream(JNIEnv* env, jobject, jstring messageJson, jobject callback) {
    if (!g_conversation) { throwException(env, "No conversation"); return; }

    jclass cbClass = env->GetObjectClass(callback);
    auto* ctx = new StreamContext();
    env->GetJavaVM(&ctx->jvm);
    ctx->callback = env->NewGlobalRef(callback);
    ctx->onToken = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)V");
    ctx->onDone = env->GetMethodID(cbClass, "onDone", "()V");
    ctx->onError = env->GetMethodID(cbClass, "onError", "(Ljava/lang/String;)V");

    const char* msg = env->GetStringUTFChars(messageJson, nullptr);
    int result = litert_lm_conversation_send_message_stream(g_conversation, msg, nullptr, nullptr, streamCallback, ctx);
    env->ReleaseStringUTFChars(messageJson, msg);

    if (result != 0) {
        env->DeleteGlobalRef(ctx->callback);
        delete ctx;
        throwException(env, "Failed to start stream");
    }
}

JNIEXPORT void JNICALL
Java_com_aiope_inf_LiteRTEngine_nativeCancel(JNIEnv*, jobject) {
    if (g_conversation) litert_lm_conversation_cancel_process(g_conversation);
}

JNIEXPORT void JNICALL
Java_com_aiope_inf_LiteRTEngine_nativeClose(JNIEnv*, jobject) {
    if (g_conversation) { litert_lm_conversation_delete(g_conversation); g_conversation = nullptr; }
    if (g_session) { litert_lm_session_delete(g_session); g_session = nullptr; }
    if (g_engine) { litert_lm_engine_delete(g_engine); g_engine = nullptr; }
}

JNIEXPORT jstring JNICALL
Java_com_aiope_inf_LiteRTEngine_nativeTokenize(JNIEnv* env, jobject, jstring text) {
    if (!g_engine) { throwException(env, "Engine not loaded"); return nullptr; }
    const char* t = env->GetStringUTFChars(text, nullptr);
    LiteRtLmTokenizeResult* result = litert_lm_engine_tokenize(g_engine, t);
    env->ReleaseStringUTFChars(text, t);
    if (!result) return env->NewStringUTF("[]");

    size_t n = litert_lm_tokenize_result_get_num_tokens(result);
    const int* tokens = litert_lm_tokenize_result_get_tokens(result);
    std::string json = "[";
    for (size_t i = 0; i < n; i++) {
        if (i > 0) json += ",";
        json += std::to_string(tokens[i]);
    }
    json += "]";
    litert_lm_tokenize_result_delete(result);
    return env->NewStringUTF(json.c_str());
}

JNIEXPORT jint JNICALL
Java_com_aiope_inf_LiteRTEngine_nativeGetTokenCount(JNIEnv*, jobject) {
    if (!g_conversation) return -1;
    return litert_lm_conversation_get_token_count(g_conversation);
}

} // extern "C"
