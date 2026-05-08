#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <fstream>

#include "llama.h"

#define TAG "AIOPE-QUANT"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ============================================================
// Quantization types mapping
// ============================================================
static llama_ftype get_ftype(int type) {
    switch (type) {
        case 0: return LLAMA_FTYPE_MOSTLY_Q4_0;
        case 1: return LLAMA_FTYPE_MOSTLY_Q4_1;
        case 2: return LLAMA_FTYPE_MOSTLY_Q5_0;
        case 3: return LLAMA_FTYPE_MOSTLY_Q5_1;
        case 4: return LLAMA_FTYPE_MOSTLY_Q8_0;
        case 5: return LLAMA_FTYPE_MOSTLY_Q2_K;
        case 6: return LLAMA_FTYPE_MOSTLY_Q3_K_S;
        case 7: return LLAMA_FTYPE_MOSTLY_Q3_K_M;
        case 8: return LLAMA_FTYPE_MOSTLY_Q3_K_L;
        case 9: return LLAMA_FTYPE_MOSTLY_Q4_K_S;
        case 10: return LLAMA_FTYPE_MOSTLY_Q4_K_M;
        case 11: return LLAMA_FTYPE_MOSTLY_Q5_K_S;
        case 12: return LLAMA_FTYPE_MOSTLY_Q5_K_M;
        case 13: return LLAMA_FTYPE_MOSTLY_Q6_K;
        default: return LLAMA_FTYPE_MOSTLY_Q4_0;
    }
}

// ============================================================
// JNI: Quantize model on-device
// ============================================================
extern "C" JNIEXPORT jboolean JNICALL
Java_com_aiope_inf_LlamaJNI_quantizeModel(
    JNIEnv *env, jobject /* this */,
    jstring inputPath,
    jstring outputPath,
    jint quantType,
    jint nThreads,
    jobject progressCallback) {

    const char *in_chars = env->GetStringUTFChars(inputPath, nullptr);
    const char *out_chars = env->GetStringUTFChars(outputPath, nullptr);
    std::string input(in_chars);
    std::string output(out_chars);
    env->ReleaseStringUTFChars(inputPath, in_chars);
    env->ReleaseStringUTFChars(outputPath, out_chars);

    LOGI("Quantizing: %s -> %s (type=%d, threads=%d)",
         input.c_str(), output.c_str(), quantType, nThreads);

    // Get progress callback
    jclass cbCls = nullptr;
    jmethodID cbMethod = nullptr;
    if (progressCallback) {
        cbCls = env->GetObjectClass(progressCallback);
        cbMethod = env->GetMethodID(cbCls, "onProgress", "(F)V");
    }

    // Quantization params
    llama_model_quantize_params params = llama_model_quantize_default_params();
    params.nthread = nThreads > 0 ? nThreads : 4;
    params.ftype = get_ftype(quantType);
    params.allow_requantize = false;
    params.quantize_output_tensor = true;

    // Run quantization
    uint32_t result = llama_model_quantize(input.c_str(), output.c_str(), &params);

    if (result != 0) {
        LOGE("Quantization failed with code: %d", result);
        return JNI_FALSE;
    }

    LOGI("Quantization complete: %s", output.c_str());

    // Report completion
    if (cbMethod) {
        env->CallVoidMethod(progressCallback, cbMethod, 1.0f);
    }

    return JNI_TRUE;
}

// ============================================================
// JNI: Get quantization type info
// ============================================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_aiope_inf_LlamaJNI_getQuantizationTypes(JNIEnv *env, jobject /* this */) {
    std::string json = "[";
    json += "{\"id\":0,\"name\":\"Q4_0\",\"bits\":4,\"desc\":\"Small, low quality\"},";
    json += "{\"id\":1,\"name\":\"Q4_1\",\"bits\":4,\"desc\":\"Small, slightly better\"},";
    json += "{\"id\":2,\"name\":\"Q5_0\",\"bits\":5,\"desc\":\"Medium, balanced\"},";
    json += "{\"id\":3,\"name\":\"Q5_1\",\"bits\":5,\"desc\":\"Medium, slightly better\"},";
    json += "{\"id\":4,\"name\":\"Q8_0\",\"bits\":8,\"desc\":\"Large, high quality\"},";
    json += "{\"id\":5,\"name\":\"Q2_K\",\"bits\":2,\"desc\":\"Smallest, lowest quality\"},";
    json += "{\"id\":6,\"name\":\"Q3_K_S\",\"bits\":3,\"desc\":\"Very small\"},";
    json += "{\"id\":7,\"name\":\"Q3_K_M\",\"bits\":3,\"desc\":\"Small\"},";
    json += "{\"id\":8,\"name\":\"Q3_K_L\",\"bits\":3,\"desc\":\"Small-medium\"},";
    json += "{\"id\":9,\"name\":\"Q4_K_S\",\"bits\":4,\"desc\":\"Medium, good quality\"},";
    json += "{\"id\":10,\"name\":\"Q4_K_M\",\"bits\":4,\"desc\":\"Medium, recommended\"},";
    json += "{\"id\":11,\"name\":\"Q5_K_S\",\"bits\":5,\"desc\":\"Large, very good\"},";
    json += "{\"id\":12,\"name\":\"Q5_K_M\",\"bits\":5,\"desc\":\"Large, excellent\"},";
    json += "{\"id\":13,\"name\":\"Q6_K\",\"bits\":6,\"desc\":\"Very large, near lossless\"}";
    json += "]";
    return env->NewStringUTF(json.c_str());
}

// ============================================================
// JNI: Estimate quantized model size
// ============================================================
extern "C" JNIEXPORT jlong JNICALL
Java_com_aiope_inf_LlamaJNI_estimateQuantizedSize(
    JNIEnv *env, jobject /* this */,
    jstring modelPath,
    jint quantType) {

    const char *path = env->GetStringUTFChars(modelPath, nullptr);

    // Get original file size
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!file.is_open()) return -1;

    long original_size = file.tellg();
    file.close();

    // Estimate based on quantization ratio
    // Original is typically F16 (16 bits per weight)
    float ratio;
    switch (quantType) {
        case 0: ratio = 0.25f; break;   // Q4_0: 4/16
        case 1: ratio = 0.28f; break;   // Q4_1
        case 2: ratio = 0.31f; break;   // Q5_0
        case 3: ratio = 0.34f; break;   // Q5_1
        case 4: ratio = 0.50f; break;   // Q8_0
        case 5: ratio = 0.15f; break;   // Q2_K
        case 6: ratio = 0.20f; break;   // Q3_K_S
        case 7: ratio = 0.22f; break;   // Q3_K_M
        case 8: ratio = 0.24f; break;   // Q3_K_L
        case 9: ratio = 0.26f; break;   // Q4_K_S
        case 10: ratio = 0.28f; break;  // Q4_K_M
        case 11: ratio = 0.33f; break;  // Q5_K_S
        case 12: ratio = 0.35f; break;  // Q5_K_M
        case 13: ratio = 0.40f; break;  // Q6_K
        default: ratio = 0.25f;
    }

    return (long)(original_size * ratio);
}
