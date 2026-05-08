#include <jni.h>
#include <android/log.h>
#include <string>

#ifdef AIOPE_VULKAN
#include <vulkan/vulkan.h>
#endif

#define TAG "AIOPE-GPU"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ============================================================
// GPU Backend Detection & Info
// ============================================================

struct GpuInfo {
    bool vulkan_available = false;
    std::string device_name;
    uint32_t api_version = 0;
    uint64_t vram_size = 0;
    uint32_t max_compute_units = 0;
    uint32_t max_workgroup_size = 0;
};

static GpuInfo g_gpu_info;

#ifdef AIOPE_VULKAN
static bool probe_vulkan() {
    VkInstance instance = VK_NULL_HANDLE;
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "AIOPE-INF";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "llama.cpp";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkResult result = vkCreateInstance(&create_info, nullptr, &instance);
    if (result != VK_SUCCESS) {
        LOGE("Vulkan instance creation failed: %d", result);
        return false;
    }

    // Enumerate physical devices
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

    if (device_count == 0) {
        LOGE("No Vulkan-capable GPU found");
        vkDestroyInstance(instance, nullptr);
        return false;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    // Get first device properties
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devices[0], &props);

    g_gpu_info.vulkan_available = true;
    g_gpu_info.device_name = props.deviceName;
    g_gpu_info.api_version = props.apiVersion;

    // Get memory properties
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(devices[0], &mem_props);

    // Find device-local heap size (VRAM equivalent)
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            g_gpu_info.vram_size = mem_props.memoryHeaps[i].size;
            break;
        }
    }

    // Get compute queue family properties
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(devices[0], &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(devices[0], &queue_family_count, queue_families.data());

    for (const auto &qf : queue_families) {
        if (qf.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            g_gpu_info.max_compute_units = qf.queueCount;
            break;
        }
    }

    LOGI("Vulkan GPU: %s (API %d.%d.%d, VRAM: %llu MB)",
         g_gpu_info.device_name.c_str(),
         VK_VERSION_MAJOR(g_gpu_info.api_version),
         VK_VERSION_MINOR(g_gpu_info.api_version),
         VK_VERSION_PATCH(g_gpu_info.api_version),
         (unsigned long long)(g_gpu_info.vram_size / (1024 * 1024)));

    vkDestroyInstance(instance, nullptr);
    return true;
}
#endif

// ============================================================
// JNI: Initialize GPU backend
// ============================================================
extern "C" JNIEXPORT jboolean JNICALL
Java_com_aiope_inf_LlamaJNI_initGpu(JNIEnv *env, jobject /* this */) {
#ifdef AIOPE_VULKAN
    bool ok = probe_vulkan();
    if (ok) {
        LOGI("GPU backend initialized: Vulkan");
    }
    return ok ? JNI_TRUE : JNI_FALSE;
#else
    LOGI("GPU backend not available (compiled without Vulkan)");
    return JNI_FALSE;
#endif
}

// ============================================================
// JNI: Get GPU info as JSON
// ============================================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_aiope_inf_LlamaJNI_getGpuInfo(JNIEnv *env, jobject /* this */) {
    std::string json = "{";
    json += "\"available\":" + std::string(g_gpu_info.vulkan_available ? "true" : "false") + ",";
    json += "\"backend\":\"vulkan\",";
    json += "\"device_name\":\"" + g_gpu_info.device_name + "\",";
    json += "\"api_version\":\"" +
        std::to_string(VK_VERSION_MAJOR(g_gpu_info.api_version)) + "." +
        std::to_string(VK_VERSION_MINOR(g_gpu_info.api_version)) + "." +
        std::to_string(VK_VERSION_PATCH(g_gpu_info.api_version)) + "\",";
    json += "\"vram_mb\":" + std::to_string(g_gpu_info.vram_size / (1024 * 1024)) + ",";
    json += "\"compute_units\":" + std::to_string(g_gpu_info.max_compute_units);
    json += "}";
    return env->NewStringUTF(json.c_str());
}

// ============================================================
// JNI: Recommend GPU layers based on model size and VRAM
// ============================================================
extern "C" JNIEXPORT jint JNICALL
Java_com_aiope_inf_LlamaJNI_recommendGpuLayers(
    JNIEnv *env, jobject /* this */,
    jlong modelSizeBytes,
    jint totalLayers) {

    if (!g_gpu_info.vulkan_available || g_gpu_info.vram_size == 0) {
        return 0;  // No GPU, use CPU only
    }

    // Reserve 256MB for overhead
    uint64_t usable_vram = g_gpu_info.vram_size > (256 * 1024 * 1024)
        ? g_gpu_info.vram_size - (256 * 1024 * 1024)
        : 0;

    if (usable_vram == 0) return 0;

    // Estimate bytes per layer
    uint64_t bytes_per_layer = modelSizeBytes / totalLayers;

    // How many layers fit in VRAM
    int gpu_layers = (int)(usable_vram / bytes_per_layer);

    // Clamp to total layers
    if (gpu_layers > totalLayers) gpu_layers = totalLayers;

    LOGI("Recommended GPU layers: %d/%d (VRAM: %llu MB, model: %lld MB)",
         gpu_layers, totalLayers,
         (unsigned long long)(g_gpu_info.vram_size / (1024 * 1024)),
         (long long)(modelSizeBytes / (1024 * 1024)));

    return gpu_layers;
}

// ============================================================
// JNI: Benchmark GPU vs CPU inference speed
// ============================================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_aiope_inf_LlamaJNI_benchmarkBackends(
    JNIEnv *env, jobject /* this */,
    jint nTokens) {

    // Return benchmark placeholder - actual benchmarking requires loaded model
    std::string json = "{";
    json += "\"gpu_available\":" + std::string(g_gpu_info.vulkan_available ? "true" : "false") + ",";
    json += "\"device\":\"" + g_gpu_info.device_name + "\",";
    json += "\"n_tokens\":" + std::to_string(nTokens) + ",";
    json += "\"note\":\"Load a model first, then run benchmark\"";
    json += "}";
    return env->NewStringUTF(json.c_str());
}
