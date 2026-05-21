package com.aiope.inf

/**
 * JNI bridge to llama.cpp native library.
 * Handles model loading, inference, streaming, quantization, and GPU management.
 */
class LlamaJNI {

    companion object {
        init {
            System.loadLibrary("aiope-inf")
        }
    }

    // ============================================================
    // Model lifecycle
    // ============================================================
    external fun loadModel(
        modelPath: String,
        nGpuLayers: Int = 0,
        contextSize: Int = 2048,
        batchSize: Int = 512,
        useVulkan: Boolean = true
    ): Boolean

    external fun unloadModel()
    external fun getModelInfo(): String

    // ============================================================
    // Inference
    // ============================================================
    external fun generate(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
        repeatPenalty: Float = 1.1f
    ): String

    external fun abort()

    // ============================================================
    // Streaming
    // ============================================================
    external fun generateStreaming(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
        repeatPenalty: Float = 1.1f,
        callback: StreamCallback
    )

    external fun generateSSE(
        prompt: String,
        modelName: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
        tokenCallback: TokenCallback?
    ): String

    // ============================================================
    // Multimodal (image/audio)
    // ============================================================
    external fun initMultimodal(mmProjPath: String): Boolean
    external fun generateWithImage(
        prompt: String,
        imageData: ByteArray,
        width: Int,
        height: Int,
        maxTokens: Int = 512,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): String
    external fun freeMultimodal()

    // ============================================================
    // LoRA adapters
    // ============================================================
    external fun loadLoraAdapter(loraPath: String, scale: Float = 1.0f): Boolean
    external fun unloadLoraAdapter()

    // ============================================================
    // Chat template
    // ============================================================
    external fun applyChatTemplate(messagesJson: String): String

    // ============================================================
    // Quantization
    // ============================================================
    external fun quantizeModel(
        inputPath: String,
        outputPath: String,
        quantType: Int,
        nThreads: Int = 4,
        progressCallback: QuantizeProgressCallback?
    ): Boolean

    external fun getQuantizationTypes(): String
    external fun estimateQuantizedSize(modelPath: String, quantType: Int): Long

    // ============================================================
    // GPU
    // ============================================================
    external fun initGpu(): Boolean
    external fun isVulkanAvailable(): Boolean
    external fun getGpuInfo(): String
    external fun recommendGpuLayers(modelSizeBytes: Long, totalLayers: Int): Int
    external fun benchmarkBackends(nTokens: Int): String
    external fun getBackendInfo(): String

    // ============================================================
    // Callback interfaces
    // ============================================================
    interface StreamCallback {
        /** Called for each token. Return false to stop generation. */
        fun onToken(token: String): Boolean
        /** Called when generation completes. */
        fun onComplete(fullResponse: String)
        /** Called on error. */
        fun onError(error: String)
    }

    interface TokenCallback {
        /** Called for each token. Return false to stop. */
        fun onToken(token: String): Boolean
    }

    interface QuantizeProgressCallback {
        fun onProgress(progress: Float)
    }
}
