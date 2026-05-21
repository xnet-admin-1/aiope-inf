package com.aiope.inf

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * High-level model manager. Handles downloading, caching, loading,
 * and provides coroutine-friendly inference APIs.
 */
class ModelManager(private val context: Context) {

    private val jni = LlamaJNI()
    private val modelDir = File(context.filesDir, "models")
    private var currentModel: String? = null

    init {
        modelDir.mkdirs()
    }

    // ============================================================
    // Model lifecycle
    // ============================================================

    data class LoadConfig(
        val gpuLayers: Int = 0,
        val contextSize: Int = 2048,
        val batchSize: Int = 512,
        val useVulkan: Boolean = true,
        val autoGpuLayers: Boolean = true
    )

    suspend fun loadModel(path: String, config: LoadConfig = LoadConfig()): Boolean {
        return withContext(Dispatchers.IO) {
            val file = File(path)
            if (!file.exists()) {
                throw IllegalArgumentException("Model not found: $path")
            }

            var gpuLayers = config.gpuLayers
            if (config.autoGpuLayers && config.useVulkan) {
                jni.initGpu()
                gpuLayers = jni.recommendGpuLayers(file.length(), 32)
            }

            val success = jni.loadModel(
                modelPath = path,
                nGpuLayers = gpuLayers,
                contextSize = config.contextSize,
                batchSize = config.batchSize,
                useVulkan = config.useVulkan
            )

            if (success) currentModel = file.name
            success
        }
    }

    fun unload() {
        jni.unloadModel()
        currentModel = null
    }

    fun isLoaded(): Boolean = currentModel != null

    fun getModelInfo(): JSONObject {
        return JSONObject(jni.getModelInfo())
    }

    // ============================================================
    // Inference
    // ============================================================

    data class GenerateParams(
        val maxTokens: Int = 512,
        val temperature: Float = 0.7f,
        val topP: Float = 0.9f,
        val repeatPenalty: Float = 1.1f
    )

    suspend fun generate(prompt: String, params: GenerateParams = GenerateParams()): String {
        return withContext(Dispatchers.IO) {
            jni.generate(prompt, params.maxTokens, params.temperature, params.topP, params.repeatPenalty)
        }
    }

    /**
     * Stream tokens as a Kotlin Flow.
     */
    fun generateStream(prompt: String, params: GenerateParams = GenerateParams()): Flow<String> {
        return callbackFlow {
            jni.generateStreaming(
                prompt = prompt,
                maxTokens = params.maxTokens,
                temperature = params.temperature,
                topP = params.topP,
                repeatPenalty = params.repeatPenalty,
                callback = object : LlamaJNI.StreamCallback {
                    override fun onToken(token: String): Boolean {
                        val result = trySend(token)
                        return result.isSuccess
                    }

                    override fun onComplete(fullResponse: String) {
                        close()
                    }

                    override fun onError(error: String) {
                        close(RuntimeException(error))
                    }
                }
            )
            awaitClose { jni.abort() }
        }
    }

    fun abort() = jni.abort()

    // ============================================================
    // LoRA adapters
    // ============================================================

    suspend fun loadLoraAdapter(path: String, scale: Float = 1.0f): Boolean {
        return withContext(Dispatchers.IO) {
            jni.loadLoraAdapter(path, scale)
        }
    }

    fun unloadLoraAdapter() = jni.unloadLoraAdapter()

    // ============================================================
    // Chat template (auto-detect from model)
    // ============================================================

    fun applyChatTemplate(messages: List<Pair<String, String>>): String {
        val json = JSONArray().apply {
            messages.forEach { (role, content) ->
                put(JSONObject().put("role", role).put("content", content))
            }
        }
        return jni.applyChatTemplate(json.toString())
    }

    // ============================================================
    // Model files
    // ============================================================

    fun listModels(): List<ModelFile> {
        return modelDir.listFiles()
            ?.filter { it.extension == "gguf" }
            ?.map { ModelFile(it.name, it.length(), it.absolutePath) }
            ?: emptyList()
    }

    fun deleteModel(filename: String): Boolean {
        return File(modelDir, filename).delete()
    }

    fun getModelPath(filename: String): String {
        return File(modelDir, filename).absolutePath
    }

    // ============================================================
    // Quantization
    // ============================================================

    suspend fun quantize(
        inputPath: String,
        outputFilename: String,
        quantType: Int,
        onProgress: ((Float) -> Unit)? = null
    ): Boolean {
        return withContext(Dispatchers.IO) {
            val outputPath = File(modelDir, outputFilename).absolutePath
            jni.quantizeModel(
                inputPath = inputPath,
                outputPath = outputPath,
                quantType = quantType,
                nThreads = Runtime.getRuntime().availableProcessors(),
                progressCallback = onProgress?.let {
                    object : LlamaJNI.QuantizeProgressCallback {
                        override fun onProgress(progress: Float) { it(progress) }
                    }
                }
            )
        }
    }

    fun getQuantizationTypes(): JSONArray {
        return JSONArray(jni.getQuantizationTypes())
    }

    fun estimateQuantizedSize(modelPath: String, quantType: Int): Long {
        return jni.estimateQuantizedSize(modelPath, quantType)
    }

    // ============================================================
    // GPU
    // ============================================================

    fun getGpuInfo(): JSONObject = JSONObject(jni.getGpuInfo())
    fun getBackendInfo(): JSONObject = JSONObject(jni.getBackendInfo())

    // ============================================================
    // Data classes
    // ============================================================

    data class ModelFile(
        val name: String,
        val sizeBytes: Long,
        val path: String
    ) {
        val sizeMB: Long get() = sizeBytes / (1024 * 1024)
    }
}
