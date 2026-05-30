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
class ModelManager private constructor(private val context: Context) {

    companion object {
        @Volatile private var instance: ModelManager? = null
        fun getInstance(context: Context): ModelManager =
            instance ?: synchronized(this) { instance ?: ModelManager(context.applicationContext).also { instance = it } }
    }

    private val jni = LlamaJNI()
    private val modelDir = File(context.filesDir, "models")
    private var currentModel: String? = null
    private var currentLora: String? = null
    private var litertEngine: LiteRTEngine? = null
    private var activeBackend: InferenceBackend = InferenceBackend.LLAMA_CPP

    enum class InferenceBackend { LLAMA_CPP, LITERT_LM }

    init {
        modelDir.mkdirs()
        litertEngine = LiteRTEngine(context)
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

            // Route to LiteRT-LM for .litertlm files
            if (path.endsWith(".litertlm")) {
                val accelerator = if (config.useVulkan) LiteRTEngine.AcceleratorType.GPU
                    else LiteRTEngine.AcceleratorType.CPU
                val success = litertEngine!!.load(path, accelerator, config.contextSize)
                if (success) {
                    litertEngine!!.createConversation()
                    currentModel = file.name
                    activeBackend = InferenceBackend.LITERT_LM
                }
                android.util.Log.d("ModelManager", "LiteRT-LM load=$success path=$path")
                return@withContext success
            }

            // GGUF path via llama.cpp
            var gpuLayers = config.gpuLayers
            if (config.autoGpuLayers && config.useVulkan) {
                try {
                    val gpuOk = jni.initGpu()
                    if (gpuOk) {
                        gpuLayers = jni.recommendGpuLayers(file.length(), 32)
                    }
                } catch (e: Exception) {
                    android.util.Log.e("ModelManager", "GPU init failed, using CPU only", e)
                    gpuLayers = 0
                }
            }

            val success = jni.loadModel(
                modelPath = path,
                nGpuLayers = gpuLayers,
                contextSize = config.contextSize,
                batchSize = config.batchSize,
                useVulkan = config.useVulkan && gpuLayers > 0
            )

            if (success) {
                currentModel = file.name
                activeBackend = InferenceBackend.LLAMA_CPP
            }
            android.util.Log.d("ModelManager", "loadModel result=$success path=$path gpuLayers=$gpuLayers currentModel=$currentModel")
            success
        }
    }

    fun unload() {
        jni.unloadModel()
        litertEngine?.close()
        currentModel = null
        activeBackend = InferenceBackend.LLAMA_CPP
    }

    fun isLoaded(): Boolean = currentModel != null || try { JSONObject(jni.getModelInfo()).has("n_params") } catch (_: Exception) { false }

    fun getActiveBackend(): InferenceBackend = activeBackend

    /**
     * Wraps a user prompt in the model's chat template via native Jinja engine.
     * Falls back to ChatML if the model has no embedded template.
     */
    private fun buildChatPrompt(userMessage: String): String {
        val messages = JSONArray().apply {
            put(JSONObject().put("role", "user").put("content", userMessage))
        }
        val templated = jni.applyChatTemplate(messages.toString(), null)
        if (templated.isNotEmpty()) return templated
        // Fallback: ChatML
        return "<|im_start|>user\n$userMessage<|im_end|>\n<|im_start|>assistant\n"
    }

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
        if (activeBackend == InferenceBackend.LITERT_LM) {
            return litertEngine!!.generate(prompt)
        }
        return withContext(Dispatchers.IO) {
            val chatPrompt = buildChatPrompt(prompt)
            jni.generate(chatPrompt, params.maxTokens, params.temperature, params.topP, params.repeatPenalty)
        }
    }

    /**
     * Stream tokens as a Kotlin Flow.
     */
    fun generateStream(prompt: String, params: GenerateParams = GenerateParams()): Flow<String> {
        if (activeBackend == InferenceBackend.LITERT_LM) {
            return litertEngine!!.generateStream(prompt)
        }
        return callbackFlow {
            val chatPrompt = buildChatPrompt(prompt)
            val thread = Thread {
                jni.generateStreaming(
                    prompt = chatPrompt,
                    maxTokens = params.maxTokens,
                    temperature = params.temperature,
                    topP = params.topP,
                    repeatPenalty = params.repeatPenalty,
                    callback = object : LlamaJNI.StreamCallback {
                        override fun onToken(token: String): Boolean = trySend(token).isSuccess
                        override fun onComplete(fullResponse: String) { close() }
                        override fun onError(error: String) { close(RuntimeException(error)) }
                    }
                )
            }
            thread.start()
            awaitClose { jni.abort() }
        }
    }

    fun abort() {
        jni.abort()
        litertEngine?.stop()
    }

    /** Generate with an already-templated prompt (no chat template applied). Used by OpenAI server. */
    suspend fun generateRaw(prompt: String, params: GenerateParams = GenerateParams()): String {
        return withContext(Dispatchers.IO) {
            jni.generate(prompt, params.maxTokens, params.temperature, params.topP, params.repeatPenalty)
        }
    }

    /** Stream with an already-templated prompt. Used by OpenAI server. */
    fun generateStreamRaw(prompt: String, params: GenerateParams = GenerateParams()): Flow<String> {
        return callbackFlow {
            val thread = Thread {
                jni.generateStreaming(
                    prompt = prompt,
                    maxTokens = params.maxTokens,
                    temperature = params.temperature,
                    topP = params.topP,
                    repeatPenalty = params.repeatPenalty,
                    callback = object : LlamaJNI.StreamCallback {
                        override fun onToken(token: String): Boolean = trySend(token).isSuccess
                        override fun onComplete(fullResponse: String) { close() }
                        override fun onError(error: String) { close(RuntimeException(error)) }
                    }
                )
            }
            thread.start()
            awaitClose { jni.abort() }
        }
    }

    // ============================================================
    // Multimodal
    // ============================================================

    fun initMultimodal(path: String): Boolean = jni.initMultimodal(path)

    suspend fun generateWithImage(prompt: String, imageData: ByteArray, width: Int, height: Int, params: GenerateParams = GenerateParams()): String {
        return withContext(Dispatchers.IO) {
            jni.generateWithImage(prompt, imageData, width, height, params.maxTokens, params.temperature, params.topP)
        }
    }

    suspend fun generateWithAudio(prompt: String, audioSamples: FloatArray, params: GenerateParams = GenerateParams()): String {
        return withContext(Dispatchers.IO) {
            jni.generateWithAudio(prompt, audioSamples, params.maxTokens, params.temperature, params.topP)
        }
    }

    fun isMultimodalReady(): Boolean = try { jni.getModelInfo().contains("mmproj") } catch (_: Exception) { false }

    // ============================================================
    // LoRA adapters
    // ============================================================

    suspend fun loadLoraAdapter(path: String, scale: Float = 1.0f): Boolean {
        return withContext(Dispatchers.IO) {
            val success = jni.loadLoraAdapter(path, scale)
            if (success) currentLora = File(path).name
            success
        }
    }

    fun unloadLoraAdapter() {
        jni.unloadLoraAdapter()
        currentLora = null
    }

    fun getLoraName(): String? = currentLora

    // ============================================================
    // Chat template (auto-detect from model)
    // ============================================================

    fun applyChatTemplate(messages: List<Pair<String, String>>, toolsJson: String? = null): String {
        val json = JSONArray().apply {
            messages.forEach { (role, content) ->
                put(JSONObject().put("role", role).put("content", content))
            }
        }
        return jni.applyChatTemplate(json.toString(), toolsJson)
    }

    fun applyChatTemplateRaw(messagesJson: String, toolsJson: String? = null): String {
        return jni.applyChatTemplate(messagesJson, toolsJson)
    }

    // ============================================================
    // Model files
    // ============================================================

    fun listModels(): List<ModelFile> {
        return modelDir.listFiles()
            ?.filter { it.extension == "gguf" && it.length() > 1024 }
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
