package com.aiope.inf

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.withContext
import com.google.ai.edge.litertlm.*

class LiteRTEngine(private val context: Context) {

    private var engine: Engine? = null
    private var conversation: Conversation? = null

    val isLoaded: Boolean get() = engine != null

    suspend fun load(
        modelPath: String,
        backend: AcceleratorType = AcceleratorType.GPU,
        maxTokens: Int = 4096
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            close()
            val backendConfig = when (backend) {
                AcceleratorType.GPU -> Backend.GPU()
                AcceleratorType.NPU -> Backend.NPU(context.applicationInfo.nativeLibraryDir)
                else -> Backend.CPU()
            }
            engine = Engine(EngineConfig(
                modelPath = modelPath,
                backend = backendConfig,
                maxNumTokens = maxTokens,
                cacheDir = context.cacheDir.resolve("litert_cache").apply { mkdirs() }.absolutePath
            ))
            engine!!.initialize()
            true
        } catch (e: Exception) {
            android.util.Log.e("LiteRTEngine", "Load failed", e)
            engine = null
            false
        }
    }

    fun createConversation(
        systemPrompt: String? = null,
        temperature: Double = 0.7,
        topK: Int = 40,
        topP: Double = 0.9
    ) {
        conversation?.close()
        conversation = engine?.createConversation(ConversationConfig(
            systemInstruction = systemPrompt?.let { Contents.of(it) },
            samplerConfig = SamplerConfig(topK = topK, topP = topP, temperature = temperature)
        ))
    }

    fun generateStream(prompt: String): Flow<String> = callbackFlow {
        val conv = conversation ?: run { close(); return@callbackFlow }
        conv.sendMessageAsync(
            Message.user(prompt),
            object : MessageCallback {
                override fun onMessage(message: Message) {
                    message.contents?.contents?.filterIsInstance<Content.Text>()?.forEach {
                        trySend(it.text)
                    }
                }
                override fun onDone() { close() }
                override fun onError(e: Throwable) { close(Exception(e)) }
            }
        )
        awaitClose { conv.cancelProcess() }
    }

    suspend fun generate(prompt: String): String = withContext(Dispatchers.IO) {
        val conv = conversation ?: return@withContext ""
        val response = conv.sendMessage(prompt)
        response.contents?.contents?.filterIsInstance<Content.Text>()?.joinToString("") { it.text } ?: ""
    }

    fun stop() { conversation?.cancelProcess() }

    fun close() {
        conversation?.close()
        conversation = null
        engine?.close()
        engine = null
    }

    enum class AcceleratorType { CPU, GPU, NPU }
}
