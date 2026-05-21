package com.aiope.inf

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collect
import org.json.JSONArray
import org.json.JSONObject
import java.io.*
import java.net.ServerSocket
import java.net.Socket
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Embedded HTTP server implementing OpenAI Chat Completions API.
 * Runs on localhost, serves GGUF model inference via llama.cpp JNI.
 *
 * Endpoints:
 *   POST /v1/chat/completions  - Chat completion (streaming & non-streaming)
 *   GET  /v1/models            - List loaded models
 *   GET  /health               - Health check
 */
class OpenAIServer(private val modelManager: ModelManager) {

    private var serverSocket: ServerSocket? = null
    private var serverJob: Job? = null
    private val running = AtomicBoolean(false)
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    fun start(port: Int = 8080) {
        if (running.get()) return

        serverJob = scope.launch {
            serverSocket = ServerSocket(port)
            running.set(true)

            while (running.get()) {
                try {
                    val client = serverSocket?.accept() ?: break
                    launch { handleClient(client) }
                } catch (e: Exception) {
                    if (running.get()) e.printStackTrace()
                }
            }
        }
    }

    fun stop() {
        running.set(false)
        serverSocket?.close()
        serverJob?.cancel()
    }

    private suspend fun handleClient(socket: Socket) {
        socket.use { client ->
            val input = BufferedReader(InputStreamReader(client.getInputStream()))
            val output = BufferedOutputStream(client.getOutputStream())

            try {
                val request = parseRequest(input)
                val response = routeRequest(request)

                if (response.isStreaming && response.streamData == "__LIVE_STREAM__") {
                    // True SSE: write headers then stream tokens directly
                    writeSseHeaders(output)
                    streamTokensToSocket(output, response)
                } else {
                    writeResponse(output, response)
                }
            } catch (e: Exception) {
                val error = errorResponse(500, e.message ?: "Internal error")
                writeResponse(output, error)
            }
        }
    }

    private fun writeSseHeaders(output: OutputStream) {
        val writer = PrintWriter(BufferedWriter(OutputStreamWriter(output)))
        writer.print("HTTP/1.1 200 OK\r\n")
        writer.print("Content-Type: text/event-stream\r\n")
        writer.print("Cache-Control: no-cache\r\n")
        writer.print("Connection: keep-alive\r\n")
        writer.print("Access-Control-Allow-Origin: *\r\n")
        writer.print("Transfer-Encoding: chunked\r\n")
        writer.print("\r\n")
        writer.flush()
    }

    private suspend fun streamTokensToSocket(output: OutputStream, response: HttpResponse) {
        val completionId = "chatcmpl-${java.util.UUID.randomUUID().toString().take(8)}"
        val model = response.streamModel

        modelManager.generateStream(
            response.streamPrompt,
            ModelManager.GenerateParams(
                maxTokens = response.streamMaxTokens,
                temperature = response.streamTemperature,
                topP = response.streamTopP
            )
        ).collect { token ->
            val escaped = token.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r")
            val chunk = """data: {"id":"$completionId","object":"chat.completion.chunk","created":${System.currentTimeMillis()/1000},"model":"$model","choices":[{"index":0,"delta":{"content":"$escaped"},"finish_reason":null}]}"""
            writeChunk(output, chunk + "\n\n")
        }

        // Final messages
        val done = """data: {"id":"$completionId","object":"chat.completion.chunk","created":${System.currentTimeMillis()/1000},"model":"$model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"""
        writeChunk(output, done + "\n\n")
        writeChunk(output, "data: [DONE]\n\n")
        // End chunked encoding
        output.write("0\r\n\r\n".toByteArray())
        output.flush()
    }

    private fun writeChunk(output: OutputStream, data: String) {
        val bytes = data.toByteArray(Charsets.UTF_8)
        val header = "${Integer.toHexString(bytes.size)}\r\n"
        output.write(header.toByteArray())
        output.write(bytes)
        output.write("\r\n".toByteArray())
        output.flush()
    }

    // ============================================================
    // Request parsing
    // ============================================================

    data class HttpRequest(
        val method: String,
        val path: String,
        val headers: Map<String, String>,
        val body: String
    )

    data class HttpResponse(
        val status: Int,
        val statusText: String,
        val contentType: String,
        val body: String,
        val isStreaming: Boolean = false,
        val streamData: String = "",
        val streamPrompt: String = "",
        val streamModel: String = "",
        val streamMaxTokens: Int = 512,
        val streamTemperature: Float = 0.7f,
        val streamTopP: Float = 0.9f
    )

    private fun parseRequest(reader: BufferedReader): HttpRequest {
        val requestLine = reader.readLine() ?: throw IOException("Empty request")
        val parts = requestLine.split(" ")
        val method = parts[0]
        val path = parts[1]

        val headers = mutableMapOf<String, String>()
        var line = reader.readLine()
        while (line != null && line.isNotEmpty()) {
            val colonIdx = line.indexOf(':')
            if (colonIdx > 0) {
                headers[line.substring(0, colonIdx).trim().lowercase()] =
                    line.substring(colonIdx + 1).trim()
            }
            line = reader.readLine()
        }

        var body = ""
        val contentLength = headers["content-length"]?.toIntOrNull() ?: 0
        if (contentLength > 0) {
            val buf = CharArray(contentLength)
            reader.read(buf, 0, contentLength)
            body = String(buf)
        }

        return HttpRequest(method, path, headers, body)
    }

    // ============================================================
    // Routing
    // ============================================================

    private suspend fun routeRequest(request: HttpRequest): HttpResponse {
        return when {
            request.path == "/health" && request.method == "GET" ->
                healthCheck()
            request.path == "/v1/models" && request.method == "GET" ->
                listModels()
            request.path == "/v1/chat/completions" && request.method == "POST" ->
                chatCompletion(request)
            else ->
                errorResponse(404, "Not found: ${request.path}")
        }
    }

    // ============================================================
    // GET /health
    // ============================================================

    private fun healthCheck(): HttpResponse {
        val json = JSONObject().apply {
            put("status", "ok")
            put("model_loaded", modelManager.isLoaded())
            put("gpu", modelManager.getGpuInfo())
        }
        return HttpResponse(200, "OK", "application/json", json.toString())
    }

    // ============================================================
    // GET /v1/models
    // ============================================================

    private fun listModels(): HttpResponse {
        val models = modelManager.listModels()
        val data = JSONArray()
        for (model in models) {
            data.put(JSONObject().apply {
                put("id", model.name)
                put("object", "model")
                put("owned_by", "aiope-inf")
                put("size_mb", model.sizeMB)
            })
        }
        val json = JSONObject().apply {
            put("object", "list")
            put("data", data)
        }
        return HttpResponse(200, "OK", "application/json", json.toString())
    }

    // ============================================================
    // POST /v1/chat/completions
    // ============================================================

    private suspend fun chatCompletion(request: HttpRequest): HttpResponse {
        val body = JSONObject(request.body)
        val messages = body.getJSONArray("messages")
        val stream = body.optBoolean("stream", false)
        val maxTokens = body.optInt("max_tokens", 512)
        val temperature = body.optDouble("temperature", 0.7).toFloat()
        val topP = body.optDouble("top_p", 0.9).toFloat()
        val model = body.optString("model", "local")

        // Build prompt from messages (ChatML format)
        val prompt = buildChatPrompt(messages)

        if (stream) {
            return streamingCompletion(prompt, model, maxTokens, temperature, topP)
        } else {
            return blockingCompletion(prompt, model, maxTokens, temperature, topP)
        }
    }

    private fun buildChatPrompt(messages: JSONArray): String {
        // Use model's native chat template (supports Gemma, ChatML, Llama, etc.)
        val msgList = mutableListOf<Pair<String, String>>()
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            msgList.add(msg.getString("role") to msg.getString("content"))
        }
        val templated = modelManager.applyChatTemplate(msgList)
        if (templated.isNotEmpty()) return templated

        // Fallback: ChatML format
        val sb = StringBuilder()
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            val role = msg.getString("role")
            val content = msg.getString("content")
            sb.append("<|im_start|>$role\n$content<|im_end|>\n")
        }
        sb.append("<|im_start|>assistant\n")
        return sb.toString()
    }

    private suspend fun blockingCompletion(
        prompt: String,
        model: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float
    ): HttpResponse {
        val result = modelManager.generate(prompt, ModelManager.GenerateParams(
            maxTokens = maxTokens,
            temperature = temperature,
            topP = topP
        ))

        val completionId = "chatcmpl-${UUID.randomUUID().toString().take(8)}"
        val json = JSONObject().apply {
            put("id", completionId)
            put("object", "chat.completion")
            put("created", System.currentTimeMillis() / 1000)
            put("model", model)
            put("choices", JSONArray().put(JSONObject().apply {
                put("index", 0)
                put("message", JSONObject().apply {
                    put("role", "assistant")
                    put("content", result)
                })
                put("finish_reason", "stop")
            }))
            put("usage", JSONObject().apply {
                put("prompt_tokens", prompt.length / 4)  // estimate
                put("completion_tokens", result.length / 4)
                put("total_tokens", (prompt.length + result.length) / 4)
            })
        }

        return HttpResponse(200, "OK", "application/json", json.toString())
    }

    private suspend fun streamingCompletion(
        prompt: String,
        model: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float
    ): HttpResponse {
        // Return a special marker — actual streaming handled in handleClient
        return HttpResponse(
            status = 200,
            statusText = "OK",
            contentType = "text/event-stream",
            body = "",
            isStreaming = true,
            streamData = "__LIVE_STREAM__",
            streamPrompt = prompt,
            streamModel = model,
            streamMaxTokens = maxTokens,
            streamTemperature = temperature,
            streamTopP = topP
        )
    }

    // ============================================================
    // Response writing
    // ============================================================

    private fun writeResponse(output: OutputStream, response: HttpResponse) {
        val writer = PrintWriter(BufferedWriter(OutputStreamWriter(output)))

        val body = if (response.isStreaming) response.streamData else response.body
        val bodyBytes = body.toByteArray(Charsets.UTF_8)

        writer.print("HTTP/1.1 ${response.status} ${response.statusText}\r\n")
        writer.print("Content-Type: ${response.contentType}\r\n")
        writer.print("Content-Length: ${bodyBytes.size}\r\n")
        writer.print("Access-Control-Allow-Origin: *\r\n")
        writer.print("Access-Control-Allow-Headers: Content-Type, Authorization\r\n")
        writer.print("Connection: close\r\n")

        if (response.isStreaming) {
            writer.print("Cache-Control: no-cache\r\n")
        }

        writer.print("\r\n")
        writer.flush()

        output.write(bodyBytes)
        output.flush()
    }

    private fun errorResponse(status: Int, message: String): HttpResponse {
        val json = JSONObject().apply {
            put("error", JSONObject().apply {
                put("message", message)
                put("type", "server_error")
                put("code", status)
            })
        }
        val statusText = when (status) {
            400 -> "Bad Request"
            404 -> "Not Found"
            500 -> "Internal Server Error"
            else -> "Error"
        }
        return HttpResponse(status, statusText, "application/json", json.toString())
    }
}
