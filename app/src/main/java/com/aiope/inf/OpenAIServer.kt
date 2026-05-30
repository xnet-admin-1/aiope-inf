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
    private var serverThread: Thread? = null
    private val running = AtomicBoolean(false)
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    fun start(port: Int = 8008) {
        if (running.get()) return
        running.set(true)

        serverThread = Thread({
            try {
                serverSocket = ServerSocket(port, 50, java.net.InetAddress.getByName("127.0.0.1"))
                android.util.Log.i("GATEWAY", "Server listening on port $port")
                while (running.get()) {
                    try {
                        val client = serverSocket?.accept() ?: break
                        client.soTimeout = 300000
                        Thread { handleClientSync(client) }.start()
                    } catch (e: Exception) {
                        if (running.get()) android.util.Log.e("GATEWAY", "accept error: ${e.message}")
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("GATEWAY", "Server start failed: ${e.message}")
            }
        }, "aiope-server").also { it.isDaemon = true; it.start() }
    }

    fun stop() {
        running.set(false)
        serverSocket?.close()
        serverThread?.interrupt()
    }

    private fun handleClientSync(socket: Socket) {
        socket.use { client ->
            val input = BufferedReader(InputStreamReader(client.getInputStream()))
            val output = BufferedOutputStream(client.getOutputStream())

            try {
                val request = parseRequest(input)
                android.util.Log.i("GATEWAY", "${request.method} ${request.path} body=${request.body.take(200)}")
                val response = routeRequestSync(request)
                android.util.Log.i("GATEWAY", "→ ${response.status} streaming=${response.isStreaming}")

                if (response.isStreaming && response.streamData == "__LIVE_STREAM__") {
                    val rawOutput = client.getOutputStream()
                    writeSseHeaders(rawOutput)
                    android.util.Log.i("GATEWAY", "SSE headers sent, starting generation")
                    streamTokensSync(rawOutput, response)
                    android.util.Log.i("GATEWAY", "SSE stream complete")
                } else {
                    writeResponse(output, response)
                }
            } catch (e: Exception) {
                android.util.Log.e("GATEWAY", "handleClient error: ${e.message}")
                try {
                    val error = errorResponse(500, e.message ?: "Internal error")
                    writeResponse(output, error)
                } catch (_: Exception) {}
            }
        }
    }

    private fun routeRequestSync(request: HttpRequest): HttpResponse {
        return when {
            request.method == "OPTIONS" -> HttpResponse(204, "No Content", "text/plain", "")
            request.path == "/health" && request.method == "GET" -> healthCheck()
            request.path == "/v1/models" && request.method == "GET" -> listModels()
            request.path == "/v1/chat/completions" && request.method == "POST" -> chatCompletionSync(request)
            else -> errorResponse(404, "Not found: ${request.path}")
        }
    }

    private fun chatCompletionSync(request: HttpRequest): HttpResponse {
        val body = JSONObject(request.body)
        val messages = body.getJSONArray("messages")
        val stream = body.optBoolean("stream", false)
        val maxTokens = body.optInt("max_tokens", 512)
        val temperature = body.optDouble("temperature", 0.7).toFloat()
        val topP = body.optDouble("top_p", 0.9).toFloat()
        val model = body.optString("model", "local")
        val tools = body.optJSONArray("tools")

        val prompt = buildChatPrompt(messages, tools)
        android.util.Log.i("GATEWAY", "chatCompletion stream=$stream prompt_len=${prompt.length}")
        android.util.Log.d("GATEWAY", "prompt_start=${prompt.take(300)}")

        if (stream) {
            return streamingCompletion(prompt, model, maxTokens, temperature, topP)
        }

        // Blocking generation
        val jni = LlamaJNI()
        val result = jni.generate(prompt, maxTokens, temperature, topP, 1.1f)
        return buildCompletionResponse(result, model)
    }

    private fun streamTokensSync(output: OutputStream, response: HttpResponse) {
        val completionId = "chatcmpl-${java.util.UUID.randomUUID().toString().take(8)}"
        val model = response.streamModel
        var tokenCount = 0

        android.util.Log.i("GATEWAY", "generateStreaming starting")
        LlamaJNI().generateStreaming(
            prompt = response.streamPrompt,
            maxTokens = response.streamMaxTokens,
            temperature = response.streamTemperature,
            topP = response.streamTopP,
            repeatPenalty = 1.1f,
            callback = object : LlamaJNI.StreamCallback {
                override fun onToken(token: String): Boolean {
                    tokenCount++
                    if (tokenCount <= 3) android.util.Log.i("GATEWAY", "token[$tokenCount]: ${token.take(20)}")
                    try {
                        val escaped = token.replace("\\", "\\\\").replace("\"", "\\\"")
                            .replace("\n", "\\n").replace("\r", "\\r")
                        val chunk = """data: {"id":"$completionId","object":"chat.completion.chunk","created":${System.currentTimeMillis()/1000},"model":"$model","choices":[{"index":0,"delta":{"content":"$escaped"},"finish_reason":null}]}"""
                        writeChunk(output, chunk + "\n\n")
                        return true
                    } catch (e: Exception) {
                        android.util.Log.e("GATEWAY", "write failed: ${e.message}")
                        return false
                    }
                }
                override fun onComplete(fullResponse: String) {
                    android.util.Log.i("GATEWAY", "complete, tokens=$tokenCount")
                }
                override fun onError(error: String) {
                    android.util.Log.e("GATEWAY", "gen error: $error")
                    try { writeChunk(output, """data: {"error":"$error"}""" + "\n\n") } catch (_: Exception) {}
                }
            }
        )

        val done = """data: {"id":"$completionId","object":"chat.completion.chunk","created":${System.currentTimeMillis()/1000},"model":"$model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"""
        writeChunk(output, done + "\n\n")
        writeChunk(output, "data: [DONE]\n\n")
        output.flush()
    }

    private fun writeSseHeaders(output: OutputStream) {
        val headers = "HTTP/1.1 200 OK\r\n" +
            "Content-Type: text/event-stream\r\n" +
            "Cache-Control: no-cache\r\n" +
            "Connection: keep-alive\r\n" +
            "Access-Control-Allow-Origin: *\r\n" +
            "\r\n"
        output.write(headers.toByteArray())
        output.flush()
    }


    private fun writeChunk(output: OutputStream, data: String) {
        output.write(data.toByteArray(Charsets.UTF_8))
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

    private fun audioTranscription(request: HttpRequest): HttpResponse {
        try {
            val audioBytes = request.body.toByteArray(Charsets.ISO_8859_1)
            val samples = FloatArray(audioBytes.size / 2)
            for (i in samples.indices) {
                val lo = audioBytes[i * 2].toInt() and 0xFF
                val hi = audioBytes[i * 2 + 1].toInt()
                samples[i] = ((hi shl 8) or lo).toFloat() / 32768f
            }
            val prompt = "Transcribe the following speech segment into text. Only output the transcription, with no newlines."
            val result = kotlinx.coroutines.runBlocking {
                modelManager.generateWithAudio(prompt, samples, ModelManager.GenerateParams(maxTokens = 512, temperature = 0.1f))
            }
            val json = JSONObject().apply { put("text", result.trim()) }
            return HttpResponse(200, "OK", "application/json", json.toString())
        } catch (e: Exception) {
            return errorResponse(500, "Audio transcription failed: ${e.message}")
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

    private fun extractImageData(messages: JSONArray): Triple<ByteArray, Int, Int>? {
        for (i in messages.length() - 1 downTo 0) {
            val msg = messages.getJSONObject(i)
            val content = msg.opt("content")
            if (content is JSONArray) {
                for (j in 0 until content.length()) {
                    val part = content.getJSONObject(j)
                    if (part.optString("type") == "image_url") {
                        val url = part.optJSONObject("image_url")?.optString("url") ?: continue
                        return decodeImageUrl(url)
                    }
                }
            }
        }
        return null
    }

    private fun decodeImageUrl(url: String): Triple<ByteArray, Int, Int>? {
        val bytes = if (url.startsWith("data:")) {
            // base64 data URI
            val b64 = url.substringAfter("base64,")
            android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
        } else {
            // HTTP URL - fetch it
            try {
                java.net.URL(url).readBytes()
            } catch (_: Exception) { return null }
        }
        // Decode to get dimensions
        val opts = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
        android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
        val w = if (opts.outWidth > 0) opts.outWidth else 512
        val h = if (opts.outHeight > 0) opts.outHeight else 512
        return Triple(bytes, w, h)
    }

    private fun buildCompletionResponse(result: String, model: String): HttpResponse {
        val toolCalls = parseGemmaToolCalls(result)
        val content = if (toolCalls.isNotEmpty()) {
            result.replace(Regex("""<\|tool_call>.*?<tool_call\|>"""), "").trim()
        } else result

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
                    if (content.isNotBlank()) put("content", content)
                    if (toolCalls.isNotEmpty()) {
                        put("tool_calls", JSONArray().apply {
                            toolCalls.forEachIndexed { i, (name, args) ->
                                put(JSONObject().apply {
                                    put("id", "call_${UUID.randomUUID().toString().take(8)}")
                                    put("type", "function")
                                    put("function", JSONObject().apply {
                                        put("name", name)
                                        put("arguments", args)
                                    })
                                })
                            }
                        })
                    }
                })
                put("finish_reason", if (toolCalls.isNotEmpty()) "tool_calls" else "stop")
            }))
            put("usage", JSONObject().apply {
                put("prompt_tokens", 0)
                put("completion_tokens", result.length / 4)
                put("total_tokens", result.length / 4)
            })
        }
        return HttpResponse(200, "OK", "application/json", json.toString())
    }

    /** Parse Gemma 4 native tool call format: <|tool_call>call:func_name{key:<|"|>val<|"|>}<tool_call|> */
    private fun parseGemmaToolCalls(text: String): List<Pair<String, String>> {
        val results = mutableListOf<Pair<String, String>>()
        val pattern = Regex("""<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>""", RegexOption.DOT_MATCHES_ALL)
        for (m in pattern.findAll(text)) {
            val name = m.groupValues[1]
            val rawArgs = m.groupValues[2]
            // Parse Gemma's key:<|"|>value<|"|> format into JSON
            val args = JSONObject()
            val argPattern = Regex("""(\w+):<\|"\|>(.*?)<\|"\|>""")
            val numPattern = Regex("""(\w+):(\d+(?:\.\d+)?)""")
            for (am in argPattern.findAll(rawArgs)) {
                args.put(am.groupValues[1], am.groupValues[2])
            }
            for (nm in numPattern.findAll(rawArgs)) {
                if (!args.has(nm.groupValues[1])) {
                    val v = nm.groupValues[2]
                    if (v.contains(".")) args.put(nm.groupValues[1], v.toDouble())
                    else args.put(nm.groupValues[1], v.toInt())
                }
            }
            results.add(name to args.toString())
        }
        return results
    }

    private fun buildChatPrompt(messages: JSONArray, tools: JSONArray? = null): String {
        // Pass full messages JSON + tools directly to native Jinja template
        val toolsStr = tools?.toString()
        val prompt = modelManager.applyChatTemplateRaw(messages.toString(), toolsStr)
        if (prompt.isNotEmpty()) return prompt

        // Fallback: ChatML format (no tools support)
        val sb = StringBuilder()
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            val role = msg.optString("role", "user")
            val content = when {
                msg.opt("content") is String -> msg.getString("content")
                msg.opt("content") is JSONArray -> {
                    val parts = msg.getJSONArray("content")
                    (0 until parts.length()).mapNotNull { j ->
                        val p = parts.getJSONObject(j)
                        if (p.optString("type") == "text") p.optString("text") else null
                    }.joinToString("\n")
                }
                else -> msg.optString("content", "")
            }
            sb.append("<|im_start|>$role\n$content<|im_end|>\n")
        }
        sb.append("<|im_start|>assistant\n")
        return sb.toString()
    }

    private fun blockingCompletion(
        prompt: String,
        model: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float
    ): HttpResponse {
        val result = LlamaJNI().generate(prompt, maxTokens, temperature, topP, 1.1f)
        return buildCompletionResponse(result, model)
    }

    private fun streamingCompletion(
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
        writer.print("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n")
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
