package com.aiope.inf

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

class MainActivity : AppCompatActivity() {

    private lateinit var modelManager: ModelManager
    private lateinit var apiServer: OpenAIServer

    private lateinit var tvStatus: TextView
    private lateinit var tvOutput: TextView
    private lateinit var etPrompt: EditText
    private lateinit var btnLoad: Button
    private lateinit var btnGenerate: Button
    private lateinit var btnStream: Button
    private lateinit var btnStartServer: Button
    private lateinit var btnStop: Button
    private lateinit var progressBar: ProgressBar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        modelManager = ModelManager(this)
        apiServer = OpenAIServer(modelManager)

        bindViews()
        setupListeners()
        updateStatus()
    }

    private fun bindViews() {
        tvStatus = findViewById(R.id.tv_status)
        tvOutput = findViewById(R.id.tv_output)
        etPrompt = findViewById(R.id.et_prompt)
        btnLoad = findViewById(R.id.btn_load)
        btnGenerate = findViewById(R.id.btn_generate)
        btnStream = findViewById(R.id.btn_stream)
        btnStartServer = findViewById(R.id.btn_start_server)
        btnStop = findViewById(R.id.btn_stop)
        progressBar = findViewById(R.id.progress_bar)
    }

    private fun setupListeners() {
        btnLoad.setOnClickListener { loadModel() }
        btnGenerate.setOnClickListener { generate() }
        btnStream.setOnClickListener { streamGenerate() }
        btnStartServer.setOnClickListener { startServer() }
        btnStop.setOnClickListener { stopGeneration() }
    }

    private fun loadModel() {
        val models = modelManager.listModels()
        if (models.isEmpty()) {
            tvStatus.text = "No models found in ${filesDir}/models/"
            return
        }

        progressBar.visibility = android.view.View.VISIBLE
        lifecycleScope.launch {
            val model = models.first()
            val config = ModelManager.LoadConfig(
                useVulkan = true,
                autoGpuLayers = true,
                contextSize = 2048
            )
            val success = modelManager.loadModel(model.path, config)
            progressBar.visibility = android.view.View.GONE

            if (success) {
                updateStatus()
                Toast.makeText(this@MainActivity, "Model loaded: ${model.name}", Toast.LENGTH_SHORT).show()
            } else {
                tvStatus.text = "Failed to load model"
            }
        }
    }

    private fun generate() {
        val prompt = etPrompt.text.toString().trim()
        if (prompt.isEmpty()) return

        tvOutput.text = ""
        progressBar.visibility = android.view.View.VISIBLE

        lifecycleScope.launch {
            val result = modelManager.generate(prompt, ModelManager.GenerateParams(
                maxTokens = 256,
                temperature = 0.7f
            ))
            progressBar.visibility = android.view.View.GONE
            tvOutput.text = result
        }
    }

    private fun streamGenerate() {
        val prompt = etPrompt.text.toString().trim()
        if (prompt.isEmpty()) return

        tvOutput.text = ""
        progressBar.visibility = android.view.View.VISIBLE

        lifecycleScope.launch {
            modelManager.generateStream(prompt, ModelManager.GenerateParams(
                maxTokens = 256,
                temperature = 0.7f
            )).collect { token ->
                withContext(Dispatchers.Main) {
                    tvOutput.append(token)
                }
            }
            progressBar.visibility = android.view.View.GONE
        }
    }

    private fun startServer() {
        lifecycleScope.launch {
            apiServer.start(port = 8080)
            tvStatus.text = "OpenAI API server running on :8080"
            Toast.makeText(this@MainActivity, "Server started on port 8080", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopGeneration() {
        modelManager.abort()
        progressBar.visibility = android.view.View.GONE
    }

    private fun updateStatus() {
        val gpu = modelManager.getGpuInfo()
        val backend = modelManager.getBackendInfo()
        val models = modelManager.listModels()

        val sb = StringBuilder()
        sb.appendLine("GPU: ${gpu.optString("device_name", "N/A")} | Vulkan: ${gpu.optBoolean("available")}")
        sb.appendLine("VRAM: ${gpu.optInt("vram_mb")} MB | Threads: ${backend.optInt("cpu_threads")}")
        sb.appendLine("Models: ${models.size} (${models.sumOf { it.sizeMB }} MB total)")
        if (modelManager.isLoaded()) {
            val info = modelManager.getModelInfo()
            sb.appendLine("Loaded: ${info.optString("description")}")
        }
        tvStatus.text = sb.toString()
    }

    override fun onDestroy() {
        apiServer.stop()
        modelManager.unload()
        super.onDestroy()
    }
}
