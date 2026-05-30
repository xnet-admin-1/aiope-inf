package com.aiope.inf

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var modelManager: ModelManager
    private var server: OpenAIServer? = null

    private val modelPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { importModel(it) }
    }

    private val loraPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { importLora(it) }
    }

    private val mmprojPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { importMmproj(it) }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { _ -> requestBatteryOptimizationExemption() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        modelManager = ModelManager.getInstance(this)
        requestAllPermissions()
        setupServer()
        setupModel()
        setupMmproj()
        setupLora()
        setupTest()
        updateSystemInfo()
        autoLoadLastModel()
    }

    private fun requestAllPermissions() {
        val needed = mutableListOf<String>()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            needed.add(Manifest.permission.POST_NOTIFICATIONS)
            needed.add(Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            needed.add(Manifest.permission.READ_EXTERNAL_STORAGE)
            needed.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }

        val ungranted = needed.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (ungranted.isNotEmpty()) {
            permissionLauncher.launch(ungranted.toTypedArray())
        } else {
            requestBatteryOptimizationExemption()
        }
    }

    private fun requestBatteryOptimizationExemption() {
        val pm = getSystemService(POWER_SERVICE) as android.os.PowerManager
        if (!pm.isIgnoringBatteryOptimizations(packageName)) {
            val intent = Intent(android.provider.Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                data = Uri.parse("package:$packageName")
            }
            startActivity(intent)
        }
    }

    // ============================================================
    // Server
    // ============================================================

    private fun setupServer() {
        val switch = findViewById<MaterialSwitch>(R.id.switch_server)
        val status = findViewById<TextView>(R.id.tv_server_status)
        val info = findViewById<TextView>(R.id.tv_server_info)
        val indicator = findViewById<View>(R.id.server_indicator)

        switch.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                // Start as foreground service for persistence
                val intent = Intent(this, InferenceService::class.java).apply {
                    action = InferenceService.ACTION_START_SERVER
                    putExtra(InferenceService.EXTRA_PORT, 8008)
                }
                startForegroundService(intent)
                status.text = "Server running"
                info.text = "http://127.0.0.1:8008/v1/"
                indicator.setBackgroundResource(R.drawable.circle_green)
            } else {
                val intent = Intent(this, InferenceService::class.java).apply {
                    action = InferenceService.ACTION_STOP_SERVER
                }
                startService(intent)
                status.text = "Server stopped"
                indicator.setBackgroundResource(R.drawable.circle_red)
            }
        }
    }

    // ============================================================
    // Model
    // ============================================================

    private fun setupModel() {
        val btnLoad = findViewById<MaterialButton>(R.id.btn_load_model)
        val btnUnload = findViewById<MaterialButton>(R.id.btn_unload_model)
        val tvName = findViewById<TextView>(R.id.tv_model_name)
        val tvInfo = findViewById<TextView>(R.id.tv_model_info)
        val cardLora = findViewById<MaterialCardView>(R.id.card_lora)
        val cardMmproj = findViewById<MaterialCardView>(R.id.card_mmproj)

        btnLoad.setOnClickListener {
            val models = modelManager.listModels()
            val options = models.map { "${it.name} (${it.sizeMB}MB)" }.toMutableList()
            options.add("📁 Import from file...")

            AlertDialog.Builder(this)
                .setTitle("Select Model")
                .setItems(options.toTypedArray()) { _, which ->
                    if (which < models.size) {
                        loadModel(models[which].path)
                    } else {
                        modelPicker.launch(arrayOf("*/*"))
                    }
                }
                .show()
        }

        btnUnload.setOnClickListener {
            modelManager.unload()
            tvName.text = "No model loaded"
            tvInfo.text = "Tap to select a model"
            btnUnload.visibility = View.GONE
        }

        // Restore UI if model already loaded in native
        if (modelManager.isLoaded()) {
            try {
                val info = modelManager.getModelInfo()
                tvName.text = info.optString("description", "Model loaded")
                val params = info.optLong("n_params", 0) / 1_000_000
                tvInfo.text = "${params}M params · GPU active"
                btnUnload.visibility = View.VISIBLE
                cardLora.visibility = View.VISIBLE
                findViewById<com.google.android.material.card.MaterialCardView>(R.id.card_mmproj).visibility = View.VISIBLE
            } catch (_: Exception) {}
        }
    }

    private fun importModel(uri: Uri) {
        val tvInfo = findViewById<TextView>(R.id.tv_model_info)
        tvInfo.text = "Importing model..."

        lifecycleScope.launch(Dispatchers.IO) {
            val filename = contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (cursor.moveToFirst() && idx >= 0) cursor.getString(idx) else null
            } ?: uri.lastPathSegment?.substringAfterLast("/") ?: "model.gguf"
            val dest = File(filesDir, "models/$filename")
            dest.parentFile?.mkdirs()

            contentResolver.openInputStream(uri)?.use { input ->
                dest.outputStream().use { output -> input.copyTo(output, 8192 * 1024) }
            }

            withContext(Dispatchers.Main) {
                tvInfo.text = "Imported: $filename (${dest.length() / (1024*1024)}MB)"
                Toast.makeText(this@MainActivity, "Model imported", Toast.LENGTH_SHORT).show()
                loadModel(dest.absolutePath)
            }
        }
    }

    private fun importLora(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            val filename = contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (cursor.moveToFirst() && idx >= 0) cursor.getString(idx) else null
            } ?: uri.lastPathSegment?.substringAfterLast("/") ?: "adapter.gguf"
            val dest = File(filesDir, "lora/$filename")
            dest.parentFile?.mkdirs()

            contentResolver.openInputStream(uri)?.use { input ->
                dest.outputStream().use { output -> input.copyTo(output, 8192 * 1024) }
            }

            withContext(Dispatchers.Main) {
                Toast.makeText(this@MainActivity, "Adapter imported: $filename", Toast.LENGTH_SHORT).show()
                val tvLora = findViewById<TextView>(R.id.tv_lora_name)
                val success = modelManager.loadLoraAdapter(dest.absolutePath)
                tvLora.text = if (success) "✓ $filename" else "Failed to load"
                if (success) getPreferences(MODE_PRIVATE).edit().putString("last_lora", dest.absolutePath).apply()
            }
        }
    }

    private fun loadModel(path: String) {
        val tvName = findViewById<TextView>(R.id.tv_model_name)
        val tvInfo = findViewById<TextView>(R.id.tv_model_info)
        val btnUnload = findViewById<MaterialButton>(R.id.btn_unload_model)
        val cardLora = findViewById<MaterialCardView>(R.id.card_lora)
        val progress = findViewById<LinearProgressIndicator>(R.id.progress_bar)

        progress.visibility = View.VISIBLE
        tvInfo.text = "Loading..."

        lifecycleScope.launch {
            try {
                val success = modelManager.loadModel(path, ModelManager.LoadConfig(
                    useVulkan = true,
                    autoGpuLayers = false,
                    gpuLayers = 99,
                    contextSize = 4096
                ))
                progress.visibility = View.GONE

                if (success) {
                    val info = modelManager.getModelInfo()
                    tvName.text = info.optString("description", path.substringAfterLast("/"))
                    val params = info.optLong("n_params", 0) / 1_000_000
                    val size = info.optLong("size", 0) / (1024 * 1024)
                    tvInfo.text = "${params}M params · ${size}MB · GPU layers: ${info.optInt("gpu_layers")}"
                    btnUnload.visibility = View.VISIBLE
                    cardLora.visibility = View.VISIBLE
                    findViewById<com.google.android.material.card.MaterialCardView>(R.id.card_mmproj).visibility = View.VISIBLE
                    getPreferences(MODE_PRIVATE).edit().putString("last_model", path).apply()
                    autoLoadMmproj()
                    autoLoadLora()
                } else {
                    tvName.text = "Load failed"
                    tvInfo.text = "jni.loadModel returned false for: ${path.substringAfterLast("/")}"
                }
            } catch (e: Exception) {
                progress.visibility = View.GONE
                tvName.text = "Crash during load"
                tvInfo.text = e.message ?: "Unknown error"
                android.util.Log.e("AIOPE", "Model load crashed", e)
            }
        }
    }

    // ============================================================
    // Multimodal Projector
    // ============================================================

    private fun setupMmproj() {
        val btnMmproj = findViewById<MaterialButton>(R.id.btn_load_mmproj)
        val tvMmproj = findViewById<TextView>(R.id.tv_mmproj_name)

        btnMmproj.setOnClickListener {
            val mmprojDir = File(filesDir, "mmproj")
            mmprojDir.mkdirs()
            val projectors = mmprojDir.listFiles()?.filter { it.extension == "gguf" } ?: emptyList()

            val options = projectors.map { it.name }.toMutableList()
            options.add("📁 Import from file...")

            AlertDialog.Builder(this)
                .setTitle("Select Multimodal Projector")
                .setItems(options.toTypedArray()) { _, which ->
                    if (which < projectors.size) {
                        lifecycleScope.launch {
                            val success = withContext(Dispatchers.IO) { modelManager.initMultimodal(projectors[which].absolutePath) }
                            tvMmproj.text = if (success) "✓ ${projectors[which].name}" else "Failed to load"
                            if (success) getPreferences(MODE_PRIVATE).edit().putString("last_mmproj", projectors[which].absolutePath).apply()
                        }
                    } else {
                        mmprojPicker.launch(arrayOf("*/*"))
                    }
                }
                .show()
        }
    }

    private fun importMmproj(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            val filename = contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (cursor.moveToFirst() && idx >= 0) cursor.getString(idx) else null
            } ?: uri.lastPathSegment?.substringAfterLast("/") ?: "mmproj.gguf"
            val dest = File(filesDir, "mmproj/$filename")
            dest.parentFile?.mkdirs()

            contentResolver.openInputStream(uri)?.use { input ->
                dest.outputStream().use { output -> input.copyTo(output, 8192 * 1024) }
            }

            val success = modelManager.initMultimodal(dest.absolutePath)
            withContext(Dispatchers.Main) {
                val tvMmproj = findViewById<TextView>(R.id.tv_mmproj_name)
                tvMmproj.text = if (success) "✓ $filename" else "Failed to load"
                if (success) getPreferences(MODE_PRIVATE).edit().putString("last_mmproj", dest.absolutePath).apply()
                Toast.makeText(this@MainActivity, if (success) "Projector loaded" else "Failed to load projector", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun autoLoadMmproj() {
        val path = getPreferences(MODE_PRIVATE).getString("last_mmproj", null) ?: return
        if (!File(path).exists()) return
        val tvMmproj = findViewById<TextView>(R.id.tv_mmproj_name)
        lifecycleScope.launch {
            val success = withContext(Dispatchers.IO) { modelManager.initMultimodal(path) }
            tvMmproj.text = if (success) "✓ ${File(path).name}" else ""
        }
    }

    // ============================================================
    // LoRA
    // ============================================================

    private fun setupLora() {
        val btnLora = findViewById<MaterialButton>(R.id.btn_load_lora)
        val tvLora = findViewById<TextView>(R.id.tv_lora_name)

        // Show current lora if already loaded
        modelManager.getLoraName()?.let { tvLora.text = "✓ $it" }

        btnLora.setOnClickListener {
            val loraDir = File(filesDir, "lora")
            loraDir.mkdirs()
            val adapters = loraDir.listFiles()?.filter { it.extension == "gguf" } ?: emptyList()

            val options = adapters.map { it.name }.toMutableList()
            options.add("📁 Import from file...")

            AlertDialog.Builder(this)
                .setTitle("Select LoRA Adapter")
                .setItems(options.toTypedArray()) { _, which ->
                    if (which < adapters.size) {
                        lifecycleScope.launch {
                            val success = modelManager.loadLoraAdapter(adapters[which].absolutePath)
                            tvLora.text = if (success) "✓ ${adapters[which].name}" else "Failed to load"
                            if (success) getPreferences(MODE_PRIVATE).edit().putString("last_lora", adapters[which].absolutePath).apply()
                        }
                    } else {
                        loraPicker.launch(arrayOf("*/*"))
                    }
                }
                .show()
        }
    }

    // ============================================================
    // Quick Test
    // ============================================================

    private fun setupTest() {
        val btnSend = findViewById<MaterialButton>(R.id.btn_send)
        val tvOutput = findViewById<TextView>(R.id.tv_output)

        btnSend.text = "Test"
        btnSend.setOnClickListener {
            tvOutput.visibility = View.VISIBLE
            tvOutput.text = "Testing..."
            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val body = """{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":false}"""
                    val req = java.net.URL("http://127.0.0.1:8008/v1/chat/completions").openConnection() as java.net.HttpURLConnection
                    req.requestMethod = "POST"
                    req.setRequestProperty("Content-Type", "application/json")
                    req.doOutput = true
                    req.connectTimeout = 5000
                    req.readTimeout = 30000
                    req.outputStream.write(body.toByteArray())
                    val code = req.responseCode
                    req.inputStream.close()
                    req.disconnect()
                    withContext(Dispatchers.Main) { tvOutput.text = "✓ $code OK" }
                } catch (e: java.net.ConnectException) {
                    withContext(Dispatchers.Main) { tvOutput.text = "❌ Server not running — toggle it on first" }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) { tvOutput.text = "❌ ${e.javaClass.simpleName}: ${e.message}" }
                }
            }
        }
    }

    // ============================================================
    // System Info
    // ============================================================

    private fun autoLoadLastModel() {
        val path = getPreferences(MODE_PRIVATE).getString("last_model", null) ?: return
        if (!File(path).exists()) return
        loadModel(path)
    }

    private fun autoLoadLora() {
        val path = getPreferences(MODE_PRIVATE).getString("last_lora", null) ?: return
        if (!File(path).exists()) return
        val tvLora = findViewById<TextView>(R.id.tv_lora_name)
        lifecycleScope.launch {
            val success = modelManager.loadLoraAdapter(path)
            tvLora.text = if (success) "✓ ${File(path).name}" else ""
        }
    }

    private fun updateSystemInfo() {
        val tvGpu = findViewById<TextView>(R.id.tv_gpu_info)
        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                try {
                    // Must init GPU before querying info
                    val jni = LlamaJNI()
                    jni.initGpu()
                    val gpu = modelManager.getGpuInfo()
                    val backend = modelManager.getBackendInfo()
                    withContext(Dispatchers.Main) {
                        val vulkan = if (gpu.optBoolean("available")) "✓" else "✗"
                        tvGpu.text = "GPU: ${gpu.optString("device_name", "N/A")} · Vulkan: $vulkan\n" +
                            "VRAM: ${gpu.optInt("vram_mb")}MB · Threads: ${backend.optInt("cpu_threads")}\n" +
                            "Models: ${modelManager.listModels().size} available"
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        tvGpu.text = "Threads: ${Runtime.getRuntime().availableProcessors()}"
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        server?.stop()
        modelManager.unload()
        super.onDestroy()
    }
}
