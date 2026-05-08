package com.aiope.inf

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    
    private lateinit var btnLoadModel: Button
    private lateinit var btnGenerate: Button
    private lateinit var btnChatAPI: Button
    private lateinit var etPrompt: EditText
    private lateinit var tvResponse: TextView
    private lateinit var tvModelStatus: TextView
    private lateinit var tvError: TextView
    private lateinit var progressBar: ProgressBar
    
    private val modelManager = ModelManager(this)
    private val viewModel = LlamaViewModel(application)
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize views
        btnLoadModel = findViewById(R.id.btnLoadModel)
        btnGenerate = findViewById(R.id.btnGenerate)
        btnChatAPI = findViewById(R.id.btnChatAPI)
        etPrompt = findViewById(R.id.etPrompt)
        tvResponse = findViewById(R.id.tvResponse)
        tvModelStatus = findViewById(R.id.tvModelStatus)
        tvError = findViewById(R.id.tvError)
        progressBar = findViewById(R.id.progressBar)
        
        // Load a sample model (you should have a GGUF file in models/
        btnLoadModel.setOnClickListener {
            loadModel("model.q4_0.gguf")
        }
        
        // Generate text
        btnGenerate.setOnClickListener {
            val prompt = etPrompt.text.toString()
            if (prompt.isNotEmpty()) {
                generateText(prompt)
            } else {
                Toast.makeText(this, "Enter a prompt first", Toast.LENGTH_SHORT).show()
            }
        }
        
        // Test OpenAI-like API
        btnChatAPI.setOnClickListener {
            testOpenAIApi()
        }
        
        // Observe LiveData from ViewModel
        viewModel.isModelLoaded.observe(this) { isLoaded ->
            tvModelStatus.text = if (isLoaded) "Model: Loaded ✓" else "Model: Not Loaded"
            btnGenerate.isEnabled = isLoaded
            btnChatAPI.isEnabled = isLoaded
        }
        
        viewModel.generatedText.observe(this) { text ->
            tvResponse.text = text
            progressBar.visibility = ProgressBar.GONE
        }
        
        viewModel.errorMessage.observe(this) { error ->
            tvError.text = error
            tvError.visibility = if (error.isNotEmpty()) TextView.VISIBLE else TextView.GONE
            progressBar.visibility = ProgressBar.GONE
        }
    }
    
    private fun loadModel(filename: String) {
        progressBar.visibility = ProgressBar.VISIBLE
        
        lifecycleScope.launch {
            try {
                // Check if model exists
                val modelFile = File(filesDir, "models/$filename")
                if (!modelFile.exists()) {
                    Toast.makeText(
                        this@MainActivity,
                        "Model file not found. Please place GGUF model in models/ directory",
                        Toast.LENGTH_LONG
                    ).show()
                } else {
                    viewModel.loadModel(filename)
                }
            } catch (e: Exception) {
                tvError.text = "Error: ${e.message}"
                tvError.visibility = TextView.VISIBLE
            } finally {
                progressBar.visibility = ProgressBar.GONE
            }
        }
    }
    
    private fun generateText(prompt: String) {
        progressBar.visibility = ProgressBar.VISIBLE
        
        lifecycleScope.launch {
            viewModel.generate(prompt)
        }
    }
    
    private fun testOpenAIApi() {
        progressBar.visibility = ProgressBar.VISIBLE
        
        lifecycleScope.launch {
            try {
                val messages = listOf(
                    OpenAILikeAPI.ChatMessage(
                        role = "system",
                        content = "You are a helpful AI assistant."
                    ),
                    OpenAILikeAPI.ChatMessage(
                        role = "user",
                        content = "Hello, how are you?"
                    )
                )
                
                val response = modelManager.chatCompletion(messages)
                tvResponse.text = "API Response:\n\n${response.choices.first().message.content}"
            } catch (e: Exception) {
                tvError.text = "API Error: ${e.message}"
                tvError.visibility = TextView.VISIBLE
            } finally {
                progressBar.visibility = ProgressBar.GONE
            }
        }
    }
    
    override fun onDestroy() {
        // Clean up resources
        modelManager.cleanup()
        super.onDestroy()
    }
}