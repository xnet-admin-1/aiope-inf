# AIOPE-INF

On-device LLM inference engine for Android ARM64-v8a. Serves GGUF models locally with an OpenAI-compatible API.

## Features

- **Direct llama.cpp integration** — compiled as JNI shared library
- **Vulkan GPU acceleration** — automatic layer offloading based on VRAM
- **Streaming responses** — token-by-token via callbacks and SSE
- **OpenAI API server** — `POST /v1/chat/completions` on localhost:8080
- **On-device quantization** — Q2_K through Q8_0
- **Foreground service** — keeps inference alive in background

## Architecture

```
┌─────────────────────────────────────────────┐
│  Android App (Kotlin)                       │
├─────────────────────────────────────────────┤
│  OpenAIServer    │  ModelManager            │
│  (HTTP :8080)    │  (lifecycle, streaming)  │
├─────────────────────────────────────────────┤
│  LlamaJNI (JNI bridge)                     │
├─────────────────────────────────────────────┤
│  libaiope-inf.so (C++17)                   │
│  ├── llama_jni.cpp    (core inference)     │
│  ├── streaming.cpp    (SSE + callbacks)    │
│  ├── quantize.cpp     (on-device quant)    │
│  └── gpu_backend.cpp  (Vulkan detection)   │
├─────────────────────────────────────────────┤
│  llama.cpp (submodule)                     │
│  └── ggml (Vulkan/NEON compute)            │
└─────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone
git clone git@github.com:xnet-admin-1/aiope-inf.git
cd aiope-inf

# 2. Setup (clones llama.cpp)
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Build
./gradlew assembleDebug

# 4. Install
adb install app/build/outputs/apk/debug/app-debug.apk

# 5. Push a GGUF model
adb push model.Q4_K_M.gguf /data/data/com.aiope.inf/files/models/
```

## OpenAI API Usage

Once the server is running on-device:

```bash
# Port forward
adb forward tcp:8080 tcp:8080

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "stream": true
  }'

# List models
curl http://localhost:8080/v1/models

# Health check
curl http://localhost:8080/health
```

## Standalone NDK Build

Build without Android Studio:

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk-r26b
chmod +x scripts/build-arm64.sh
./scripts/build-arm64.sh
```

## Quantization

Quantize models on-device:

| Type | Bits | Size Ratio | Quality |
|------|------|-----------|---------|
| Q2_K | 2 | ~15% | Lowest |
| Q3_K_M | 3 | ~22% | Low |
| Q4_K_M | 4 | ~28% | Good (recommended) |
| Q5_K_M | 5 | ~35% | Very good |
| Q6_K | 6 | ~40% | Excellent |
| Q8_0 | 8 | ~50% | Near lossless |

## GPU Acceleration

Vulkan is auto-detected. GPU layer count is recommended based on available VRAM:

```kotlin
val manager = ModelManager(context)
val gpuInfo = manager.getGpuInfo()
// {"available":true,"device_name":"Adreno 740","vram_mb":2048,...}
```

## Requirements

- Android 8.0+ (API 26)
- ARM64-v8a device
- 2GB+ RAM (4GB+ recommended for 7B models)
- Vulkan 1.1+ for GPU acceleration (optional)

## Project Structure

```
aiope-inf/
├── app/src/main/
│   ├── java/com/aiope/inf/
│   │   ├── LlamaJNI.kt          # JNI interface
│   │   ├── ModelManager.kt      # High-level API
│   │   ├── OpenAIServer.kt      # HTTP server
│   │   ├── InferenceService.kt  # Foreground service
│   │   └── MainActivity.kt      # UI
│   ├── jni/
│   │   ├── llama_jni.cpp        # Core JNI (load, generate, abort)
│   │   ├── streaming.cpp        # Streaming + SSE generation
│   │   ├── quantize.cpp         # On-device quantization
│   │   └── gpu_backend.cpp      # Vulkan detection & management
│   ├── cpp/
│   │   ├── CMakeLists.txt       # Native build config
│   │   └── llama.cpp/           # Submodule
│   └── res/layout/
│       └── activity_main.xml
├── scripts/
│   ├── setup.sh                 # Project setup
│   └── build-arm64.sh           # Standalone NDK build
├── build.gradle.kts
├── settings.gradle.kts
└── app/build.gradle.kts
```

## License

MIT
