#!/bin/bash
# ============================================================
# AIOPE-INF: Setup script
# Clones llama.cpp and prepares the project for building
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CPP_DIR="$PROJECT_DIR/app/src/main/cpp"
LLAMA_DIR="$CPP_DIR/llama.cpp"

echo "=== AIOPE-INF Setup ==="
echo "Project: $PROJECT_DIR"

# Clone llama.cpp if not present
if [ ! -d "$LLAMA_DIR" ]; then
    echo "[1/3] Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
else
    echo "[1/3] llama.cpp already present, pulling latest..."
    cd "$LLAMA_DIR" && git pull && cd "$PROJECT_DIR"
fi

# Verify NDK
echo "[2/3] Checking Android NDK..."
if [ -z "$ANDROID_NDK_HOME" ]; then
    # Try common locations
    if [ -d "$HOME/Android/Sdk/ndk" ]; then
        NDK_DIR=$(ls -d "$HOME/Android/Sdk/ndk"/*/ 2>/dev/null | tail -1)
        export ANDROID_NDK_HOME="$NDK_DIR"
    elif [ -d "$ANDROID_HOME/ndk" ]; then
        NDK_DIR=$(ls -d "$ANDROID_HOME/ndk"/*/ 2>/dev/null | tail -1)
        export ANDROID_NDK_HOME="$NDK_DIR"
    fi
fi

if [ -n "$ANDROID_NDK_HOME" ]; then
    echo "  NDK found: $ANDROID_NDK_HOME"
else
    echo "  WARNING: ANDROID_NDK_HOME not set"
    echo "  Install NDK via: Android Studio > SDK Manager > SDK Tools > NDK"
fi

# Verify Vulkan headers
echo "[3/3] Checking Vulkan support..."
if [ -n "$ANDROID_NDK_HOME" ]; then
    VULKAN_H="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/vulkan/vulkan.h"
    if [ -f "$VULKAN_H" ]; then
        echo "  Vulkan headers found"
    else
        echo "  WARNING: Vulkan headers not found at expected path"
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Build commands:"
echo "  ./gradlew assembleDebug     # Debug APK"
echo "  ./gradlew assembleRelease   # Release APK"
echo ""
echo "APK output:"
echo "  app/build/outputs/apk/debug/app-debug.apk"
echo ""
echo "To test OpenAI API after install:"
echo "  curl http://localhost:8080/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
