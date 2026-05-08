#!/bin/bash
# ============================================================
# AIOPE-INF: Standalone NDK build for ARM64-v8a
# Builds libaiope-inf.so without Android Studio
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CPP_DIR="$PROJECT_DIR/app/src/main/cpp"
BUILD_DIR="$PROJECT_DIR/build-arm64"
OUTPUT_DIR="$PROJECT_DIR/app/src/main/jniLibs/arm64-v8a"

# Check NDK
if [ -z "$ANDROID_NDK_HOME" ]; then
    echo "ERROR: ANDROID_NDK_HOME not set"
    echo "Export it: export ANDROID_NDK_HOME=/path/to/android-ndk-r26b"
    exit 1
fi

TOOLCHAIN="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"
if [ ! -f "$TOOLCHAIN" ]; then
    echo "ERROR: NDK toolchain not found at: $TOOLCHAIN"
    exit 1
fi

echo "=== AIOPE-INF ARM64-v8a Build ==="
echo "NDK: $ANDROID_NDK_HOME"
echo "Source: $CPP_DIR"
echo "Build: $BUILD_DIR"
echo ""

# Configure
echo "[1/3] Configuring CMake..."
cmake -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-26 \
    -DANDROID_ARM_NEON=ON \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DAIOPE_VULKAN=ON \
    "$CPP_DIR"

# Build
echo "[2/3] Building..."
cmake --build "$BUILD_DIR" --config Release -j$(nproc)

# Copy output
echo "[3/3] Copying to jniLibs..."
mkdir -p "$OUTPUT_DIR"
cp "$BUILD_DIR/libaiope-inf.so" "$OUTPUT_DIR/"

# Also copy c++ shared lib
LIBCPP="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -f "$LIBCPP" ]; then
    cp "$LIBCPP" "$OUTPUT_DIR/"
fi

echo ""
echo "=== Build Complete ==="
echo "Output: $OUTPUT_DIR/libaiope-inf.so"
ls -lh "$OUTPUT_DIR/"
