cmake_minimum_required(VERSION 3.18)
project("aiope-inf" LANGUAGES C CXX)

# Build configuration
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# ARM64-v8a optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fno-finite-math-only")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math -fno-finite-math-only")

# Option: GPU acceleration via Vulkan
option(AIOPE_VULKAN "Enable Vulkan GPU acceleration" OFF)

# ============================================================
# llama.cpp integration (submodule at app/src/main/cpp/llama.cpp)
# ============================================================
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")

if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp not found at ${LLAMA_DIR}. Run: git submodule update --init")
endif()

# llama.cpp build options
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_NATIVE OFF CACHE BOOL "" FORCE)
set(LLAMA_MPI OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENBLAS OFF CACHE BOOL "" FORCE)
set(LLAMA_METAL OFF CACHE BOOL "" FORCE)
set(LLAMA_CUDA OFF CACHE BOOL "" FORCE)
set(LLAMA_HIPBLAS OFF CACHE BOOL "" FORCE)
set(LLAMA_SYCL OFF CACHE BOOL "" FORCE)
set(LLAMA_KOMPUTE OFF CACHE BOOL "" FORCE)

# Vulkan GPU acceleration with prebuilt shaders
if(AIOPE_VULKAN)
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)  # We build it ourselves
    add_compile_definitions(GGML_USE_VULKAN=1)  # Global so backend-reg.cpp sees it
    message(STATUS "AIOPE-INF: Vulkan GPU with prebuilt shaders")

    set(VK_SHADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders-prebuilt")
    file(GLOB VK_SHADER_CPPS "${VK_SHADERS_DIR}/*.comp.cpp")

    add_library(ggml-vulkan STATIC
        ${LLAMA_DIR}/ggml/src/ggml-vulkan/ggml-vulkan.cpp
        ${VK_SHADER_CPPS}
    )
    target_include_directories(ggml-vulkan PRIVATE
        ${LLAMA_DIR}/ggml/include
        ${LLAMA_DIR}/ggml/src
        ${LLAMA_DIR}/ggml/src/ggml-vulkan
        ${VK_SHADERS_DIR}
    )
    target_compile_definitions(ggml-vulkan PRIVATE GGML_USE_VULKAN=1)
    target_link_libraries(ggml-vulkan PRIVATE ggml-base)
else()
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    message(STATUS "AIOPE-INF: CPU-only mode")
endif()

# Include llama.cpp as subdirectory
add_subdirectory(${LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)

# ============================================================
# JNI shared library: libaiope-inf.so
# ============================================================
add_library(aiope-inf SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/../jni/llama_jni.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../jni/streaming.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../jni/quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../jni/gpu_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../jni/multimodal.cpp
)

# Build mtmd (multimodal) library - full: vision + audio
file(GLOB MTMD_MODEL_SOURCES ${LLAMA_DIR}/tools/mtmd/models/*.cpp)
set(MTMD_SOURCES
    ${LLAMA_DIR}/tools/mtmd/clip.cpp
    ${LLAMA_DIR}/tools/mtmd/mtmd.cpp
    ${LLAMA_DIR}/tools/mtmd/mtmd-audio.cpp
    ${LLAMA_DIR}/tools/mtmd/mtmd-helper.cpp
    ${LLAMA_DIR}/tools/mtmd/mtmd-image.cpp
    ${MTMD_MODEL_SOURCES}
)
add_library(mtmd STATIC ${MTMD_SOURCES})
target_include_directories(mtmd PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
    ${LLAMA_DIR}/tools/mtmd
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}/vendor
)
target_link_libraries(mtmd PRIVATE llama ggml)

target_include_directories(aiope-inf PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}/ggml/include
    ${LLAMA_DIR}/tools/mtmd
    ${CMAKE_CURRENT_SOURCE_DIR}/../jni
)

target_link_libraries(aiope-inf PRIVATE
    llama
    llama-common
    ggml
    mtmd
    android
    log
    atomic
)

# Vulkan linkage
if(AIOPE_VULKAN)
    set(VK_LIB "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/30/libvulkan.so")
    target_link_libraries(aiope-inf PRIVATE -Wl,--whole-archive ggml-vulkan -Wl,--no-whole-archive ${VK_LIB})
    target_link_libraries(ggml PRIVATE ggml-vulkan ${VK_LIB})
    target_compile_definitions(aiope-inf PRIVATE AIOPE_VULKAN=1)
endif()

# Compiler definitions
target_compile_definitions(aiope-inf PRIVATE
    AIOPE_VERSION="1.0.0"
    ANDROID_ARM_NEON=1
)

set_target_properties(aiope-inf PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "aiope-inf"
)
