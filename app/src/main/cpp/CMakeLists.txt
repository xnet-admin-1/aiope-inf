cmake_minimum_required(VERSION 3.18)
project("llama-jni-android")

# Set Android-specific variables
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Download and build llama.cpp
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")
    message(STATUS "Downloading llama.cpp...")
    execute_process(
        COMMAND git clone --depth 1 --branch master https://github.com/ggerganov/llama.cpp.git
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# Configure llama.cpp
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "")
set(LLAMA_CURL OFF CACHE BOOL "")
set(LLAMA_NATIVE OFF CACHE BOOL "")
set(LLAMA_MPI OFF CACHE BOOL "")
set(LLAMA_OPENBLAS OFF CACHE BOOL "")
set(LLAMA_CUBLAS OFF CACHE BOOL "")
set(LLAMA_CLBLAST OFF CACHE BOOL "")
set(LLAMA_VULKAN OFF CACHE BOOL "")  # Enable if you have Vulkan support
set(LLAMA_METAL OFF CACHE BOOL "")   # Android doesn't support Metal
set(LLAMA_KOMPUTE OFF CACHE BOOL "")
set(LLAMA_SYCL OFF CACHE BOOL "")
set(LLAMA_OPENCL OFF CACHE BOOL "")
set(LLAMA_HIPBLAS OFF CACHE BOOL "")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "")

# Include llama.cpp
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp EXCLUDE_FROM_ALL)

# Create our JNI library
add_library(llama-jni SHARED
    src/main/jni/llama_jni.cpp
    src/main/jni/jni_utils.cpp
)

# Include directories
target_include_directories(llama-jni PRIVATE
    ${LLAMA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jni
    ${Android_INCLUDE_DIRS}
)

# Link dependencies
target_link_libraries(llama-jni PRIVATE 
    llama
    atomic  # Needed for some ARM64 operations
)

# Properties for Android JNI
set_target_properties(llama-jni PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# Add prebuilt directories for OpenBLAS if needed
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/libs/arm64-v8a")
    add_library(openblas SHARED IMPORTED)
    set_target_properties(openblas PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/libs/arm64-v8a/libopenblas.so"
    )
    target_link_libraries(llama-jni PUBLIC openblas)
endif()

# Copy built library to libs directory
add_custom_command(TARGET llama-jni POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/libs/arm64-v8a
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:llama-jni> ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/libs/arm64-v8a/
    COMMENT "Copying llama-jni to Android libs directory"
)