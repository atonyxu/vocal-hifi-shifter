cmake_minimum_required(VERSION 3.18)
project(hifishifter VERSION 1.0.0 LANGUAGES CXX)

# -------------------------------------------------------------------
# Options
# -------------------------------------------------------------------
# USE_TENSORRT: build the TensorRT execution provider path. Requires the
#   ONNXRuntime GPU (CUDA) package, which has the TensorRT EP compiled in.
# STATIC_LINK: link ONNXRuntime as a *static* library instead of the shared
#   DLL/.so. This produces a true single-binary file but requires you to
#   provide a statically-built ONNXRuntime that includes the CUDA + TensorRT
#   providers (build ORT from source with --build_shared_lib=OFF
#   --use_tensorrt --use_cuda). When OFF, the shared library is linked and
#   must be shipped next to the executable (or installed system-wide).
option(USE_TENSORRT "Enable TensorRT execution provider (requires ONNXRuntime GPU build)" ON)
option(STATIC_LINK "Statically link ONNXRuntime (needs a custom static ORT build)" OFF)
option(ORT_FROM_SOURCE_DIR "Use a pre-built ONNXRuntime source dir (set ORT_ROOT)" OFF)
set(ORT_ROOT "" CACHE PATH "Path to a pre-extracted ONNXRuntime package (root with include/ and lib/)")
set(ORT_VERSION "1.24.3" CACHE STRING "ONNX Runtime version to auto-download")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Static-link runtime when requested (single-file output).
if(STATIC_LINK)
    if(MSVC)
        # /MT instead of /MD for static CRT.
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        # Try static libstdc++ / libc.
        add_link_options(-static-libgcc -static-libstdc++)
    endif()
endif()

include(FetchContent)

# -------------------------------------------------------------------
# Locate ONNX Runtime (GPU/CUDA build for TensorRT support)
# -------------------------------------------------------------------
# Resolves the real ORT root: ORT_ROOT may point either at a flat
# layout (include/, lib/) or at a parent containing a versioned
# subfolder (onnxruntime-.../include, onnxruntime-.../lib).
set(ORT_FOUND FALSE)

function(_hifishifter_resolve_ort root out_var)
    if(EXISTS "${root}/include/onnxruntime_cxx_api.h")
        set(${out_var} "${root}" PARENT_SCOPE)
    else()
        file(GLOB children LIST_DIRECTORIES true "${root}/onnxruntime-*")
        foreach(d IN LISTS children)
            if(EXISTS "${d}/include/onnxruntime_cxx_api.h")
                set(${out_var} "${d}" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endif()
endfunction()

if(ORT_ROOT)
    _hifishifter_resolve_ort("${ORT_ROOT}" ORT_RESOLVED)
    if(ORT_RESOLVED)
        set(ORT_INCLUDE_DIR "${ORT_RESOLVED}/include")
        set(ORT_ROOT "${ORT_RESOLVED}")
        set(ORT_FOUND TRUE)
    endif()
endif()

if(NOT ORT_FOUND)
    set(ORT_CACHE_DIR "${CMAKE_BINARY_DIR}/_deps/ort-src")
    _hifishifter_resolve_ort("${ORT_CACHE_DIR}" ORT_RESOLVED)
    if(ORT_RESOLVED)
        message(STATUS "ONNX Runtime already downloaded, reusing cache.")
        set(ORT_INCLUDE_DIR "${ORT_RESOLVED}/include")
        set(ORT_ROOT "${ORT_RESOLVED}")
        set(ORT_FOUND TRUE)
    else()
        if(WIN32)
            set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-gpu_cuda13-${ORT_VERSION}.zip")
        else()
            set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu_cuda13-${ORT_VERSION}.tgz")
        endif()
        message(STATUS "Downloading ONNX Runtime GPU/CUDA from: ${ORT_URL}")
        FetchContent_Declare(ort URL ${ORT_URL} DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(ort)
        _hifishifter_resolve_ort("${ort_SOURCE_DIR}" ORT_RESOLVED)
        set(ORT_INCLUDE_DIR "${ORT_RESOLVED}/include")
        set(ORT_ROOT "${ORT_RESOLVED}")
        if(ORT_RESOLVED AND EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h")
            set(ORT_FOUND TRUE)
        endif()
    endif()
endif()

if(NOT ORT_FOUND)
    message(FATAL_ERROR "ONNX Runtime not found. Set -DORT_ROOT=<path> or enable network download.")
endif()

if(WIN32)
    if(STATIC_LINK)
        set(ORT_LIB "${ORT_ROOT}/lib/onnxruntime_static.lib")
        if(NOT EXISTS "${ORT_LIB}")
            set(ORT_LIB "${ORT_ROOT}/lib/onnxruntime.lib")
        endif()
    else()
        set(ORT_LIB "${ORT_ROOT}/lib/onnxruntime.lib")
    endif()
    set(ORT_DLL "${ORT_ROOT}/lib/onnxruntime.dll")
    # CUDA runtime DLLs that the ORT GPU package depends on (if bundled).
    file(GLOB ORT_CUDNN_DLLS "${ORT_ROOT}/lib/cudnn*.dll")
else()
    find_library(ORT_LIB onnxruntime PATHS "${ORT_ROOT}/lib" NO_DEFAULT_PATH)
    set(ORT_DLL "")
    set(ORT_CUDNN_DLLS "")
endif()

message(STATUS "ONNX Runtime include: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ORT_LIB}")

# -------------------------------------------------------------------
# Executable
# -------------------------------------------------------------------
add_executable(hifishifter
    src/main.cpp
    src/onnx_pipeline.cpp)

target_include_directories(hifishifter PRIVATE src ${ORT_INCLUDE_DIR})

if(USE_TENSORRT)
    target_compile_definitions(hifishifter PRIVATE HIFISHIFTER_USE_TENSORRT=1)
    message(STATUS "TensorRT execution provider: ENABLED")
else()
    message(STATUS "TensorRT execution provider: DISABLED (CPU/CUDA only)")
endif()

if(WIN32)
    target_compile_definitions(hifishifter PRIVATE NOMINMAX)
    if(STATIC_LINK)
        target_compile_options(hifishifter PRIVATE /EHsc)
    endif()
endif()

# Link ONNX Runtime (static or shared).
if(STATIC_LINK)
    target_link_libraries(hifishifter PRIVATE ${ORT_LIB})
    if(WIN32)
        # Static ORT on Windows needs these system libraries.
        target_link_libraries(hifishifter PRIVATE
            Ws2_32 Shlwapi Bcrypt UserEnv Rpcrt4 Advapi32)
    endif()
else()
    target_link_libraries(hifishifter PRIVATE ${ORT_LIB})
endif()

# -------------------------------------------------------------------
# Post-build: copy shared libs next to the executable (dynamic mode)
# -------------------------------------------------------------------
if(NOT STATIC_LINK AND WIN32)
    foreach(DLL IN LISTS ORT_DLL ORT_CUDNN_DLLS)
        if(DLL AND EXISTS "${DLL}")
            add_custom_command(TARGET hifishifter POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DLL}" "$<TARGET_FILE_DIR:hifishifter>")
            message(STATUS "Will copy: ${DLL}")
        endif()
    endforeach()
endif()

# -------------------------------------------------------------------
# Copy the models directory next to the binary so it runs out of the box
# (models are loaded from ./models by default).
# -------------------------------------------------------------------
add_custom_command(TARGET hifishifter POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:hifishifter>/models"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/models/pc_nsf_hifigan.onnx"
        "${CMAKE_CURRENT_SOURCE_DIR}/models/fcpe.onnx"
        "${CMAKE_CURRENT_SOURCE_DIR}/models/cent_table.bin"
        "${CMAKE_CURRENT_SOURCE_DIR}/models/mel_filterbank.bin"
        "$<TARGET_FILE_DIR:hifishifter>/models/"
    COMMENT "Copying models to output directory")
