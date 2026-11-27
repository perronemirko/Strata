# engine/src/inklin2ristrata/metal/CMakeLists.txt - Apple Silicon Metal backend plugin for strata-inklin2ristrata
# MIT License - Copyright (c) 2026 Strata contributors

enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

set(_inklin2ristrata_eng_root "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(_inklin2ristrata_gen_hdr "${CMAKE_CURRENT_BINARY_DIR}/inklin2ristrata_metal_embedded_shaders.h")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(
  OUTPUT "${_inklin2ristrata_gen_hdr}"
  COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.py"
          "${CMAKE_CURRENT_SOURCE_DIR}" "${_inklin2ristrata_gen_hdr}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/inklin2ristrata_metal_codecs.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/inklin2ristrata_metal_ops.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/inklin2ristrata_metal_dspark.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/inklin2ristrata_metal_vision.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/inklin2ristrata_metal_mpp.metal"
  COMMENT "Embedding Inklin2RiStrata-2-Mini Metal shaders"
)

add_library(strata_inklin2ristrata_metal STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/inklin2ristrata_metal_backend.mm"
  "${_inklin2ristrata_eng_root}/metal/src/mtl_rt.mm"
  "${_inklin2ristrata_gen_hdr}"
)
target_include_directories(strata_inklin2ristrata_metal PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${_inklin2ristrata_eng_root}/include"
  "${_inklin2ristrata_eng_root}/src/hy4flash/run"
  "${_inklin2ristrata_eng_root}/metal/src"
  "${_inklin2ristrata_eng_root}/third_party/stb"
)
target_compile_definitions(strata_inklin2ristrata_metal PRIVATE
  INK_METAL_EMBEDDED_SHADERS=1
  INK_METAL_HAS_MPP_SRC=1
  STRATA_GGML_COMMON_DEFAULT=""
  STRATA_METAL_KERNELS_DEFAULT=""
)
target_compile_options(strata_inklin2ristrata_metal PRIVATE -O3 -fobjc-arc)
target_link_libraries(strata_inklin2ristrata_metal PUBLIC strata_zstd_dec
  "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders")

target_compile_definitions(strata_inklin2ristrata_core PUBLIC INKLIN2RISTRATA_ENABLE_METAL=1)
target_link_libraries(strata-inklin2ristrata PRIVATE strata_inklin2ristrata_metal)
