# engine/src/inklin2ristrata/sycl/CMakeLists.txt - Intel Arc SYCL (dg2 + bmg_g31 B70) backend plugin for strata-inklin2ristrata
# CRITICAL: Zero reqd_sub_group_size(8); strictly SG16!
# MIT License - Copyright (c) 2026 Strata contributors

set(_inklin2ristrata_eng_root "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

add_library(strata_inklin2ristrata_sycl STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/inklin2ristrata_sycl_backend.cpp"
)
target_include_directories(strata_inklin2ristrata_sycl PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${_inklin2ristrata_eng_root}/include"
  "${_inklin2ristrata_eng_root}/src/hy4flash/sycl"
  "${_inklin2ristrata_eng_root}/third_party/stb"
)

set(STRATA_SYCL_TARGETS "spir64,intel_gpu_acm_g10,intel_gpu_bmg_g21,intel_gpu_bmg_g31"
    CACHE STRING "SYCL AOT/JIT targets for Intel Arc Alchemist + Battlemage")

if(MSVC OR WIN32)
  target_compile_options(strata_inklin2ristrata_sycl PRIVATE /fsycl /O2 -fno-fast-math "-fsycl-targets=${STRATA_SYCL_TARGETS}")
  target_link_options(strata-inklin2ristrata PRIVATE /fsycl "-fsycl-targets=${STRATA_SYCL_TARGETS}")
else()
  target_compile_options(strata_inklin2ristrata_sycl PRIVATE -fsycl -O3 -fno-fast-math "-fsycl-targets=${STRATA_SYCL_TARGETS}")
  target_link_options(strata-inklin2ristrata PRIVATE -fsycl "-fsycl-targets=${STRATA_SYCL_TARGETS}")
endif()

# Optional Intel oneMKL SYCL BLAS (unlocks Intel Arc XMX DPAS GEMM on Alchemist & Battlemage!)
set(_inklin2ristrata_mkl_root "$ENV{MKLROOT}")
if(NOT _inklin2ristrata_mkl_root AND EXISTS "/opt/intel/oneapi/mkl/latest/include/oneapi/mkl/blas.hpp")
  set(_inklin2ristrata_mkl_root "/opt/intel/oneapi/mkl/latest")
endif()
if(_inklin2ristrata_mkl_root AND EXISTS "${_inklin2ristrata_mkl_root}/include/oneapi/mkl/blas.hpp")
  message(STATUS "Inklin2RiStrata SYCL: enabling Intel oneMKL XMX GEMM (${_inklin2ristrata_mkl_root})")
  target_compile_definitions(strata_inklin2ristrata_sycl PRIVATE INK_SYCL_USE_ONEMKL=1)
  target_include_directories(strata_inklin2ristrata_sycl PRIVATE "${_inklin2ristrata_mkl_root}/include")
  if(NOT WIN32)
    target_link_options(strata-inklin2ristrata PRIVATE "-L${_inklin2ristrata_mkl_root}/lib" "-Wl,-rpath,${_inklin2ristrata_mkl_root}/lib")
    target_link_libraries(strata_inklin2ristrata_sycl PUBLIC mkl_sycl_blas mkl_intel_lp64 mkl_sequential mkl_core)
  endif()
endif()

target_link_libraries(strata_inklin2ristrata_sycl PUBLIC strata_inklin2ristrata_core strata_zstd_dec)
target_compile_definitions(strata_inklin2ristrata_core PUBLIC INKLIN2RISTRATA_ENABLE_SYCL=1)
target_link_libraries(strata-inklin2ristrata PRIVATE strata_inklin2ristrata_sycl)
