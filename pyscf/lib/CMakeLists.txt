# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 3.25)
project (pyscf LANGUAGES C)

include(CheckSymbolExists)
include(CMakePrintHelpers)
include(CheckIPOSupported)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  # TODO: If updating to minimum requirement cmake>=3.7, use
  # CMAKE_SHARED_LINKER_FLAGS_INIT to combine LDFLAGS options.
  # https://cmake.org/cmake/help/v3.7/variable/CMAKE_SHARED_LINKER_FLAGS_INIT.html
  # see also issue #661
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()

option(BUILD_MARCH_NATIVE "gcc flag -march=native or equivalent" OFF)
option(ENABLE_VECTOR_MATH "Use vectorized mathematical functions" OFF)
option(ENABLE_IPO "Use interprocedural optimization (IPO)/link time optimization (LTO)" OFF)
option(ENABLE_OPENMP "Compiling C extensions with openmp" ON)
option(FORCE_INTEL_OPENMP "Force use of Intel OpenMP library" OFF)
option(ENABLE_FFTW "Using fftw3" OFF)
option(BUILD_FFTW "Building fftw3" OFF)
option(USE_MKL_FFT "Using MKL for FFT" OFF)
option(ENABLE_SMD "Compiling SMD Fortran code" OFF)
option(WITH_F12 "Compling F12 integrals" ON)
option(USE_QCINT "Using the qcint library (optimized for x86-64) for gaussian integral evaluation" OFF)
option(ENABLE_LIBXC "Using libxc for XC functional library" ON)
option(ENABLE_XCFUN "Using xcfun for XC functional library" ON)

option(ENABLE_LIBXSMM "Using libxsmm" OFF)

message(STATUS "Summary of provided options:")
cmake_print_variables(BUILD_MARCH_NATIVE
  ENABLE_VECTOR_MATH ENABLE_IPO
  ENABLE_OPENMP FORCE_INTEL_OPENMP
  ENABLE_FFTW BUILD_FFTW USE_MKL_FFT
  ENABLE_SMD
  WITH_F12 USE_QCINT
  ENABLE_LIBXC ENABLE_XCFUN
  ENABLE_LIBXSMM)

include(cmake/optflags.cmake)
include(cmake/jobserver.cmake)

# outputs for include(cmake/openmp.cmake):
# boolean variable HAVE_OPENMP
# variable OPENMP_C_PROPERTIES, used as follows
# target_link_libraries(tgt ${OPENMP_C_PROPERTIES})
include(cmake/openmp.cmake)


add_library(corepyscf SHARED)
set_property(TARGET corepyscf PROPERTY C_STANDARD 99)

if(${CMAKE_C_COMPILER_ID} IN_LIST "GNU;Clang;IntelLLVM")
  target_compile_options(corepyscf PRIVATE -Wall)
endif()


if(ENABLE_IPO)
  check_ipo_supported()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_property(TARGET corepyscf PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  message(STATUS "Enabled interprocedural optimization (IPO)/link time optimization (LTO)")
endif()

# Architecture specified cmake flags.  See also the templates in
# pyscf/lib/cmake_arch_config
if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.arch.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.arch.inc")
endif()

check_symbol_exists(ffsll "strings.h" HAVE_FFS)

if (NOT BLAS_LIBRARIES)
  find_package(BLAS)
endif()
if (NOT BLAS_LIBRARIES)
  message(FATAL_ERROR "A required library with BLAS API not found.")
else()
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()

# You can use pkgconfig to find MKL. Pass
# the following options:
# -DBLA_PREFER_PKGCONFIG -DBLA_PKGCONFIG_BLAS="mkl-sdl"
# (conda env with mkl, mkl-include and mkl-devel installed)
#
# or on HPC, etc., you can use
# -DBLA_PREFER_PKGCONFIG -DBLA_PKGCONFIG_BLAS="mkl-dynamic-[i]lp64-[seq,tbb,iomp,gomp]"
# for more flexibility.
# You can also set these variables in your cmake.arch.inc.

# if unable to find mkl library, just create BLAS_LIBRARIES here, e.g.
# set(BLAS_LIBRARIES "-L/path/to/mkl/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lmkl_avx -lm")
# or
# set(BLAS_LIBRARIES "                  /path/to/mkl/lib/intel64/libmkl_intel_lp64.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_sequential.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_core.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_avx.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_def.so")
# set(BLAS_LIBRARIES "-Wl,-rpath=${MKLROOT}/lib/intel64/ ${BLAS_LIBRARIES}")
#
# These settings can be written in the cmake.arch.inc file. This config file
# will automatically load all settings specified in cmake.arch.inc


if(NOT XCFUN_MAX_ORDER)
  set(XCFUN_MAX_ORDER 3)
endif()

if(ENABLE_LIBXC)
  message(STATUS "Looking for LibXC")
  find_package(Libxc REQUIRED)
  cmake_print_properties(TARGETS Libxc::xc PROPERTIES LOCATION INTERFACE_LINK_LIBRARIES)
  target_link_libraries(corepyscf PRIVATE Libxc::xc)
endif()

if(ENABLE_XCFUN)
  message(STATUS "Looking for XCFun")
  find_package(XCFun REQUIRED)
  cmake_print_properties(TARGETS XCFun::xcfun PROPERTIES LOCATION INTERFACE_LINK_LIBRARIES)
  target_link_libraries(corepyscf PRIVATE Libxc::xc)
endif()



configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_SOURCE_DIR}/config.h")

target_include_directories(corepyscf PRIVATE ${PROJECT_SOURCE_DIR})

add_subdirectory(np_helper)
add_subdirectory(gto)
add_subdirectory(vhf)
add_subdirectory(ao2mo)
add_subdirectory(mcscf)
add_subdirectory(cc)
add_subdirectory(ccsdt)
add_subdirectory(mp)
add_subdirectory(ri)
add_subdirectory(pbc)
add_subdirectory(agf2)
add_subdirectory(dft)

if(ENABLE_SMD)
  add_subdirectory(solvent)
endif(ENABLE_SMD)

if(APPLE)
    set(ENABLE_LIBXSMM OFF)
endif()

if(USE_MKL_FFT AND ENABLE_FFTW)
  message(FATAL_ERROR "Cannot use both MKL and FFTW for FFT. Please choose only one.")
endif()

message(STATUS "pyscf sources: " ${PYSCF_SOURCES})
target_sources(corepyscf PRIVATE ${PYSCF_SOURCES})
target_link_libraries(corepyscf PRIVATE ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
if(ENABLE_LIBXC)
  target_link_libraries(corepyscf PRIVATE xc)
endif()


if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()
