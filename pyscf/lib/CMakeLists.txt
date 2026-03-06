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
project (pyscf LANGUAGES C VERSION 2.12.1)

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

# Optimization flags. See cmake/optflags.cmake for details.
option(BUILD_MARCH_NATIVE "gcc flag -march=native or equivalent" OFF)
option(ENABLE_VECTOR_MATH "Use vectorized mathematical functions" OFF)
option(ENABLE_IPO "Use interprocedural optimization (IPO)/link time optimization (LTO)" OFF)

# OpenMP options.
option(ENABLE_OPENMP "Compiling C extensions with openmp" ON)
option(FORCE_INTEL_OPENMP "Force use of Intel OpenMP library" OFF)

# FFTW options
option(ENABLE_FFTW "Using fftw3" OFF)
option(BUILD_FFTW "Building fftw3" OFF)

# It's recommended to set FORCE_INTEL_OPENMP for performance.
option(ENABLE_MKL "Use MKL for BLAS, LAPACK and FFT" OFF)

option(ENABLE_SMD "Compiling SMD Fortran code" OFF)
option(WITH_F12 "Compling F12 integrals" ON)
option(BUILD_LIBCINT "Building libcint for gaussian integral evaluation" ON)
option(USE_QCINT "Using the qcint library (optimized for x86-64) for gaussian integral evaluation" OFF)
option(ENABLE_LIBXC "Using libxc for XC functional library" ON)
option(ENABLE_XCFUN "Using xcfun for XC functional library" ON)
option(BUILD_LIBXC "Build libxc if we can't find it" ON)
option(BUILD_XCFUN "Build xcfun if we can't find it" ON)

option(ENABLE_LIBXSMM "Using libxsmm" OFF)
option(FIXRPATH "Modify RPATH for building wheels." OFF)

message(STATUS "Summary of provided options:")
cmake_print_variables(BUILD_MARCH_NATIVE
  ENABLE_VECTOR_MATH ENABLE_IPO
  ENABLE_OPENMP FORCE_INTEL_OPENMP
  ENABLE_FFTW BUILD_FFTW ENABLE_MKL
  ENABLE_SMD
  WITH_F12 USE_QCINT
  ENABLE_LIBXC ENABLE_XCFUN
  ENABLE_LIBXSMM)

include(cmake/optflags.cmake)
include(cmake/openmp.cmake)
include(cmake/mkl.cmake)

if(FIXRPATH)
  include(cmake/fixrpath.cmake)
endif()

if(ENABLE_IPO)
  check_ipo_supported()
  message(STATUS "Enabled interprocedural optimization (IPO)/link time optimization (LTO)")
endif()

# Enable warnings for supported compilers.
if(${CMAKE_C_COMPILER_ID} IN_LIST "GNU;Clang;IntelLLVM")
  list(APPEND PYSCF_OPTFLAGS "-Wall")
endif()


# Architecture specified cmake flags.  See also the templates in
# pyscf/lib/cmake_arch_config
if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.arch.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.arch.inc")
endif()

check_symbol_exists(ffsll "strings.h" HAVE_FFS)

# If MKL was found, BLAS_LIBRARIES will already be set by mkl.cmake.
# If not, try to find a BLAS library with find_package(BLAS).
if (NOT BLAS_LIBRARIES)
  find_package(BLAS)
endif()
if (NOT BLAS_LIBRARIES)
  message(FATAL_ERROR "A required library with BLAS API not found.")
else()
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()

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
  include(cmake/libxc.cmake)
endif()

if(ENABLE_XCFUN)
  include(cmake/xcfun.cmake)
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_SOURCE_DIR}/config.h")

# each subdirectory appends its library targets to
# PYSCF_TARGETS.
set(PYSCF_TARGETS)

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

if(ENABLE_MKL AND ENABLE_FFTW)
  message(FATAL_ERROR "Cannot use both MKL and FFTW for FFT. Please choose only one.")
endif()

if(BUILD_LIBCINT)
  set(LIBCINT_GIT https://github.com/sunqm/libcint.git) # libcint is a portable, cross-platform implementation
  set(LIBCINT_VERSION v6.1.3)
  if (USE_QCINT)
    set(LIBCINT_GIT https://github.com/sunqm/qcint.git) # qcint is an optimized implementation for x86-64 architecture
    set(LIBCINT_VERSION v6.1.3)
    if(NOT BUILD_MARCH_NATIVE)
      message(WARNING "The BUILD_MARCH_NATIVE option is not specified! qcint may not compile unless you explicitly pass compiler flags that turn on vectorization!")
    endif()
  endif()

  include(FetchContent)
  FetchContent_Declare(libcint
    GIT_REPOSITORY ${LIBCINT_GIT}
    GIT_TAG ${LIBCINT_VERSION}
    FIND_PACKAGE_ARGS
  )
  message(STATUS "Fetching libcint from ${LIBCINT_GIT} at tag ${LIBCINT_VERSION}")
  FetchContent_MakeAvailable(libcint)
  set_target_properties(cint PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  install(TARGETS cint
          EXPORT pyscf-targets
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif() # BUILD_LIBCINT

# Loop over all library targets.
foreach(target_lib ${PYSCF_TARGETS})

  target_compile_options(${target_lib} PRIVATE ${PYSCF_OPTFLAGS})
  target_include_directories(${target_lib} PRIVATE ${PROJECT_SOURCE_DIR})

  if(ENABLE_IPO)
    set_property(TARGET ${target_lib} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()

  set_target_properties(${target_lib} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    C_STANDARD 99
  )

endforeach()






#                               Installation
# ----------------------------------------------------------------------------
#

install(FILES
    config.h
    agf2/ragf2.h
    ao2mo/nr_ao2mo.h
    ao2mo/nr_incore.h
    ao2mo/r_ao2mo.h
    config.h
    dft/grid_basis.h
    dft/grid_common.h
    dft/multigrid.h
    gto/ft_ao.h
    gto/grid_ao_drv.h
    gto/gto.h
    gto/nr_ecp.h
    mcscf/fci.h
    mp/mp2.h
    np_helper/np_helper.h
    pbc/cell.h
    pbc/fft.h
    pbc/fill_ints.h
    pbc/neighbor_list.h
    pbc/optimizer.h
    pbc/pbc.h
    vhf/cvhf.h
    vhf/fblas.h
    vhf/nr_direct.h
    vhf/optimizer.h
    vhf/r_direct_dot.h
    vhf/time_rev.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pyscf
  )

# target_include_directories(corepyscf
#       PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/pyscf>)

# include(CMakePackageConfigHelpers)
# configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/pyscfConfig.cmake.in
#   ${CMAKE_CURRENT_BINARY_DIR}/pyscfConfig.cmake
#   INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pyscf)

# install(TARGETS corepyscf
#         EXPORT pyscf-targets
#         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#         ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#         RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#         INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )



# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/pyscfConfigVersion.cmake"
#     COMPATIBILITY AnyNewerVersion)

# install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pyscfConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/pyscfConfigVersion.cmake"
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pyscf)

# export(EXPORT pyscf-targets
#     FILE ${CMAKE_CURRENT_BINARY_DIR}/pyscfTargets.cmake
#         NAMESPACE pyscf::
# )

# install(EXPORT pyscf-targets
#         FILE pyscfTargets.cmake
#         NAMESPACE pyscf::
#         DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pyscf
# )
