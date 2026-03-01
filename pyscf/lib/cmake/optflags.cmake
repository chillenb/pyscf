# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

# Author: Chris Hillenbrand <chillenbrand15@gmail.com>

# Supported compilers: GCC, Clang, Intel Classic (icc),
# Intel LLVM-based (icx). Partial support for NVHPC (nvc).

# Inputs:
# BUILD_MARCH_NATIVE -- Whether to enable -march=native and similar flags.
# ENABLE_VECTOR_MATH -- Whether to enable vector math library support.

# Output:
# PYSCF_OPTFLAGS -- List of optimization flags to use when compiling PySCF.


include(CheckCCompilerFlag)

if(BUILD_MARCH_NATIVE)
  # Classic icc and LLVM-based icx
  if (${CMAKE_C_COMPILER_ID} MATCHES "Intel")
    list(APPEND _PYSCF_OPTFLAGS -xHost -unroll-aggressive -fno-math-errno)
  # Clang and GCC
  elseif(${CMAKE_C_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
    list(APPEND _PYSCF_OPTFLAGS -march=native -funroll-loops -ftree-vectorize -fno-math-errno)
  # nvc
  elseif(${CMAKE_C_COMPILER_ID} STREQUAL "NVHPC")
    list(APPEND _PYSCF_OPTFLAGS -mcpu=host -Munroll -Mvect=simd)
  endif()
endif()

if(ENABLE_VECTOR_MATH)
  # Classic icc and LLVM-based icx
  if (${CMAKE_C_COMPILER_ID} MATCHES "Intel")
    list(APPEND _PYSCF_OPTFLAGS -fimf-use-svml=true)
  # GCC
  elseif(${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
    list(APPEND _PYSCF_OPTFLAGS -ffast-math)
  # Clang doesn't have its own vector math library.
  # Need to tell it to use libmvec from glibc
  elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
    list(APPEND _PYSCF_OPTFLAGS -ffast-math -fveclib=libmvec)
  endif()
endif()

# qcint 6.1.3 can check the supported instruction set
# and will error if it doesn't have at least SSE3. So we don't need to
# add -msse3.


foreach(OPTFLAG IN LISTS _PYSCF_OPTFLAGS)
  string(REGEX REPLACE "[-=]" "" FLAG_NAME ${OPTFLAG})
  string(TOUPPER ${FLAG_NAME} FLAG_NAME)
  check_c_compiler_flag(${OPTFLAG} COMPILER_SUPPORTS_${FLAG_NAME})
  if(COMPILER_SUPPORTS_${FLAG_NAME})
    list(APPEND PYSCF_OPTFLAGS ${OPTFLAG})
  else()
    message(WARNING "The compiler does not support the flag ${OPTFLAG}, which will be ignored.")
  endif()
endforeach()


message(STATUS "PySCF optimization flags: ${PYSCF_OPTFLAGS}")

