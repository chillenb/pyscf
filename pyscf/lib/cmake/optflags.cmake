
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

