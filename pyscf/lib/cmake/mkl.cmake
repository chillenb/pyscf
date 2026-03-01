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

if(ENABLE_MKL)

  message(STATUS "Looking for MKL")

  # Pick an openmp runtime for MKL.
  if(NOT FORCE_INTEL_OPENMP AND ${CMAKE_C_COMPILER_ID} MATCHES "GNU")
    set(MKL_THREADING "gnu_thread")
    set(MKL_OMP "gomp")
  else()
    set(MKL_THREADING "intel_thread")
    set(MKL_OMP "iomp")
  endif()

  set(MKL_INTERFACE "lp64")
  set(MKL_LINK "dynamic")
  find_package(MKL QUIET)

  if(MKL_FOUND)
    message(STATUS "MKL found: ${MKL_INCLUDE}")
    set(BLAS_LIBRARIES MKL::MKL)
  else()

    # MKLConfig.cmake not found, so we will search for MKL with pkg-config.
    message(STATUS "find_package(MKL) didn't work; we'll try FindBLAS with pkg-config.")

    # construct the name of the .pc file we want.
    set(MKL_PKGCONFNAME "mkl-${MKL_LINK}-${MKL_INTERFACE}-${MKL_OMP}")

    # load pkg-config.
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
      pkg_check_modules(MKL_PKGCONFIG IMPORTED_TARGET ${MKL_PKGCONFNAME})
      if(MKL_PKGCONFIG_FOUND)
        message(STATUS "Found MKL with pkg-config: ${MKL_PKGCONFIG_LDFLAGS}")
        set(BLAS_LIBRARIES PkgConfig::MKL_PKGCONFIG)
      else()
        message(STATUS "pkg-config couldn't find MKL with the name ${MKL_PKGCONFNAME}.")
      endif()
    endif()
  endif()
endif()