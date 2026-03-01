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


# This file is included by the top-level CMakeLists.txt.
# Inputs:
# ENABLE_OPENMP       -- Whether to enable OpenMP support.
# FORCE_INTEL_OPENMP  -- Use Intel OpenMP runtime.

# Outputs:
# HAVE_OPENMP        -- Whether OpenMP was found.
# OPENMP_C_PROPERTIES -- Target containing OpenMP compile options and libraries.

if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS C)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

if(FORCE_INTEL_OPENMP)
  if(ENABLE_OPENMP AND HAVE_OPENMP)

    # Try to locate Intel OpenMP with pkg-config.
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
      pkg_check_modules(IOMP_PKGCONFIG openmp)
    endif()

    # Try to find it with find_library.
	  find_library(iomp5_LIBRARY NAMES iomp5 libiomp5 HINTS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
                 ENV LD_LIBRARY_PATH ENV LIBRARY_PATH
                 ${IOMP_PKGCONFIG_LIBRARY_DIRS})

    if(iomp5_LIBRARY)
      # This is a workaround to use Intel OpenMP with GCC.
      # Add -liomp5 to interface link libraries for OpenMP
      set_property(TARGET ${OPENMP_C_PROPERTIES} PROPERTY
        INTERFACE_LINK_LIBRARIES ${iomp5_LIBRARY}
      )
      # and remove -fopenmp from the link options.
      set_property(TARGET ${OPENMP_C_PROPERTIES} PROPERTY
        INTERFACE_LINK_OPTIONS ""
      )
      message(STATUS "FORCE_INTEL_OPENMP: ${FORCE_INTEL_OPENMP}")
      message(STATUS "Linking Intel OpenMP library: ${iomp5_LIBRARY}")
    else()
      message(WARNING "You asked to link against libiomp5, but it wasn't found. Reverting to default OpenMP runtime provided by the compiler.")
    endif()

  else()
    message(WARNING "You asked to link against libiomp5, but didn't enable OpenMP.")
  endif()
endif()

cmake_print_properties(TARGETS ${OPENMP_C_PROPERTIES} PROPERTIES INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_LIBRARIES INTERFACE_LINK_OPTIONS)
