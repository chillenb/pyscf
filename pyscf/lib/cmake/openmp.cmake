set(OPENMP_C_PROPERTIES "")
if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS C)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

if(FORCE_INTEL_OPENMP)
  if(ENABLE_OPENMP AND HAVE_OPENMP)
	  find_library(iomp5_LIBRARY NAMES iomp5 libiomp5 HINTS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} ENV LD_LIBRARY_PATH ENV LIBRARY_PATH)
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
      message(WARNING "You asked to link against libiomp5, but it wasn't found. Switching to default OpenMP runtime provided by the compiler.")
    endif()

  else()
    message(WARNING "You asked to link against libiomp5, but didn't enable OpenMP.")
  endif()
endif()

cmake_print_properties(TARGETS ${OPENMP_C_PROPERTIES} PROPERTIES INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_LIBRARIES INTERFACE_LINK_OPTIONS)
