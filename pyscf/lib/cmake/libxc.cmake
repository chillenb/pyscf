find_package(Libxc QUIET)

if(Libxc_FOUND)
  message(STATUS "Found preinstalled LibXC")
  cmake_print_properties(TARGETS Libxc::xc PROPERTIES LOCATION INTERFACE_LINK_LIBRARIES)

elseif(BUILD_LIBXC)

  message(STATUS "LibXC not found, fetching and building the source.")

  set(LIBXC_INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps)
  include(ExternalProject)
  ExternalProject_Add(libxc
      #GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
      #GIT_TAG master
      URL https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.gz
      PREFIX ${PROJECT_BINARY_DIR}/deps
      INSTALL_DIR ${LIBXC_INSTALL_DIR}
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=1
              -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
              -DCMAKE_INSTALL_LIBDIR:PATH=lib
              -DENABLE_FORTRAN=0 -DDISABLE_KXC=1 -DDISABLE_LXC=1
              -DENABLE_XHOST:STRING=${BUILD_MARCH_NATIVE}
              -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 # remove when libxc update version min in next release
    )

  # Make an imported library target for libxc.
  add_library(xc SHARED IMPORTED)
  set_target_properties(xc PROPERTIES IMPORTED_LOCATION ${LIBXC_INSTALL_DIR}/lib/libxc${CMAKE_SHARED_LIBRARY_SUFFIX})
  # Tell CMake that the imported library is built by the external project.
  add_dependencies(xc libxc)
  target_include_directories(xc INTERFACE ${LIBXC_INSTALL_DIR}/include)
  # libxc_itrf will link to Libxc::xc. This is what find_package would have provided.
  add_library(Libxc::xc ALIAS xc)

else()
  message(FATAL_ERROR "LibXC not found and BUILD_LIBXC is OFF. Please install LibXC or set BUILD_LIBXC to ON.")
endif()