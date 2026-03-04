find_package(XCFun QUIET)
if(XCFun_FOUND)
  message(STATUS "Found preinstalled XCFun")
  cmake_print_properties(TARGETS XCFun::xcfun PROPERTIES LOCATION INTERFACE_LINK_LIBRARIES)

elseif(BUILD_XCFUN)
  include(ExternalProject)
  enable_language(CXX)
  ExternalProject_Add(libxcfun
    GIT_REPOSITORY https://github.com/dftlibs/xcfun.git
    GIT_TAG a89b783
    # This patch adds the xcfun derivative order up to 5
    PATCH_COMMAND git apply --reject ${PROJECT_SOURCE_DIR}/libxcfun.patch || true
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=1
            -DXCFUN_MAX_ORDER=${XCFUN_MAX_ORDER}
            -DENABLE_TESTALL=0
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  )

  # Make an imported library target for xcfun.
  add_library(xcfun SHARED IMPORTED)
  set_target_properties(xcfun PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/deps/lib/libxcfun${CMAKE_SHARED_LIBRARY_SUFFIX})
  # Tell CMake that the imported library is built by the external project.
  add_dependencies(xcfun libxcfun)
  target_include_directories(xcfun INTERFACE ${PROJECT_SOURCE_DIR}/deps/include)
  # libxcfun_itrf will link to XCFun::xcfun. This is what find_package would have provided.
  add_library(XCFun::xcfun ALIAS xcfun)

else()
  message(FATAL_ERROR "XCFun not found and BUILD_XCFUN is OFF. Please install XCFun or set BUILD_XCFUN to ON.")
endif()