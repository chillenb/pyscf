# Add + to the Make recipe line if CMake is new enough.
# This enables parallel compilation of the ExternalProjects that are
# built using Make.
# It does nothing if the generator isn't Make.
# Tested with CMake<3.28 and >=3.28
set(PYSCF_JOBSERVER_AWARE_ARG "")
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28.0")
  set(PYSCF_JOBSERVER_AWARE_ARG JOB_SERVER_AWARE TRUE)
endif()
