# Build the PDFT library

add_library (pdft SHARED nr_numint.c)
target_link_libraries (pdft dft ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
set_target_properties (pdft PROPERTIES
    CLEAN_DIRECT_OUTPUT 1
    OUTPUT_NAME "pdft")

list(APPEND PYSCF_TARGETS pdft)
set(PYSCF_TARGETS ${PYSCF_TARGETS} PARENT_SCOPE)