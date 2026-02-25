# Build the PDFT library

add_library (pdft SHARED nr_numint.c)
target_link_libraries (pdft dft ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
set_target_properties (pdft PROPERTIES
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "pdft")
target_compile_options (pdft PRIVATE ${pyscf_cflags})