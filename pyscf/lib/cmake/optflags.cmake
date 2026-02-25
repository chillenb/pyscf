include(CheckCCompilerFlag)

list(APPEND pyscf_cflags
  -Wall

  # options for Clang and GCC
  $<$<C_COMPILER_ID:GNU,Clang>:$<$<BOOL:${BUILD_MARCH_NATIVE}>:-march=native>>
  $<$<C_COMPILER_ID:GNU,Clang>:$<$<BOOL:${BUILD_MARCH_NATIVE}>:-ftree-vectorize>>
  $<$<C_COMPILER_ID:GNU,Clang>:$<$<BOOL:${BUILD_MARCH_NATIVE}>:-funroll-loops>>
  $<$<C_COMPILER_ID:GNU,Clang>:$<$<BOOL:${BUILD_MARCH_NATIVE}>:-fno-math-errno>>

  # options for icc (Intel Classic) and icx (Intel LLVM)
  $<$<C_COMPILER_ID:Intel,IntelLLVM>:$<$<BOOL:${BUILD_MARCH_NATIVE}>:-xHost>>
  $<$<C_COMPILER_ID:Intel>:$<$<BOOL:${BUILD_MARCH_NATIVE}>:-unroll-aggressive>>
  $<$<C_COMPILER_ID:IntelLLVM>:$<$<BOOL:${BUILD_MARCH_NATIVE}>:-unroll>>
  $<$<C_COMPILER_ID:Intel,IntelLLVM>:$<$<BOOL:${BUILD_MARCH_NATIVE}>:ipo>>

  # Avoids error "‘SIMDD’ undeclared here (not in a function)" in the qcint two-electron integral interface
  $<$<C_COMPILER_ID:GNU,Clang>:$<$<NOT:$<BOOL:${BUILD_MARCH_NATIVE}>>:-msse3>>
)

message(STATUS "C compiler flags: ${pyscf_cflags}")