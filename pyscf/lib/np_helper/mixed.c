#include "config.h"
#include <complex.h>
#include <stdlib.h>

#include "np_helper/np_helper.h"

void NPomp_real_plus_imag(const size_t m,
                          const size_t n,
                          const double *__restrict real,
                          const double *__restrict imag,
                          const size_t in_stride,
                          double complex *__restrict out,
                          const size_t out_stride) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m; i++) {
#pragma omp simd
    for(size_t j = 0; j < n; j++) {
      out[i * out_stride + j] = real[i * in_stride + j] + imag[i * in_stride + j] * _Complex_I;
    }
  }
}

void NPomp_extract_real(const size_t m,
                        const size_t n,
                        const double complex *__restrict in,
                        const size_t in_stride,
                        double *__restrict real,
                        const size_t out_stride) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m; i++) {
#pragma omp simd
    for(size_t j = 0; j < n; j++) {
      real[i * out_stride + j] = creal(in[i * in_stride + j]);
    }
  }
}

void NPomp_extract_imag(const size_t m,
                        const size_t n,
                        const double complex *__restrict in,
                        const size_t in_stride,
                        double *__restrict imag,
                        const size_t out_stride) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m; i++) {
#pragma omp simd
    for(size_t j = 0; j < n; j++) {
      imag[i * out_stride + j] = cimag(in[i * in_stride + j]);
    }
  }
}

void NPomp_promote_real(const size_t m,
                        const size_t n,
                        const double *__restrict real,
                        const size_t in_stride,
                        double complex *__restrict out,
                        const size_t out_stride) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m; i++) {
#pragma omp simd
    for(size_t j = 0; j < n; j++) {
      out[i * out_stride + j] = real[i * in_stride + j];
    }
  }
}

void NPomp_promote_imag(const size_t m,
                        const size_t n,
                        const double *__restrict imag,
                        const size_t in_stride,
                        double complex *__restrict out,
                        const size_t out_stride) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m; i++) {
#pragma omp simd
    for(size_t j = 0; j < n; j++) {
      out[i * out_stride + j] = imag[i * in_stride + j];
    }
  }
}

void NPomp_axpy_zd(const size_t n, const double complex *alphaptr,
                   const double *__restrict x, double complex *__restrict y) {
  const double complex alpha = *alphaptr;
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    y[i] += alpha * x[i];
  }
}