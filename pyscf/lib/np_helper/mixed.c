#include "config.h"
#include <complex.h>
#include <stdlib.h>

#include "np_helper/np_helper.h"

void NPomp_real_plus_imag(double complex *RESTRICT out,
                          const double *RESTRICT real,
                          const double *RESTRICT imag, const size_t n) {
  out = __builtin_assume_aligned(out, sizeof(double complex));
  real = __builtin_assume_aligned(real, sizeof(double));
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    out[i] = real[i] + imag[i] * I;
  }
}

void NPomp_extract_real(const double complex *RESTRICT in,
                        double *RESTRICT real, const size_t n) {
  in = __builtin_assume_aligned(in, sizeof(double complex));
  real = __builtin_assume_aligned(real, sizeof(double));
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    real[i] = creal(in[i]);
  }
}
void NPomp_extract_imag(const double complex *RESTRICT in,
                        double *RESTRICT imag, const size_t n) {
  in = __builtin_assume_aligned(in, sizeof(double complex));
  imag = __builtin_assume_aligned(imag, sizeof(double));
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    imag[i] = cimag(in[i]);
  }
}
void NPomp_promote_real(double complex *RESTRICT out,
                        const double *RESTRICT real, const size_t n) {
  out = __builtin_assume_aligned(out, sizeof(double complex));
  real = __builtin_assume_aligned(real, sizeof(double));
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    out[i] = real[i];
  }
}

void NPomp_promote_imag(double complex *RESTRICT out,
                        const double *RESTRICT imag, const size_t n) {
  out = __builtin_assume_aligned(out, sizeof(double complex));
  imag = __builtin_assume_aligned(imag, sizeof(double));
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    out[i] = imag[i] * I;
  }
}

void NPomp_axpy_zd(double complex *RESTRICT y, const double *RESTRICT x,
                   const double a_r, const double a_i, const size_t n) {
  y = __builtin_assume_aligned(y, sizeof(double complex));
  x = __builtin_assume_aligned(x, sizeof(double));
  const double complex a = a_r + a_i * I;

#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    y[i] += a * x[i];
  }
}