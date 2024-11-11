/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 */

#include <stdlib.h>
#include "np_helper/np_helper.h"

void NPdset0(double *p, const size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                p[i] = 0;
        }
}

void NPzset0(double complex *p, const size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                p[i] = 0;
        }
}

void NPdcopy(double *out, const double *in, const size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                out[i] = in[i];
        }
}

void NPzcopy(double complex *out, const double complex *in, const size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                out[i] = in[i];
        }
}


void NPomp_dcopy(const size_t m,
                 const size_t n,
                 const double *__restrict in, const size_t in_stride,
                 double *__restrict out, const size_t out_stride)
{
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < m; i++) {
#pragma omp simd
                for (size_t j = 0; j < n; j++) {
                        out[i * out_stride + j] = in[i * in_stride + j];
                }
        }
}

void NPomp_zcopy(const size_t m,
                 const size_t n,
                 const double complex *__restrict in, const size_t in_stride,
                 double complex *__restrict out, const size_t out_stride)
{
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < m; i++) {
#pragma omp simd
                for (size_t j = 0; j < n; j++) {
                        out[i * out_stride + j] = in[i * in_stride + j];
                }
        }
}

void NPomp_dmul(const size_t m,
                const size_t n,
                const double *__restrict a, const size_t a_stride,
                double *__restrict b, const size_t b_stride)
{
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < m; i++) {
#pragma omp simd
                for (size_t j = 0; j < n; j++) {
                        b[i * b_stride + j] *= a[i * a_stride + j];
                }
        }
}

void NPomp_zmul(const size_t m,
                const size_t n,
                const double complex *__restrict a, const size_t a_stride,
                double complex *__restrict b, const size_t b_stride)
{
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < m; i++) {
#pragma omp simd
                for (size_t j = 0; j < n; j++) {
                        b[i * b_stride + j] *= a[i * a_stride + j];
                }
        }
}


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