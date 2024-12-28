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
#include <complex.h>
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

/*
 * These are mostly useful for first-touch array allocation on NUMA systems.
 * Use with numpy.empty.
 */
void NPomp_dset0(const size_t n, double *out)
{
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
                out[i] = 0.0;
        }
}

void NPomp_zset0(const size_t n, double complex *out)
{
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
                out[i] = 0.0;
        }
}


/*
 * Copy a double precision matrix with multithreading.
 */
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

/*
 * Copy a complex double precision matrix with multithreading.
 */
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

/*
 * Elementwise multiplication of two double matrices.
 * B <- A \circ B
 */
void NPomp_dmul(const size_t m,
                const size_t n,
                const double *__restrict a, const size_t a_stride,
                double *__restrict b, const size_t b_stride,
                double *__restrict out, const size_t out_stride)
{
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < m; i++) {
#pragma omp simd
                for (size_t j = 0; j < n; j++) {
                        out[i * out_stride + j] = b[i * b_stride + j] * a[i * a_stride + j];
                }
        }
}

/*
 * Elementwise multiplication of two complex double matrices.
 * B <- A \circ B
 */
void NPomp_zmul(const size_t m,
                const size_t n,
                const double complex *__restrict a, const size_t a_stride,
                double complex *__restrict b, const size_t b_stride,
                double complex *__restrict out, const size_t out_stride)
{
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < m; i++) {
#pragma omp simd
                for (size_t j = 0; j < n; j++) {
                        out[i * out_stride + j] = b[i * b_stride + j] * a[i * a_stride + j];
                }
        }
}

/*
 * Copy a double precision 3D tensor with multithreading.
 * (ishape0, ishape1, ishape2) describes the shape of the input tensor.
 * The input tensor and output tensors are both assumed to be stored in C-order, i.e.
 * T[i, j, k] == T[i*istride0 + j*istride1 + k].
 */
void NPomp_dcopy_012(const size_t ishape0, const size_t ishape1, const size_t ishape2,
                    const double *__restrict in, const size_t istride0, const size_t istride1,
                    double *__restrict out, const size_t ostride0, const size_t ostride1)
{
#pragma omp parallel for schedule(static)
    for(size_t i = 0; i < ishape0; i++) {
        for(size_t j = 0; j < ishape1; j++) {
#pragma omp simd
            for(size_t k = 0; k < ishape2; k++) {
                out[i*ostride0 + j*ostride1 + k] = in[i*istride0 + j*istride1 + k];
            }
        }
    }
}

/*
 * Copy a complex double 3D tensor with multithreading.
 * (ishape0, ishape1, ishape2) describes the shape of the input tensor.
 * The input tensor and output tensors are both assumed to be stored in C-order, i.e.
 * T[i, j, k] == T[i*istride0 + j*istride1 + k].
 *
 * The output tensor is conjugated if conja is not 0.
 */
void NPomp_zcopy_012(const int conja,
                    const size_t ishape0, const size_t ishape1, const size_t ishape2,
                    const double complex *__restrict in, const size_t istride0, const size_t istride1,
                    double complex *__restrict out, const size_t ostride0, const size_t ostride1)
{
    if (!conja) {
#pragma omp parallel for schedule(static)
        for(size_t i = 0; i < ishape0; i++) {
            for(size_t j = 0; j < ishape1; j++) {
#pragma omp simd
                for(size_t k = 0; k < ishape2; k++) {
                    out[i*ostride0 + j*ostride1 + k] = in[i*istride0 + j*istride1 + k];
                }
            }
        }
    } else {
#pragma omp parallel for schedule(static)
        for(size_t i = 0; i < ishape0; i++) {
            for(size_t j = 0; j < ishape1; j++) {
#pragma omp simd
                for(size_t k = 0; k < ishape2; k++) {
                    out[i*ostride0 + j*ostride1 + k] = conj(in[i*istride0 + j*istride1 + k]);
                }
            }
        }
    }
}