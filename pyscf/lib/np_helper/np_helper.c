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
#ifdef PYSCF_USE_MKL
#include "mkl.h"
#endif

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

void* pyscf_malloc(size_t alloc_size)
{
#ifdef PYSCF_USE_MKL
        return mkl_malloc(alloc_size, 64);
#else
        return malloc(alloc_size);
#endif
}

void *pyscf_calloc(size_t n, size_t size)
{
#ifdef PYSCF_USE_MKL
        return mkl_calloc(n, size, 64);
#else
        return calloc(n, size);
#endif
}

void pyscf_free(void *ptr)
{
#ifdef PYSCF_USE_MKL
        mkl_free(ptr);
#else
        free(ptr);
#endif
}

void *pyscf_realloc(void *ptr, size_t size)
{
#ifdef PYSCF_USE_MKL
        return mkl_realloc(ptr, size);
#else
        return realloc(ptr, size);
#endif
}

int pyscf_has_mkl(void) {
#ifdef PYSCF_USE_MKL
        return 1;
#else
        return 0;
#endif
}

void NPomp_dset0(double *p, const size_t n)
{
#ifndef PYSCF_USE_MKL
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
                p[i] = 0;
        }
#else
        LAPACKE_dlaset(LAPACK_COL_MAJOR, 'A', n, 1, 0.0, 0.0, p, n);
#endif
}

void NPomp_zset0(double complex *p, const size_t n)
{
#ifndef PYSCF_USE_MKL
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
                p[i] = 0;
        }
#else
        MKL_Complex16 zero = {0.0, 0.0};
        LAPACKE_zlaset(LAPACK_COL_MAJOR, 'A', n, 1, zero, zero, (MKL_Complex16*) p, n);
#endif
}

void NPomp_dmul(double *A, double *B, double *out, size_t n)
{
#ifndef PYSCF_USE_MKL
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
                out[i] = A[i] * B[i];
        }
#else
        vdMul(n, A, B, out);
#endif
}

void NPomp_zmul(double complex *A, double complex *B, double complex *out, size_t n)
{
#ifndef PYSCF_USE_MKL
        size_t i;
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
                out[i] = A[i] * B[i];
        }
#else
        vzMul(n, (MKL_Complex16*) A, (MKL_Complex16*) B, (MKL_Complex16*) out);
#endif
}