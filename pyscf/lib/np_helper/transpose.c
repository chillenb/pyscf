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

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <complex.h>
#include "np_helper.h"

#ifdef PYSCF_USE_MKL
#include "mkl.h"
#endif

/*
 * matrix a[n,m], out of place transpose
 */
void NPdtranspose(int n, int m, double *a, double *at)
{
#ifndef PYSCF_USE_MKL
        size_t i, j, j0, j1;
        for (j0 = 0; j0 < n; j0+=BLOCK_DIM) {
                j1 = MIN(j0+BLOCK_DIM, n);
                for (i = 0; i < m; i++) {
                        for (j = j0; j < j1; j++) {
                                at[i*n+j] = a[j*m+i];
                        }
                }
        }
#else
        int dynamic = mkl_get_dynamic();
        mkl_set_dynamic(1);
        mkl_domatcopy('R', 'T', n, m, 1.0, a, m, at, n);
        mkl_set_dynamic(dynamic);
#endif
}

void NPztranspose(int n, int m, double complex *a, double complex *at)
{
#ifndef PYSCF_USE_MKL
        size_t i, j, j0, j1;
        for (j0 = 0; j0 < n; j0+=BLOCK_DIM) {
                j1 = MIN(j0+BLOCK_DIM, n);
                for (i = 0; i < m; i++) {
                        for (j = j0; j < j1; j++) {
                                at[i*n+j] = a[j*m+i];
                        }
                }
        }
#else
        int dynamic = mkl_get_dynamic();
        mkl_set_dynamic(1);
        const MKL_Complex16 one = {1.0, 0.0};
        mkl_zomatcopy('R', 'T', n, m, one, (MKL_Complex16 *)a, m,
                      (MKL_Complex16 *)at, n);
        mkl_set_dynamic(dynamic);
#endif
}


void NPdtranspose_021(int *shape, double *a, double *at)
{
#ifndef PYSCF_USE_MKL
#pragma omp parallel default(none) \
        shared(shape, a, at)
{
        int ic;
        size_t nm = shape[1] * shape[2];
#pragma omp for schedule (static)
        for (ic = 0; ic < shape[0]; ic++) {
                NPdtranspose(shape[1], shape[2], a+ic*nm, at+ic*nm);
        }
}
#else
        int dynamic = mkl_get_dynamic();
        mkl_set_dynamic(1);
        int nslice = shape[0];
        int nrows = shape[1];
        int ncols = shape[2];
        mkl_domatcopy_batch_strided('R', 'T', nrows, ncols, 1.0, a, ncols,
                                    nrows * ncols, at, nrows, nrows * ncols,
                                    nslice);
        mkl_set_dynamic(dynamic);
#endif
}

void NPztranspose_021(int *shape, double complex *a, double complex *at)
{
#ifndef PYSCF_USE_MKL
#pragma omp parallel default(none) \
        shared(shape, a, at)
{
        int ic;
        size_t nm = shape[1] * shape[2];
#pragma omp for schedule (static)
        for (ic = 0; ic < shape[0]; ic++) {
                NPztranspose(shape[1], shape[2], a+ic*nm, at+ic*nm);
        }
}
#else
        int dynamic = mkl_get_dynamic();
        mkl_set_dynamic(1);
        int nslice = shape[0];
        int nrows = shape[1];
        int ncols = shape[2];
        const MKL_Complex16 one = {1.0, 0.0};
        mkl_zomatcopy_batch_strided(
            'R', 'T', nrows, ncols, one, (MKL_Complex16 *)a, ncols,
            nrows * ncols, (MKL_Complex16 *)at, nrows, nrows * ncols, nslice);
        mkl_set_dynamic(dynamic);
#endif
}


void NPdsymm_sum(int n, double *a, double *out, int hermi)
{
        size_t i, j, j0, j1;
        double tmp;

        if (hermi == HERMITIAN || hermi == SYMMETRIC) {
                TRIU_LOOP(i, j) {
                        tmp = a[i*n+j] + a[j*n+i];
                        out[i*n+j] = tmp;
                        out[j*n+i] = tmp;
                }
        } else {
                TRIU_LOOP(i, j) {
                        tmp = a[i*n+j] - a[j*n+i];
                        out[i*n+j] = tmp;
                        out[j*n+i] =-tmp;
                }
        }
}

void NPzhermi_sum(int n, double complex *a, double complex *out, int hermi)
{
        size_t i, j, j0, j1;
        double complex tmp;

        if (hermi == HERMITIAN) {
                TRIU_LOOP(i, j) {
                        tmp = a[i*n+j] + conj(a[j*n+i]);
                        out[i*n+j] = tmp;
                        out[j*n+i] = conj(tmp);
                }
        } else if (hermi == SYMMETRIC) {
                TRIU_LOOP(i, j) {
                        tmp = a[i*n+j] + a[j*n+i];
                        out[i*n+j] = tmp;
                        out[j*n+i] = tmp;
                }
        } else {
                TRIU_LOOP(i, j) {
                        tmp = a[i*n+j] - conj(a[j*n+i]);
                        out[i*n+j] = tmp;
                        out[j*n+i] =-conj(tmp);
                }
        }
}


void NPdsymm_021_sum(int *shape, double *a, double *out, int hermi)
{
#pragma omp parallel default(none) \
        shared(shape, a, out, hermi)
{
        int ic;
        size_t nn = shape[1] * shape[1];
#pragma omp for schedule (static)
        for (ic = 0; ic < shape[0]; ic++) {
                NPdsymm_sum(shape[1], a+ic*nn, out+ic*nn, hermi);
        }
}
}

void NPzhermi_021_sum(int *shape, double complex *a, double complex *out, int hermi)
{
#pragma omp parallel default(none) \
        shared(shape, a, out, hermi)
{
        int ic;
        size_t nn = shape[1] * shape[1];
#pragma omp for schedule (static)
        for (ic = 0; ic < shape[0]; ic++) {
                NPzhermi_sum(shape[1], a+ic*nn, out+ic*nn, hermi);
        }
}
}
