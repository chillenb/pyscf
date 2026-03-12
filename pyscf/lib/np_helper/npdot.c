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
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))

/*
 * numpy.dot may call unoptimized blas
 */
void NPdgemm(const char trans_a, const char trans_b,
             const int m, const int n, const int k,
             const int lda, const int ldb, const int ldc,
             const int offseta, const int offsetb, const int offsetc,
             double *a, double *b, double *c,
             const double alpha, const double beta)
{
        const size_t Ldc = ldc;
        int i, j;
        if (m == 0 || n == 0) {
                return;
        } else if (k == 0) {
                for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                        c[i*Ldc+j] = 0;
                } }
                return;
        }
        a += offseta;
        b += offsetb;
        c += offsetc;

        dgemm_(&trans_a, &trans_b, &m, &n, &k,
                &alpha, a, &lda, b, &ldb,
                &beta, c, &ldc);

}


void NPzgemm(const char trans_a, const char trans_b,
             const int m, const int n, const int k,
             const int lda, const int ldb, const int ldc,
             const int offseta, const int offsetb, const int offsetc,
             double complex *a, double complex *b, double complex *c,
             const double complex *alpha, const double complex *beta)
{
        const size_t Ldc = ldc;
        int i, j;
        if (m == 0 || n == 0) {
                return;
        } else if (k == 0) {
                for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                        c[i*Ldc+j] = 0;
                } }
                return;
        }
        a += offseta;
        b += offsetb;
        c += offsetc;

        zgemm_(&trans_a, &trans_b, &m, &n, &k,
                alpha, a, &lda, b, &ldb,
                beta, c, &ldc);

}
