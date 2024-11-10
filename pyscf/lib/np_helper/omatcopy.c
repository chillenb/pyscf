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
 * Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
 */

#include <complex.h>
#include "np_helper.h"

/*
 * Double prec transpose-scale is done on 6x8 tiles.
 * Complex version is done on 3x4 tiles.
 *
 *       o o o o o o o o
 *       o o o o o o o o          o o o o
 *       o o o o o o o o          o o o o
 *       o o o o o o o o          o o o o
 *       o o o o o o o o
 *       o o o o o o o o
 *
 */

const int TILE_COL = 8;
const int TILE_ROW = 6;
const int TILE_COL_CPLX = 6;
const int TILE_ROW_CPLX = 4;

/*
 * Substitute for mkl_domatcopy. Performs the operation
 * B <- alpha * op(A), where op(A) is either A or A^T,
 * and A is a row-major matrix of size m x n.
 */
void NPomp_d_otranspose_scale(char trans, const int m, const int n, const double alpha,
                              double* __restrict A, const int lda,
                              double* __restrict B, const int ldb)
{
  int m_b, n_b;
  if (trans == 'N') {
    m_b = m; n_b = n;
  } else {
    m_b = n; n_b = m;
  }
  const size_t lda_w = (size_t) lda;
  const size_t ldb_w = (size_t) ldb;

  if(alpha == 0.0) { // if alpha is 0, then fill B with zeros
    NPomp_dset(m_b, n_b, alpha, B, ldb);
  }
  else if(trans == 'N') { // matrix scale-copy
#pragma omp parallel for schedule(static)
    for(int i = 0; i < m; i++) {
#pragma omp simd
      for(int j = 0; j < n; j++) {
        B[i * ldb_w + j] = alpha * A[i * lda_w + j];
      }
    } // end parallel region
  } else { // matrix scale-transpose
    const int mtile_a = m / TILE_ROW;
    const int ntile_a = n / TILE_COL;

#pragma omp parallel
    {
#pragma omp for schedule(static) collapse(2) nowait
      for(int ii = 0; ii < mtile_a * TILE_ROW; ii+=TILE_ROW) {
        for(int jj = 0; jj < ntile_a * TILE_COL; jj+=TILE_COL) {
#pragma GCC unroll 6
#pragma GCC ivdep
          for(int i = ii; i < ii + TILE_ROW; i++) {
#pragma omp simd
            for(int j = jj; j < jj + TILE_COL; j++) {
              B[j * ldb_w + i] = alpha * A[i * lda_w + j];
            }
          }
        }
      }

#pragma omp for schedule(static) collapse(2) nowait
      for(int i = mtile_a * TILE_ROW; i < m; i++) {
        for(int j = 0; j < n; j++) {
          B[j * ldb_w + i] = alpha * A[i * lda_w + j];
        }
      }

#pragma omp for schedule(static) collapse(2) nowait
      for(int j = ntile_a * TILE_COL; j < n; j++) {
        for(int i = 0; i < m; i++) {
          B[j * ldb_w + i] = alpha * A[i * lda_w + j];
        }
      }
    } // end parallel region
    for(int i = mtile_a * TILE_ROW; i < m; i++) {
      for(int j = ntile_a * TILE_COL; j < n; j++) {
        B[j * ldb_w + i] = alpha * A[i * lda_w + j];
      }
    }
  }
}



/*
 * Substitute for mkl_domatcopy. Performs the operation
 * B <- alpha * op(A), where op(A) is either A or A^T,
 * and A is a row-major matrix of size m x n.
 */
void NPomp_z_otranspose_scale(char trans, const int m, const int n,
                              const double complex *alphaptr,
                              double complex *__restrict A, const int lda,
                              double complex *__restrict B, const int ldb)
{
  int m_b, n_b;
  const double complex alpha = *alphaptr;
  if (trans == 'N') {
    m_b = m; n_b = n;
  } else {
    m_b = n; n_b = m;
  }
  const size_t lda_w = (size_t) lda;
  const size_t ldb_w = (size_t) ldb;

  if(alpha == 0.0) { // if alpha is 0, then fill B with zeros
    NPomp_zset(m_b, n_b, &alpha, B, ldb);
  }
  else if(trans == 'N') { // matrix scale-copy
#pragma omp parallel for schedule(static)
    for(int i = 0; i < m; i++) {
#pragma omp simd
      for(int j = 0; j < n; j++) {
        B[i * ldb_w + j] = alpha * A[i * lda_w + j];
      }
    } // end parallel region
  } else { // matrix scale-transpose
    const int mtile_a = m / TILE_ROW_CPLX;
    const int ntile_a = n / TILE_COL_CPLX;

#pragma omp parallel
    {
#pragma omp for schedule(static) collapse(2) nowait
      for(int ii = 0; ii < mtile_a * TILE_ROW_CPLX; ii+=TILE_ROW_CPLX) {
        for(int jj = 0; jj < ntile_a * TILE_COL_CPLX; jj+=TILE_COL_CPLX) {
#pragma GCC unroll 4
#pragma GCC ivdep
          for(int i = ii; i < ii + TILE_ROW_CPLX; i++) {
#pragma omp simd
            for(int j = jj; j < jj + TILE_COL_CPLX; j++) {
              B[j * ldb_w + i] = alpha * A[i * lda_w + j];
            }
          }
        }
      }

#pragma omp for schedule(static) collapse(2) nowait
      for(int i = mtile_a * TILE_ROW_CPLX; i < m; i++) {
        for(int j = 0; j < n; j++) {
          B[j * ldb_w + i] = alpha * A[i * lda_w + j];
        }
      }

#pragma omp for schedule(static) collapse(2) nowait
      for(int j = ntile_a * TILE_COL_CPLX; j < n; j++) {
        for(int i = 0; i < m; i++) {
          B[j * ldb_w + i] = alpha * A[i * lda_w + j];
        }
      }
    } // end parallel region
    for(int i = mtile_a * TILE_ROW_CPLX; i < m; i++) {
      for(int j = ntile_a * TILE_COL_CPLX; j < n; j++) {
        B[j * ldb_w + i] = alpha * A[i * lda_w + j];
      }
    }
  }
}
