/*  Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
   
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
 *  Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
 */

#include <stdlib.h>
#include <complex.h>
#include "np_helper/np_helper.h"


/* computes g_ia = alpha * eia / (eia**2 + omega**2) */
void rho_kernel_restricted(const int nocc, const int nvir,
                           const double *mo_energy_occ, const double *mo_energy_vir,
                           double *__restrict out,
                           const double alpha, const double omega)
{
    double omega2 = omega * omega;
#pragma omp parallel for schedule(static)
    for(int i = 0; i < nocc; i++) {
        for(int j = 0; j < nvir; j++) {
            double eia = mo_energy_occ[i] - mo_energy_vir[j];
            out[i*nvir+j] = alpha * eia / (eia*eia + omega2);
        }
    }
}


/* computes Pia = Lia * gia[None, ...] */
void dmul_Lia_gia(const int naux, const int nocc, const int nvir,
                 double *__restrict Lpq, size_t lstride0, size_t lstride1,
                 const double *__restrict g_ia,
                 double *__restrict out, size_t ostride0, size_t ostride1)
{
#pragma omp parallel for schedule(static)
    for(size_t i = 0; i < naux; i++) {
        for(size_t j = 0; j < nocc; j++) {
#pragma omp simd
            for(size_t k = 0; k < nvir; k++) {
                out[i*ostride0 + j*ostride1 + k] = Lpq[i*lstride0 + j*lstride1 + k] * g_ia[j*nvir+k];
            }
        }
    }
}


/* computes Pia = Lia * gia[None, ...] */
void zmul_Lia_gia(const int naux, const int nocc, const int nvir,
                 double complex *__restrict Lpq, size_t lstride0, size_t lstride1,
                 const double *__restrict g_ia,
                 double complex *__restrict out, size_t ostride0, size_t ostride1)
{
#pragma omp parallel for schedule(static)
    for(size_t i = 0; i < naux; i++) {
        for(size_t j = 0; j < nocc; j++) {
#pragma omp simd
            for(size_t k = 0; k < nvir; k++) {
                out[i*ostride0 + j*ostride1 + k] = Lpq[i*lstride0 + j*lstride1 + k] * g_ia[j*nvir+k];
            }
        }
    }
}


/* computes Pia = alpha * Lia * eia / (eia**2 + omega**2) */
void dmul_Lia_response(const int naux, const int nocc, const int nvir,
                 double *__restrict Lpq, size_t lstride0, size_t lstride1,
                 double *mo_energy_occ, double *mo_energy_vir,
                 double *__restrict out, size_t ostride0, size_t ostride1,
                 double alpha, double omega)
{
    double *g_ia = (double*) malloc((size_t)nocc * (size_t)nvir * sizeof(double));
    rho_kernel_restricted(nocc, nvir, mo_energy_occ, mo_energy_vir, g_ia, alpha, omega);
    dmul_Lia_gia(naux, nocc, nvir,
                    Lpq, lstride0, lstride1,
                    g_ia,
                    out, ostride0, ostride1);
    free(g_ia);
}

/* computes Pia = alpha * Lia * eia / (eia**2 + omega**2) */
void zmul_Lia_response(const int naux, const int nocc, const int nvir,
                 double complex *__restrict Lpq, size_t lstride0, size_t lstride1,
                 double *mo_energy_occ, double *mo_energy_vir,
                 double complex *__restrict out, size_t ostride0, size_t ostride1,
                 double alpha, double omega)
{
    double *g_ia = (double*) malloc((size_t)nocc * (size_t)nvir * sizeof(double));
    rho_kernel_restricted(nocc, nvir, mo_energy_occ, mo_energy_vir, g_ia, alpha, omega);
    zmul_Lia_gia(naux, nocc, nvir,
                    Lpq, lstride0, lstride1,
                    g_ia,
                    out, ostride0, ostride1);
    free(g_ia);
}
