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
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include <omp.h>

#define MAX_THREADS     256


/* 
 *  PiI_PQ, Lpqorbs_Pmn, Lpqorbs_Qml -> mnl
 *  where P and Q have dimension naux,
 *  m has dimension nmo, and n and l have dimension norb.
 */
void gwgf_contract(double *PiI, double *Lpqorbs, double *Wmn,
                   int naux, int nmo, int norb,
                   int lstride0, int lstride1,
                   int wstride0)
{
    const double D0 = 0.0;
    const double D1 = 1.0;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const char SIDE = 'R';
    const char UPLO = 'L';
#pragma omp parallel
{
    double *qslice_buf = calloc(naux*norb, sizeof(double));
#pragma omp for schedule(static)
    for(int i = 0; i < nmo; i++) {
        // dgemm_(&TRANS_N, &TRANS_N, &norb, &naux, &naux,
        //         &D1,
        //         Lpqorbs + i*norb, &ldb_lpqorbs,
        //         PiI, &naux,
        //         &D0, qslice_buf, &norb);
        dsymm_(&SIDE, &UPLO, &norb, &naux,
                &D1,
                &PiI, &naux,
                Lpqorbs + i*lstride1, &lstride0,
                &D0, qslice_buf, &norb);
        dgemm_(&TRANS_N, &TRANS_T, &norb, &norb, &naux,
                &D1,
                qslice_buf, &norb,
                Lpqorbs + i*lstride1, &lstride0,
                &D0, Wmn + i*wstride0, &norb);
    }
    free(qslice_buf);
}
}

//jLa, Lji -> ia
void bse_contract_a(double *jLa, double *Lii_bar, double *Wia,
                    int naux, int nocc, int nvir,
                    int jlastride0, int jlastride1,
                    int liibarstride0, int liibarstride1,
                    double alpha)
{
    const double D0 = 0.0;
    const double D1 = 1.0;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    double *wiabufs[MAX_THREADS];

#pragma omp parallel
{
    double *wia_buf;
    if(omp_get_thread_num() != 0)
        wia_buf = calloc(nocc*nvir, sizeof(double));
    else
        wia_buf = Wia;
    wiabufs[omp_get_thread_num()] = wia_buf;

#pragma omp for schedule(static)
    for(int j = 0; j < nocc; j++) {
        dgemm_(&TRANS_N, &TRANS_T,
                &nvir, &nocc, &naux,
                &alpha,
                jLa + j*jlastride0, &jlastride1,
                Lii_bar + j*liibarstride1, &liibarstride0,
                &D1, wia_buf, &nvir);
    }
    NPomp_dsum_reduce_inplace(wiabufs, nocc*nvir);
    if(omp_get_thread_num() != 0)
        free(wia_buf);
}
}