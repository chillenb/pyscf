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

int pyscf_has_mkl(void) {
#ifdef PYSCF_USE_MKL
        return 1;
#else
        return 0;
#endif
}