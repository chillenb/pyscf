/* Copyright 2021- The PySCF Developers. All Rights Reserved.

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
 * Author: Xing Zhang <zhangxing.nju@gmail.com>
 */

#include <complex.h>
#include <omp.h>
#include <fftw3.h>
#include <stdio.h>



#include "fft.h"
#include "config.h"


// https://www.fftw.org/fftw3_doc/Usage-of-Multi_002dthreaded-FFTW.html
// Allows FFTW to use the same OpenMP runtime as PySCF, no matter
// how it was built.
static void openmp_callback_for_fftw(void *(*work)(char *), char *jobdata, size_t elsize, int njobs, void *data)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < njobs; ++i)
        work(jobdata + elsize * i);
}

fftw_plan fft_create_r2c_plan(double* in, double complex *out, int rank, int* mesh)
{
    fftw_plan p;
    p = fftw_plan_dft_r2c(rank, mesh, in, out, FFTW_ESTIMATE);
    return p;
}

fftw_plan fft_create_c2r_plan(double complex* in, double* out, int rank, int* mesh)
{
    fftw_plan p;
    p = fftw_plan_dft_c2r(rank, mesh, in, out, FFTW_ESTIMATE);
    return p;
}

void fft_execute(fftw_plan p)
{
    fftw_execute(p);
}

void fft_destroy_plan(fftw_plan p)
{
    fftw_destroy_plan(p);
}

void _complex_fft(complex double* in, complex double* out, int* mesh, int rank, int sign)
{
    fftw_init_threads();
    fftw_threads_set_callback(&openmp_callback_for_fftw, (void *) 0);
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_plan p;

    switch(rank) {
        case 1:
            p = fftw_plan_dft_1d(mesh[0], in, out, sign, FFTW_ESTIMATE);
            break;
        case 2:
            p = fftw_plan_dft_2d(mesh[0], mesh[1], in, out, sign, FFTW_ESTIMATE);
            break;
        case 3:
            p = fftw_plan_dft_3d(mesh[0], mesh[1], mesh[2], in, out, sign, FFTW_ESTIMATE);
            break;
        default:
            p = fftw_plan_dft(rank, mesh, in, out, sign, FFTW_ESTIMATE);
    }
    fftw_execute(p);
    fftw_destroy_plan(p);
}

void fft(complex double* in, complex double* out, int* mesh, int rank)
{
    _complex_fft(in, out, mesh, rank, FFTW_FORWARD);
}

void ifft(complex double* in, complex double* out, int* mesh, int rank)
{
    _complex_fft(in, out, mesh, rank, FFTW_BACKWARD);
    size_t i, n = 1;
    for (i = 0; i < rank; i++) {
        n *= mesh[i];
    }
    double fac = 1. / (double)n;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] *= fac;
    }
}

void rfft(double* in, double complex* out, int* mesh, int rank)
{
    fftw_plan p = fftw_plan_dft_r2c(rank, mesh, in, out, FFTW_ESTIMATE); 
    fftw_execute(p);
    fftw_destroy_plan(p);
}

void irfft(double complex* in, double* out, int* mesh, int rank)
{
    fftw_plan p = fftw_plan_dft_c2r(rank, mesh, in, out, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    size_t i, n = 1;
    for (i = 0; i < rank; i++) {
        n *= mesh[i];
    }
    double fac = 1. / (double)n;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] *= fac;
    }
}

void _copy_d2z(double complex *out, const double *in, const size_t n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = in[i] + 0*_Complex_I;
    }
}
}
