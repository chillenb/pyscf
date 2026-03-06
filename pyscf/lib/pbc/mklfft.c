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
 * Author: Chris Hillenbrand <chillenbrand15@gmail.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "mkl_dfti.h"

#include "config.h"


DFTI_DESCRIPTOR_HANDLE fft_create_r2c_plan(int rank, int *mesh)
{
    if(rank < 1 || rank > 3) {
        fprintf(stderr, "FFT rank %d not supported\n", rank);
        exit(1);
    }

    DFTI_DESCRIPTOR_HANDLE fft_descriptor = NULL;
    MKL_LONG status;
    MKL_LONG dimensions[3];
    size_t total_size = 1;
    for (int i = 0; i < rank; i++) {
        dimensions[i] = (MKL_LONG) mesh[i];
        total_size *= (size_t) mesh[i];
    }

    if(rank > 1) {
      status = DftiCreateDescriptor(&fft_descriptor, DFTI_DOUBLE,
                                    DFTI_REAL, rank, dimensions);
    } else {
      status = DftiCreateDescriptor(&fft_descriptor, DFTI_DOUBLE,
                                    DFTI_REAL, 1, dimensions[0]);
    }

    status = DftiSetValue(fft_descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);

    // Backward transforms should be scaled by 1/N.
    status = DftiSetValue(fft_descriptor, DFTI_BACKWARD_SCALE, 1.0 / (double) total_size);

    status = DftiCommitDescriptor(fft_descriptor);
    if (status && !DftiErrorClass(status, DFTI_NO_ERROR))
    {
      fprintf(stderr, "Error: %s\n", DftiErrorMessage(status));
    }
    return fft_descriptor;
}

DFTI_DESCRIPTOR_HANDLE fft_create_c2c_plan(int rank, int *mesh)
{
    if(rank < 1 || rank > 3) {
        fprintf(stderr, "FFT rank %d not supported\n", rank);
        exit(1);
    }

    DFTI_DESCRIPTOR_HANDLE fft_descriptor = NULL;
    MKL_LONG status;
    MKL_LONG dimensions[3];
    size_t total_size = 1;
    for (int i = 0; i < rank; i++) {
        dimensions[i] = (MKL_LONG) mesh[i];
        total_size *= (size_t) mesh[i];
    }

    if(rank > 1) {
      status = DftiCreateDescriptor(&fft_descriptor, DFTI_DOUBLE,
                                    DFTI_COMPLEX, rank, dimensions);
    } else {
      status = DftiCreateDescriptor(&fft_descriptor, DFTI_DOUBLE,
                                    DFTI_COMPLEX, 1, dimensions[0]);
    }

    status = DftiSetValue(fft_descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);

    // Backward transforms should be scaled by 1/N.
    status = DftiSetValue(fft_descriptor, DFTI_BACKWARD_SCALE, 1.0 / (double) total_size);

    status = DftiCommitDescriptor(fft_descriptor);
    if (status && !DftiErrorClass(status, DFTI_NO_ERROR))
    {
      fprintf(stderr, "Error: %s\n", DftiErrorMessage(status));
    }
    return fft_descriptor;
}


void fft(complex double* in, complex double* out, int* mesh, int rank)
{
    DFTI_DESCRIPTOR_HANDLE fft_descriptor = fft_create_c2c_plan(rank, mesh);
    MKL_LONG status;

    status = DftiComputeForward(fft_descriptor, in, out);
    DftiFreeDescriptor(&fft_descriptor);
}

void ifft(complex double* in, complex double* out, int* mesh, int rank)
{
    DFTI_DESCRIPTOR_HANDLE fft_descriptor = fft_create_c2c_plan(rank, mesh);
    MKL_LONG status;

    status = DftiComputeBackward(fft_descriptor, in, out);
    // No scaling is needed for the backward transform.
    // We already set the backward scale factor to 1/N.
    DftiFreeDescriptor(&fft_descriptor);
}

void rfft(double* in, complex double* out, int* mesh, int rank)
{
    DFTI_DESCRIPTOR_HANDLE fft_descriptor = fft_create_r2c_plan(rank, mesh);
    MKL_LONG status;
    status = DftiComputeForward(fft_descriptor, in, out);

    DftiFreeDescriptor(&fft_descriptor);
}

void irfft(complex double* in, double* out, int* mesh, int rank)
{
    DFTI_DESCRIPTOR_HANDLE fft_descriptor = fft_create_r2c_plan(rank, mesh);
    MKL_LONG status;
    status = DftiComputeBackward(fft_descriptor, in, out);

    DftiFreeDescriptor(&fft_descriptor);
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
