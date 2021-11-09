/**
** @file:  dslash_new.cpp
** @brief: Dslash and Dlash-dagger operaters.
**/

#include <mpi.h>
#include "dslash_new.h"
#include "operator.h"
using namespace std;

#include <immintrin.h>
#include "fast_complex.hpp"

const __m256d half_vec = {0.5, 0.5, 0.5, 0.5};
const __m256d odd_vec = _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);
const __m256d even_vec = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

void U33_P1(fast_complex *src, double *sender, double flag, int b)
{
    // for (int c2 = 0; c2 < 3; c2++) {
    //     tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half;
    //     send_x_b[b * 2 + (0 * 3 + c2) * 2 + 0] = tmp.real();
    //     send_x_b[b * 2 + (0 * 3 + c2) * 2 + 1] = tmp.imag();
    //     tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half;
    //     send_x_b[b * 2 + (1 * 3 + c2) * 2 + 0] = tmp.real();
    //     send_x_b[b * 2 + (1 * 3 + c2) * 2 + 1] = tmp.imag();
    // }

    const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    __m256d vec1 = _mm256_loadu_pd((double *) (src));
    __m256d vec2 = _mm256_loadu_pd((double *) (src + 9));
    vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // __m256d tmp1 = complex_256_mul(vec1, vecA1);
    _mm256_storeu_pd(&sender[b * 2 + (0 * 3 + 0) * 2 + 0], vec1);

    vec1 = _mm256_loadu_pd((double *) (src + 3));
    vec2 = _mm256_loadu_pd((double *) (src + 6));
    vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // __m256d tmp2 = complex_256_mul(vec1, vecA1);
    _mm256_storeu_pd(&sender[b * 2 + (1 * 3 + 0) * 2 + 0], vec1);

    vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
    vec2 = _mm256_load2_m128d((double *) (src + 8), (double *) (src + 11));
    vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // vec1 = complex_256_mul(vec1, vecA2);

    __m128d res1 = _mm256_extractf128_pd(vec1, 0);
    __m128d res2 = _mm256_extractf128_pd(vec1, 1);

    _mm_store_pd(&sender[b * 2 + (0 * 3 + 2) * 2 + 0], res1);
    _mm_store_pd(&sender[b * 2 + (1 * 3 + 2) * 2 + 0], res2);
}

void U33_P2(fast_complex *A, fast_complex *src, double *sender, double flag, int b)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {
    //         tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);

    //         send_x_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
    //         send_x_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();

    //         tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);

    //         send_x_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
    //         send_x_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
    //     }
    // }
    // const __m256d flag_vec = _mm256_xor_pd(_mm256_set1_pd(flag), odd_vec);
    const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    for (int c1 = 0; c1 < 3; c1++) {
        // __m256d vecA1 =
        //     _mm256_xor_pd(_mm256_load2_m128d((double *) &A[c1 + 3], (double *) &A[c1]), even_vec);
        // __m256d vecA2 = _mm256_xor_pd(
        //     _mm256_load2_m128d((double *) &A[c1 + 6], (double *) &A[c1 + 6]), even_vec);

        __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 9));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 6));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d((double *) (src + 8), (double *) (src + 11));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd(&sender[b * 2 + (0 * 3 + c1) * 2 + 0], res1);
        _mm_store_pd(&sender[b * 2 + (1 * 3 + c1) * 2 + 0], res2);
    }
}

void U33_P3(fast_complex *src, double *sender, double flag, int b)
{
    // for (int c2 = 0; c2 < 3; c2++) {
    //     tmp = -(srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half;
    //     send_y_b[b * 2 + (0 * 3 + c2) * 2 + 0] = tmp.real();
    //     send_y_b[b * 2 + (0 * 3 + c2) * 2 + 1] = tmp.imag();
    //     tmp = -(srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half;
    //     send_y_b[b * 2 + (1 * 3 + c2) * 2 + 0] = tmp.real();
    //     send_y_b[b * 2 + (1 * 3 + c2) * 2 + 1] = tmp.imag();
    // }

    const __m256d flag_vec = _mm256_set1_pd(flag);
    // const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    __m256d vec1 = _mm256_loadu_pd((double *) (src));
    __m256d vec2 = _mm256_loadu_pd((double *) (src + 9));
    // vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // __m256d tmp1 = complex_256_mul(vec1, vecA1);
    _mm256_storeu_pd(&sender[b * 2 + (0 * 3 + 0) * 2 + 0], vec1);

    vec1 = _mm256_loadu_pd((double *) (src + 3));
    vec2 = _mm256_loadu_pd((double *) (src + 6));
    // vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // __m256d tmp2 = complex_256_mul(vec1, vecA1);
    _mm256_storeu_pd(&sender[b * 2 + (1 * 3 + 0) * 2 + 0], vec1);

    vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
    vec2 = _mm256_load2_m128d_lrv((double *) (src + 8), (double *) (src + 11));
    // vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // vec1 = complex_256_mul(vec1, vecA2);

    __m128d res1 = _mm256_extractf128_pd(vec1, 0);
    __m128d res2 = _mm256_extractf128_pd(vec1, 1);

    _mm_store_pd(&sender[b * 2 + (0 * 3 + 2) * 2 + 0], res1);
    _mm_store_pd(&sender[b * 2 + (1 * 3 + 2) * 2 + 0], res2);
}

void U33_P4(fast_complex *A, fast_complex *src, double *sender, double flag, int b)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {

    //         tmp = -(srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         send_y_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
    //         send_y_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();
    //         tmp = -(srcO[1 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         send_y_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
    //         send_y_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
    //     }
    // }

    const __m256d flag_vec = _mm256_set1_pd(flag);
    // const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    for (int c1 = 0; c1 < 3; c1++) {
        // __m256d vecA1 =
        //     _mm256_xor_pd(_mm256_load2_m128d((double *) &A[c1 + 3], (double *) &A[c1]), even_vec);
        // __m256d vecA2 = _mm256_xor_pd(
        //     _mm256_load2_m128d((double *) &A[c1 + 6], (double *) &A[c1 + 6]), even_vec);

        __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 9));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 6));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d_lrv((double *) (src + 8), (double *) (src + 11));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd(&sender[b * 2 + (0 * 3 + c1) * 2 + 0], res1);
        _mm_store_pd(&sender[b * 2 + (1 * 3 + c1) * 2 + 0], res2);
    }
}

void U33_P5(fast_complex *src, double *sender, double flag, int b)
{
    //for (int c2 = 0; c2 < 3; c2++) {
    //     tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half;
    //     send_z_b[b * 2 + (0 * 3 + c2) * 2 + 0] += tmp.real();
    //     send_z_b[b * 2 + (0 * 3 + c2) * 2 + 1] += tmp.imag();
    //     tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half;
    //     send_z_b[b * 2 + (1 * 3 + c2) * 2 + 0] += tmp.real();
    //     send_z_b[b * 2 + (1 * 3 + c2) * 2 + 1] += tmp.imag();
    // }

    // const __m256d flag_vec = _mm256_set1_pd(flag);
    const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    __m256d vec1 = _mm256_loadu_pd((double *) (src));
    __m256d vec2 = _mm256_loadu_pd((double *) (src + 6));
    vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // __m256d tmp1 = complex_256_mul(vec1, vecA1);
    _mm256_storeu_pd(&sender[b * 2 + (0 * 3 + 0) * 2 + 0], vec1);

    vec1 = _mm256_loadu_pd((double *) (src + 3));
    vec2 = _mm256_loadu_pd((double *) (src + 9));
    vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // __m256d tmp2 = complex_256_mul(vec1, vecA1);
    _mm256_storeu_pd(&sender[b * 2 + (1 * 3 + 0) * 2 + 0], vec1);

    vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
    vec2 = _mm256_load2_m128d_hrv((double *) (src + 11), (double *) (src + 8));
    vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // vec1 = complex_256_mul(vec1, vecA2);

    __m128d res1 = _mm256_extractf128_pd(vec1, 0);
    __m128d res2 = _mm256_extractf128_pd(vec1, 1);

    _mm_store_pd(&sender[b * 2 + (0 * 3 + 2) * 2 + 0], res1);
    _mm_store_pd(&sender[b * 2 + (1 * 3 + 2) * 2 + 0], res2);
}

void U33_P6(fast_complex *A, fast_complex *src, double *sender, double flag, int b)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {
    //         tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         send_z_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
    //         send_z_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();
    //         tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         send_z_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
    //         send_z_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
    //     }
    // }

    // const __m256d flag_vec = _mm256_set1_pd(flag);
    const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    for (int c1 = 0; c1 < 3; c1++) {
        // __m256d vecA1 =
        //     _mm256_xor_pd(_mm256_load2_m128d((double *) &A[c1 + 3], (double *) &A[c1]), even_vec);
        // __m256d vecA2 = _mm256_xor_pd(
        //     _mm256_load2_m128d((double *) &A[c1 + 6], (double *) &A[c1 + 6]), even_vec);

        __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 6));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 9));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d_hrv((double *) (src + 11), (double *) (src + 8));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd(&sender[b * 2 + (0 * 3 + c1) * 2 + 0], res1);
        _mm_store_pd(&sender[b * 2 + (1 * 3 + c1) * 2 + 0], res2);
    }
}

void U33_P7(fast_complex *src, double *sender, double flag, int b)
{
    // for (int c2 = 0; c2 < 3; c2++) {
    //     tmp = -(srcO[0 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half;
    //     send_t_b[b * 2 + (0 * 3 + c2) * 2 + 0] += tmp.real();
    //     send_t_b[b * 2 + (0 * 3 + c2) * 2 + 1] += tmp.imag();
    //     tmp = -(srcO[1 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half;
    //     send_t_b[b * 2 + (1 * 3 + c2) * 2 + 0] += tmp.real();
    //     send_t_b[b * 2 + (1 * 3 + c2) * 2 + 1] += tmp.imag();
    // }

    const __m256d flag_vec = _mm256_set1_pd(flag);
    // const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    __m256d vec1 = _mm256_loadu_pd((double *) (src));
    __m256d vec2 = _mm256_loadu_pd((double *) (src + 6));
    // vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // __m256d tmp1 = complex_256_mul(vec1, vecA1);
    _mm256_storeu_pd(&sender[b * 2 + (0 * 3 + 0) * 2 + 0], vec1);

    vec1 = _mm256_loadu_pd((double *) (src + 3));
    vec2 = _mm256_loadu_pd((double *) (src + 9));
    // vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // __m256d tmp2 = complex_256_mul(vec1, vecA1);
    _mm256_storeu_pd(&sender[b * 2 + (1 * 3 + 0) * 2 + 0], vec1);

    vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
    vec2 = _mm256_load2_m128d((double *) (src + 11), (double *) (src + 8));
    // vec2 = _mm256_permute_pd(vec2, 0b0101);
    vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
    vec1 = _mm256_mul_pd(vec1, half_vec);
    // vec1 = complex_256_mul(vec1, vecA2);

    __m128d res1 = _mm256_extractf128_pd(vec1, 0);
    __m128d res2 = _mm256_extractf128_pd(vec1, 1);

    _mm_store_pd(&sender[b * 2 + (0 * 3 + 2) * 2 + 0], res1);
    _mm_store_pd(&sender[b * 2 + (1 * 3 + 2) * 2 + 0], res2);
}

void U33_P8(fast_complex *A, fast_complex *src, double *sender, double flag, int b)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {
    //         tmp = -(srcO[0 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         send_t_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
    //         send_t_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();
    //         tmp = -(srcO[1 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         send_t_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
    //         send_t_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
    //     }
    // }

    const __m256d flag_vec = _mm256_set1_pd(flag);
    // const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    for (int c1 = 0; c1 < 3; c1++) {
        // __m256d vecA1 =
        //     _mm256_xor_pd(_mm256_load2_m128d((double *) &A[c1 + 3], (double *) &A[c1]), even_vec);
        // __m256d vecA2 = _mm256_xor_pd(
        //     _mm256_load2_m128d((double *) &A[c1 + 6], (double *) &A[c1 + 6]), even_vec);

        __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 6));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 9));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d((double *) (src + 11), (double *) (src + 8));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd(&sender[b * 2 + (0 * 3 + c1) * 2 + 0], res1);
        _mm_store_pd(&sender[b * 2 + (1 * 3 + c1) * 2 + 0], res2);
    }
}

void U33_P9(fast_complex *A, fast_complex *src, fast_complex *dest, double flag)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {
    //         {
    //             tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half * AE[c1 * 3 + c2];
    //             destE[0 * 3 + c1] += tmp;
    //             destE[3 * 3 + c1] += flag * (I * tmp);
    //             tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half * AE[c1 * 3 + c2];
    //             destE[1 * 3 + c1] += tmp;
    //             destE[2 * 3 + c1] += flag * (I * tmp);
    //         }
    //     }
    // }

    const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    for (int c1 = 0; c1 < 3; c1++) {

        __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]);
        __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 9));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 6));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d((double *) (src + 8), (double *) (src + 11));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[0 * 3 + c1], res1);
        _mm_store_pd((double *) &dest[1 * 3 + c1], res2);

        vec1 = _mm256_set_m128d(res2, res1);
        vec1 = _mm256_permute_pd(vec1, 0b0101);
        vec1 = _mm256_mul_pd(flag_vec, vec1);

        res1 = _mm256_extractf128_pd(vec1, 0);
        res2 = _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[3 * 3 + c1], res1);
        _mm_store_pd((double *) &dest[2 * 3 + c1], res2);
    }
}

void U33_P10(fast_complex *A, fast_complex *src, fast_complex *dest, double flag)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {
    //         tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);

    //         destE[0 * 3 + c1] += tmp;
    //         destE[3 * 3 + c1] += flag * (-I * tmp);

    //         tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);

    //         destE[1 * 3 + c1] += tmp;
    //         destE[2 * 3 + c1] += flag * (-I * tmp);
    //     }
    // }

    const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    for (int c1 = 0; c1 < 3; c1++) {

        // __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]) * even_vec;
        // __m256d vecA2 =
        //     _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]) * even_vec;

        __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 9));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 6));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d((double *) (src + 8), (double *) (src + 11));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[0 * 3 + c1],
                     res1 + _mm_load_pd((double *) &dest[0 * 3 + c1]));
        _mm_store_pd((double *) &dest[1 * 3 + c1],
                     res2 + _mm_load_pd((double *) &dest[1 * 3 + c1]));

        vec1 = _mm256_set_m128d(res2, res1);
        vec1 = _mm256_permute_pd(vec1, 0b0101);
        vec1 = _mm256_mul_pd(-flag_vec, vec1);
        vec2 = _mm256_load2_m128d((double *) &dest[2 * 3 + c1], (double *) &dest[3 * 3 + c1]);
        vec1 = _mm256_add_pd(vec1, vec2);

        res1 = _mm256_extractf128_pd(vec1, 0);
        res2 = _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[3 * 3 + c1], res1);
        _mm_store_pd((double *) &dest[2 * 3 + c1], res2);
    }
}

void U33_P11(fast_complex *A, fast_complex *src, fast_complex *dest, double flag)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {
    //         tmp = -(srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
    //               AE[c1 * 3 + c2];
    //         destE[0 * 3 + c1] += tmp;
    //         destE[3 * 3 + c1] += flag * (tmp);
    //         tmp = -(srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half *
    //               AE[c1 * 3 + c2];
    //         destE[1 * 3 + c1] += tmp;
    //         destE[2 * 3 + c1] -= flag * (tmp);
    //     }
    // }

    const __m256d flag_vec = _mm256_set1_pd(flag);

    for (int c1 = 0; c1 < 3; c1++) {

        // __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]) * even_vec;
        // __m256d vecA2 =
        //     _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]) * even_vec;

        // __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        // __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        // __m256d vecA1 = _mm256_load2_m128d((double *) &A[c1], (double *) &A[c1 + 3]);
        // __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 + 6], (double *) &A[c1 + 6]);

        __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]);
        __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 9));
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 6));
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d_hrv((double *) (src + 8), (double *) (src + 11));
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[0 * 3 + c1],
                     res1 + _mm_load_pd((double *) &dest[0 * 3 + c1]));
        _mm_store_pd((double *) &dest[1 * 3 + c1],
                     res2 + _mm_load_pd((double *) &dest[1 * 3 + c1]));

        vec1 = _mm256_set_m128d(-res2, res1);
        vec1 = _mm256_mul_pd(flag_vec, vec1);
        vec2 = _mm256_load2_m128d((double *) &dest[2 * 3 + c1], (double *) &dest[3 * 3 + c1]);
        vec1 = _mm256_add_pd(vec1, vec2);

        res1 = _mm256_extractf128_pd(vec1, 0);
        res2 = _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[3 * 3 + c1], res1);
        _mm_store_pd((double *) &dest[2 * 3 + c1], res2);
    }
}

void U33_P12(fast_complex *A, fast_complex *src, fast_complex *dest, double flag)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {
    //         tmp = -(srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         destE[0 * 3 + c1] += tmp;
    //         destE[3 * 3 + c1] -= flag * (tmp);
    //         tmp = -(srcO[1 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         destE[1 * 3 + c1] += tmp;
    //         destE[2 * 3 + c1] += flag * (tmp);
    //     }
    // }

    const __m256d flag_vec = _mm256_set1_pd(flag);

    for (int c1 = 0; c1 < 3; c1++) {

        // __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]) * even_vec;
        // __m256d vecA2 =
        //     _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]) * even_vec;

        __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        // __m256d vecA1 = _mm256_load2_m128d((double *) &A[c1], (double *) &A[c1 + 3]);
        // __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 + 6], (double *) &A[c1 + 6]);

        // __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]);
        // __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 9));
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 6));
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d_lrv((double *) (src + 8), (double *) (src + 11));
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[0 * 3 + c1],
                     res1 + _mm_load_pd((double *) &dest[0 * 3 + c1]));
        _mm_store_pd((double *) &dest[1 * 3 + c1],
                     res2 + _mm_load_pd((double *) &dest[1 * 3 + c1]));

        vec1 = _mm256_set_m128d(res2, -res1);
        vec1 = _mm256_mul_pd(flag_vec, vec1);
        vec2 = _mm256_load2_m128d((double *) &dest[2 * 3 + c1], (double *) &dest[3 * 3 + c1]);
        vec1 = _mm256_add_pd(vec1, vec2);

        res1 = _mm256_extractf128_pd(vec1, 0);
        res2 = _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[3 * 3 + c1], res1);
        _mm_store_pd((double *) &dest[2 * 3 + c1], res2);
    }
}
void U33_P13(fast_complex *A, fast_complex *src, fast_complex *dest, double flag)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {

    //         tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half *
    //               AE[c1 * 3 + c2];
    //         destE[0 * 3 + c1] += tmp;
    //         destE[2 * 3 + c1] += flag * (I * tmp);
    //         tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half *
    //               AE[c1 * 3 + c2];
    //         destE[1 * 3 + c1] += tmp;
    //         destE[3 * 3 + c1] += flag * (-I * tmp);
    //     }
    // }

    // const __m256d flag_vec = _mm256_set1_pd(flag);
    const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    for (int c1 = 0; c1 < 3; c1++) {

        // __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]) * even_vec;
        // __m256d vecA2 =
        //     _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]) * even_vec;

        // __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        // __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        // __m256d vecA1 = _mm256_load2_m128d((double *) &A[c1], (double *) &A[c1 + 3]);
        // __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 + 6], (double *) &A[c1 + 6]);

        __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]);
        __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 6));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 9));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d_lrv((double *) (src + 11), (double *) (src + 8));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[0 * 3 + c1],
                     res1 + _mm_load_pd((double *) &dest[0 * 3 + c1]));
        _mm_store_pd((double *) &dest[1 * 3 + c1],
                     res2 + _mm_load_pd((double *) &dest[1 * 3 + c1]));

        vec1 = _mm256_set_m128d(-res2, res1);
        vec1 = _mm256_permute_pd(vec1, 0b0101);
        vec1 = _mm256_mul_pd(flag_vec, vec1);
        vec2 = _mm256_load2_m128d((double *) &dest[3 * 3 + c1], (double *) &dest[2 * 3 + c1]);
        vec1 = _mm256_add_pd(vec1, vec2);

        res1 = _mm256_extractf128_pd(vec1, 0);
        res2 = _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[2 * 3 + c1], res1);
        _mm_store_pd((double *) &dest[3 * 3 + c1], res2);
    }
}

void U33_P14(fast_complex *A, fast_complex *src, fast_complex *dest, double flag)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {
    //         tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         destE[0 * 3 + c1] += tmp;
    //         destE[2 * 3 + c1] += flag * (-I * tmp);
    //         tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         destE[1 * 3 + c1] += tmp;
    //         destE[3 * 3 + c1] += flag * (I * tmp);
    //     }
    // }

    // const __m256d flag_vec = _mm256_set1_pd(flag);
    const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    for (int c1 = 0; c1 < 3; c1++) {

        // __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]) * even_vec;
        // __m256d vecA2 =
        //     _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]) * even_vec;

        __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        // __m256d vecA1 = _mm256_load2_m128d((double *) &A[c1], (double *) &A[c1 + 3]);
        // __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 + 6], (double *) &A[c1 + 6]);

        // __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]);
        // __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 6));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 9));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d_hrv((double *) (src + 11), (double *) (src + 8));
        vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[0 * 3 + c1],
                     res1 + _mm_load_pd((double *) &dest[0 * 3 + c1]));
        _mm_store_pd((double *) &dest[1 * 3 + c1],
                     res2 + _mm_load_pd((double *) &dest[1 * 3 + c1]));

        vec1 = _mm256_set_m128d(res2, -res1);
        vec1 = _mm256_permute_pd(vec1, 0b0101);
        vec1 = _mm256_mul_pd(flag_vec, vec1);
        vec2 = _mm256_load2_m128d((double *) &dest[3 * 3 + c1], (double *) &dest[2 * 3 + c1]);
        vec1 = _mm256_add_pd(vec1, vec2);

        res1 = _mm256_extractf128_pd(vec1, 0);
        res2 = _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[2 * 3 + c1], res1);
        _mm_store_pd((double *) &dest[3 * 3 + c1], res2);
    }
}
void U33_P15(fast_complex *A, fast_complex *src, fast_complex *dest, double flag)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {
    //         tmp = -(srcO[0 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half *
    //               AE[c1 * 3 + c2];
    //         destE[0 * 3 + c1] += tmp;
    //         destE[2 * 3 + c1] -= flag * (tmp);
    //         tmp = -(srcO[1 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half *
    //               AE[c1 * 3 + c2];
    //         destE[1 * 3 + c1] += tmp;
    //         destE[3 * 3 + c1] -= flag * (tmp);
    //     }
    // }

    const __m256d flag_vec = _mm256_set1_pd(flag);
    // const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    for (int c1 = 0; c1 < 3; c1++) {

        // __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]) * even_vec;
        // __m256d vecA2 =
        //     _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]) * even_vec;

        // __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        // __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        // __m256d vecA1 = _mm256_load2_m128d((double *) &A[c1], (double *) &A[c1 + 3]);
        // __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 + 6], (double *) &A[c1 + 6]);

        __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]);
        __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 6));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 9));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d((double *) (src + 11), (double *) (src + 8));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[0 * 3 + c1],
                     res1 + _mm_load_pd((double *) &dest[0 * 3 + c1]));
        _mm_store_pd((double *) &dest[1 * 3 + c1],
                     res2 + _mm_load_pd((double *) &dest[1 * 3 + c1]));

        vec1 = _mm256_set_m128d(res2, res1);
        // vec1 = _mm256_permute_pd(vec1, 0b0101);
        vec1 = _mm256_mul_pd(flag_vec, vec1);
        vec2 = _mm256_load2_m128d((double *) &dest[3 * 3 + c1], (double *) &dest[2 * 3 + c1]);
        vec1 = _mm256_sub_pd(vec2, vec1);

        res1 = _mm256_extractf128_pd(vec1, 0);
        res2 = _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[2 * 3 + c1], res1);
        _mm_store_pd((double *) &dest[3 * 3 + c1], res2);
    }
}
void U33_P16(fast_complex *A, fast_complex *src, fast_complex *dest, double flag)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {
    //         tmp = -(srcO[0 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         destE[0 * 3 + c1] += tmp;
    //         destE[2 * 3 + c1] += flag * (tmp);
    //         tmp = -(srcO[1 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
    //               conj(AO[c2 * 3 + c1]);
    //         destE[1 * 3 + c1] += tmp;
    //         destE[3 * 3 + c1] += flag * (tmp);
    //     }
    // }

    const __m256d flag_vec = _mm256_set1_pd(flag);
    // const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    for (int c1 = 0; c1 < 3; c1++) {

        // __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]) * even_vec;
        // __m256d vecA2 =
        //     _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]) * even_vec;

        __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        // __m256d vecA1 = _mm256_load2_m128d((double *) &A[c1], (double *) &A[c1 + 3]);
        // __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 + 6], (double *) &A[c1 + 6]);

        // __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]);
        // __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2 = _mm256_loadu_pd((double *) (src + 6));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        vec2 = _mm256_loadu_pd((double *) (src + 9));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        vec2 = _mm256_load2_m128d((double *) (src + 11), (double *) (src + 8));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[0 * 3 + c1],
                     res1 + _mm_load_pd((double *) &dest[0 * 3 + c1]));
        _mm_store_pd((double *) &dest[1 * 3 + c1],
                     res2 + _mm_load_pd((double *) &dest[1 * 3 + c1]));

        vec1 = _mm256_set_m128d(res2, res1);
        // vec1 = _mm256_permute_pd(vec1, 0b0101);
        vec1 = _mm256_mul_pd(flag_vec, vec1);
        vec2 = _mm256_load2_m128d((double *) &dest[3 * 3 + c1], (double *) &dest[2 * 3 + c1]);
        vec1 = _mm256_add_pd(vec2, vec1);

        res1 = _mm256_extractf128_pd(vec1, 0);
        res2 = _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[2 * 3 + c1], res1);
        _mm_store_pd((double *) &dest[3 * 3 + c1], res2);
    }
}

void U33_P17(fast_complex *A, fast_complex *src, fast_complex *dest, double flag)
{
    // for (int c1 = 0; c1 < 3; c1++) {
    //     for (int c2 = 0; c2 < 3; c2++) {
    //         tmp = srcO[0 * 3 + c2] * AE[c1 * 3 + c2];
    //         destE[0 * 3 + c1] += tmp;
    //         destE[3 * 3 + c1] += flag * (I * tmp);
    //         tmp = srcO[1 * 3 + c2] * AE[c1 * 3 + c2];
    //         destE[1 * 3 + c1] += tmp;
    //         destE[2 * 3 + c1] += flag * (I * tmp);
    //     }
    // }

    // const __m256d flag_vec = _mm256_set1_pd(flag);
    const __m256d flag_vec = _mm256_setr_pd(-flag, flag, -flag, flag);

    for (int c1 = 0; c1 < 3; c1++) {

        // __m256d vecA1 = _mm256_setr_pd(A[c1].r, -A[c1].i, A[c1 + 3].r, -A[c1 + 3].i);
        // __m256d vecA2 = _mm256_setr_pd(A[c1 + 6].r, -A[c1 + 6].i, A[c1 + 6].r, -A[c1 + 6].i);

        // __m256d vecA1 = _mm256_load2_m128d((double *) &A[c1], (double *) &A[c1 + 3]);
        // __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 + 6], (double *) &A[c1 + 6]);

        __m256d vecA1 = _mm256_loadu_pd((double *) &A[c1 * 3]);
        __m256d vecA2 = _mm256_load2_m128d((double *) &A[c1 * 3 + 2], (double *) &A[c1 * 3 + 2]);

        __m256d vec1 = _mm256_loadu_pd((double *) (src));
        __m256d vec2;
        // __m256d vec2 = _mm256_loadu_pd((double *) (src + 6));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        // vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        // vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp1 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_loadu_pd((double *) (src + 3));
        // vec2 = _mm256_loadu_pd((double *) (src + 9));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        // vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        // vec1 = _mm256_mul_pd(vec1, half_vec);
        __m256d tmp2 = complex_256_mul(vec1, vecA1);

        vec1 = _mm256_load2_m128d((double *) (src + 5), (double *) (src + 2));
        // vec2 = _mm256_load2_m128d((double *) (src + 11), (double *) (src + 8));
        // vec2 = _mm256_permute_pd(vec2, 0b0101);
        // vec1 = _mm256_fnmsub_pd(flag_vec, vec2, vec1);
        // vec1 = _mm256_mul_pd(vec1, half_vec);
        vec1 = complex_256_mul(vec1, vecA2);

        __m128d res1 = _mm256_extractf128_pd(tmp1, 0) + _mm256_extractf128_pd(tmp1, 1) +
                       _mm256_extractf128_pd(vec1, 0);
        __m128d res2 = _mm256_extractf128_pd(tmp2, 0) + _mm256_extractf128_pd(tmp2, 1) +
                       _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[0 * 3 + c1],
                     res1 + _mm_load_pd((double *) &dest[0 * 3 + c1]));
        _mm_store_pd((double *) &dest[1 * 3 + c1],
                     res2 + _mm_load_pd((double *) &dest[1 * 3 + c1]));

        vec1 = _mm256_set_m128d(res2, res1);
        vec1 = _mm256_permute_pd(vec1, 0b0101);
        vec1 = _mm256_mul_pd(flag_vec, vec1);
        vec2 = _mm256_load2_m128d((double *) &dest[2 * 3 + c1], (double *) &dest[3 * 3 + c1]);
        vec1 = _mm256_add_pd(vec2, vec1);

        res1 = _mm256_extractf128_pd(vec1, 0);
        res2 = _mm256_extractf128_pd(vec1, 1);

        _mm_store_pd((double *) &dest[3 * 3 + c1], res1);
        _mm_store_pd((double *) &dest[2 * 3 + c1], res2);
    }
}

void DslashEEOONew(lattice_fermion &src, lattice_fermion &dest, const double mass)
{

    // dest.clean();
    const double a = 4.0;

    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    // int subgrid_vol_cb = (subgrid_vol) >> 1;

    __m256d vec_factor = _mm256_set1_pd(a + mass);

    // for (int i = 0; i < subgrid_vol_cb * 3 * 4; i += 12) {
    for (int i = 0; i < subgrid_vol * 3 * 4; i += 12) {
        __m256d vec1 = _mm256_load_pd((double *) &src.A[i]);
        __m256d vec2 = _mm256_load_pd((double *) &src.A[i + 2]);
        __m256d vec3 = _mm256_load_pd((double *) &src.A[i + 4]);
        __m256d vec4 = _mm256_load_pd((double *) &src.A[i + 6]);
        __m256d vec5 = _mm256_load_pd((double *) &src.A[i + 8]);
        __m256d vec6 = _mm256_load_pd((double *) &src.A[i + 10]);
        vec1 = _mm256_mul_pd(vec_factor, vec1);
        vec2 = _mm256_mul_pd(vec_factor, vec2);
        vec3 = _mm256_mul_pd(vec_factor, vec3);
        vec4 = _mm256_mul_pd(vec_factor, vec4);
        vec5 = _mm256_mul_pd(vec_factor, vec5);
        vec6 = _mm256_mul_pd(vec_factor, vec6);
        _mm256_stream_pd((double *) &dest.A[i], vec1);
        _mm256_stream_pd((double *) &dest.A[i + 2], vec2);
        _mm256_stream_pd((double *) &dest.A[i + 4], vec3);
        _mm256_stream_pd((double *) &dest.A[i + 6], vec4);
        _mm256_stream_pd((double *) &dest.A[i + 8], vec5);
        _mm256_stream_pd((double *) &dest.A[i + 10], vec6);
        // dest.A[i] = (a + mass) * src.A[i];
    }

    // for (int i = subgrid_vol_cb * 3 * 4; i < subgrid_vol * 3 * 4; i += 12) {
    //     dest.A[i] = (a + mass) * src.A[i];
    // }
}

// cb = 0  EO  ;  cb = 1 OE
void DslashoffdNew(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag,
                   int cb)
{
    // sublattice
    int N_sub[4] = {src.site_vec[0] / src.subgs[0], src.site_vec[1] / src.subgs[1],
                    src.site_vec[2] / src.subgs[2], src.site_vec[3] / src.subgs[3]};

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int site_x_f[4] = {(rank % N_sub[0] + 1) % N_sub[0], (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_x_b[4] = {(rank % N_sub[0] - 1 + N_sub[0]) % N_sub[0], (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    const int nodenum_x_b = get_nodenum(site_x_b, N_sub, 4);
    const int nodenum_x_f = get_nodenum(site_x_f, N_sub, 4);

    int site_y_f[4] = {(rank % N_sub[0]), ((rank / N_sub[0]) % N_sub[1] + 1) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_y_b[4] = {(rank % N_sub[0]), ((rank / N_sub[0]) % N_sub[1] - 1 + N_sub[1]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    const int nodenum_y_b = get_nodenum(site_y_b, N_sub, 4);
    const int nodenum_y_f = get_nodenum(site_y_f, N_sub, 4);

    int site_z_f[4] = {(rank % N_sub[0]), (rank / N_sub[0]) % N_sub[1],
                       ((rank / (N_sub[1] * N_sub[0])) % N_sub[2] + 1) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_z_b[4] = {(rank % N_sub[0]), (rank / N_sub[0]) % N_sub[1],
                       ((rank / (N_sub[1] * N_sub[0])) % N_sub[2] - 1 + N_sub[2]) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    const int nodenum_z_b = get_nodenum(site_z_b, N_sub, 4);
    const int nodenum_z_f = get_nodenum(site_z_f, N_sub, 4);

    int site_t_f[4] = {(rank % N_sub[0]), (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       (rank / (N_sub[2] * N_sub[1] * N_sub[0]) + 1) % N_sub[3]};

    int site_t_b[4] = {(rank % N_sub[0]), (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       (rank / (N_sub[2] * N_sub[1] * N_sub[0]) - 1 + N_sub[3]) % N_sub[3]};

    const int nodenum_t_b = get_nodenum(site_t_b, N_sub, 4);
    const int nodenum_t_f = get_nodenum(site_t_f, N_sub, 4);

    dest.clean();
    double flag = (dag == true) ? -1 : 1;

    int subgrid[4] = {src.subgs[0], src.subgs[1], src.subgs[2], src.subgs[3]};
    int subgrid_vol = (subgrid[0] * subgrid[1] * subgrid[2] * subgrid[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;
    subgrid[0] >>= 1;
    const double half = 0.5;
    const complex<double> I(0, 1);
    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * subgrid[1] +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * subgrid[2] +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * subgrid[3];

    MPI_Request reqs[8 * size];
    MPI_Request reqr[8 * size];
    MPI_Status stas[8 * size];
    MPI_Status star[8 * size];

    int len_x_f = (subgrid[1] * subgrid[2] * subgrid[3] + cb) >> 1;

    double *resv_x_f = new double[len_x_f * 6 * 2];
    double *send_x_b = new double[len_x_f * 6 * 2];
    if (N_sub[0] != 1) {
        // for (int i = 0; i < len_x_f * 6 * 2; i++) {
        //     send_x_b[i] = 0;
        // }

        int cont = 0;

        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {

                    if ((y + z + t + x_p) % 2 == cb) {
                        continue;
                    }
                    int x = 0;
                    complex<double> tmp;
                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;
                    int b = cont * 6;
                    cont += 1;

                    U33_P1((fast_complex *) srcO, send_x_b, flag, b);

                    // for (int c2 = 0; c2 < 3; c2++) {
                    //     tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half;
                    //     send_x_b[b * 2 + (0 * 3 + c2) * 2 + 0] = tmp.real();
                    //     send_x_b[b * 2 + (0 * 3 + c2) * 2 + 1] = tmp.imag();
                    //     tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half;
                    //     send_x_b[b * 2 + (1 * 3 + c2) * 2 + 0] = tmp.real();
                    //     send_x_b[b * 2 + (1 * 3 + c2) * 2 + 1] = tmp.imag();
                    // }
                }
            }
        }

        MPI_Isend(send_x_b, len_x_f * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * rank, MPI_COMM_WORLD,
                  &reqs[8 * rank]);
        MPI_Irecv(resv_x_f, len_x_f * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * nodenum_x_f,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_x_f]);
    }

    int len_x_b = (subgrid[1] * subgrid[2] * subgrid[3] + 1 - cb) >> 1;

    double *resv_x_b = new double[len_x_b * 6 * 2];
    double *send_x_f = new double[len_x_b * 6 * 2];

    if (N_sub[0] != 1) {
        // for (int i = 0; i < len_x_b * 6 * 2; i++) {
        //     send_x_f[i] = 0;
        // }

        int cont = 0;

        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    if (((y + z + t + x_p) % 2) != cb) {
                        continue;
                    }

                    int x = subgrid[0] - 1;

                    complex<double> *AO = U.A[0] + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                    subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                    x + (1 - cb) * subgrid_vol_cb) *
                                                       9;

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    complex<double> tmp;

                    int b = cont * 6;
                    cont += 1;

                    U33_P2((fast_complex *) AO, (fast_complex *) srcO, send_x_f, flag, b);
                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {
                    //         tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);

                    //         send_x_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
                    //         send_x_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();

                    //         tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);

                    //         send_x_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
                    //         send_x_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
                    //     }
                    // }
                }
            }
        }

        MPI_Isend(send_x_f, len_x_b * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * rank + 1, MPI_COMM_WORLD,
                  &reqs[8 * rank + 1]);
        MPI_Irecv(resv_x_b, len_x_b * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * nodenum_x_b + 1,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_x_b + 1]);
    }

    int len_y_f = subgrid[0] * subgrid[2] * subgrid[3];

    double *resv_y_f = new double[len_y_f * 6 * 2];
    double *send_y_b = new double[len_y_f * 6 * 2];
    if (N_sub[1] != 1) {
        // for (int i = 0; i < len_y_f * 6 * 2; i++) {
        //     send_y_b[i] = 0;
        // }

        int cont = 0;

        for (int x = 0; x < subgrid[0]; x++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    int y = 0;
                    complex<double> tmp;
                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    int b = cont * 6;
                    cont += 1;

                    U33_P3((fast_complex *) srcO, send_y_b, flag, b);

                    // for (int c2 = 0; c2 < 3; c2++) {
                    //     tmp = -(srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half;
                    //     send_y_b[b * 2 + (0 * 3 + c2) * 2 + 0] = tmp.real();
                    //     send_y_b[b * 2 + (0 * 3 + c2) * 2 + 1] = tmp.imag();
                    //     tmp = -(srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half;
                    //     send_y_b[b * 2 + (1 * 3 + c2) * 2 + 0] = tmp.real();
                    //     send_y_b[b * 2 + (1 * 3 + c2) * 2 + 1] = tmp.imag();
                    // }
                }
            }
        }

        MPI_Isend(send_y_b, len_y_f * 6 * 2, MPI_DOUBLE, nodenum_y_b, 8 * rank + 2, MPI_COMM_WORLD,
                  &reqs[8 * rank + 2]);
        MPI_Irecv(resv_y_f, len_y_f * 6 * 2, MPI_DOUBLE, nodenum_y_f, 8 * nodenum_y_f + 2,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_y_f + 2]);
    }

    int len_y_b = subgrid[0] * subgrid[2] * subgrid[3];

    double *resv_y_b = new double[len_y_b * 6 * 2];
    double *send_y_f = new double[len_y_b * 6 * 2];

    if (N_sub[1] != 1) {

        // for (int i = 0; i < len_y_b * 6 * 2; i++) {
        //     send_y_f[i] = 0;
        // }

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    complex<double> tmp;

                    int y = subgrid[1] - 1;

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    complex<double> *AO = U.A[1] + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                    subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                    x + (1 - cb) * subgrid_vol_cb) *
                                                       9;

                    int b = cont * 6;
                    cont += 1;

                    U33_P4((fast_complex *) AO, (fast_complex *) srcO, send_y_f, flag, b);

                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {

                    //         tmp = -(srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         send_y_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
                    //         send_y_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();
                    //         tmp = -(srcO[1 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         send_y_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
                    //         send_y_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
                    //     }
                    // }
                }
            }
        }

        MPI_Isend(send_y_f, len_y_b * 6 * 2, MPI_DOUBLE, nodenum_y_f, 8 * rank + 3, MPI_COMM_WORLD,
                  &reqs[8 * rank + 3]);
        MPI_Irecv(resv_y_b, len_y_b * 6 * 2, MPI_DOUBLE, nodenum_y_b, 8 * nodenum_y_b + 3,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_y_b + 3]);
    }

    int len_z_f = subgrid[0] * subgrid[1] * subgrid[3];

    double *resv_z_f = new double[len_z_f * 6 * 2];
    double *send_z_b = new double[len_z_f * 6 * 2];
    if (N_sub[2] != 1) {
        // for (int i = 0; i < len_z_f * 6 * 2; i++) {
        //     send_z_b[i] = 0;
        // }

        int cont = 0;

        for (int x = 0; x < subgrid[0]; x++) {
            for (int y = 0; y < subgrid[1]; y++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    int z = 0;

                    complex<double> tmp;
                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    int b = cont * 6;
                    cont += 1;

                    U33_P5((fast_complex *) srcO, send_z_b, flag, b);

                    // for (int c2 = 0; c2 < 3; c2++) {
                    //     tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half;
                    //     send_z_b[b * 2 + (0 * 3 + c2) * 2 + 0] += tmp.real();
                    //     send_z_b[b * 2 + (0 * 3 + c2) * 2 + 1] += tmp.imag();
                    //     tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half;
                    //     send_z_b[b * 2 + (1 * 3 + c2) * 2 + 0] += tmp.real();
                    //     send_z_b[b * 2 + (1 * 3 + c2) * 2 + 1] += tmp.imag();
                    // }
                }
            }
        }

        MPI_Isend(send_z_b, len_z_f * 6 * 2, MPI_DOUBLE, nodenum_z_b, 8 * rank + 4, MPI_COMM_WORLD,
                  &reqs[8 * rank + 4]);
        MPI_Irecv(resv_z_f, len_z_f * 6 * 2, MPI_DOUBLE, nodenum_z_f, 8 * nodenum_z_f + 4,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_z_f + 4]);
    }

    int len_z_b = subgrid[0] * subgrid[1] * subgrid[3];

    double *resv_z_b = new double[len_z_b * 6 * 2];
    double *send_z_f = new double[len_z_b * 6 * 2];
    if (N_sub[2] != 1) {

        // for (int i = 0; i < len_z_b * 6 * 2; i++) {
        //     send_z_f[i] = 0;
        // }

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++) {
            for (int y = 0; y < subgrid[1]; y++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    complex<double> tmp;

                    int z = subgrid[2] - 1;

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    complex<double> *AO = U.A[2] + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                    subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                    x + (1 - cb) * subgrid_vol_cb) *
                                                       9;

                    int b = cont * 6;
                    cont += 1;

                    U33_P6((fast_complex *) AO, (fast_complex *) srcO, send_z_f, flag, b);

                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {

                    //         tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         send_z_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
                    //         send_z_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();
                    //         tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         send_z_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
                    //         send_z_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
                    //     }
                    // }
                }
            }
        }

        MPI_Isend(send_z_f, len_z_b * 6 * 2, MPI_DOUBLE, nodenum_z_f, 8 * rank + 5, MPI_COMM_WORLD,
                  &reqs[8 * rank + 5]);
        MPI_Irecv(resv_z_b, len_z_b * 6 * 2, MPI_DOUBLE, nodenum_z_b, 8 * nodenum_z_b + 5,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_z_b + 5]);
    }

    int len_t_f = subgrid[0] * subgrid[1] * subgrid[2];

    double *resv_t_f = new double[len_t_f * 6 * 2];
    double *send_t_b = new double[len_t_f * 6 * 2];
    if (N_sub[3] != 1) {
        // for (int i = 0; i < len_t_f * 6 * 2; i++) {
        //     send_t_b[i] = 0;
        // }

        int cont = 0;

        for (int x = 0; x < subgrid[0]; x++) {
            for (int y = 0; y < subgrid[1]; y++) {
                for (int z = 0; z < subgrid[2]; z++) {
                    int t = 0;

                    complex<double> tmp;
                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    int b = cont * 6;
                    cont += 1;

                    U33_P7((fast_complex *) srcO, send_t_b, flag, b);

                    // for (int c2 = 0; c2 < 3; c2++) {
                    //     tmp = -(srcO[0 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half;
                    //     send_t_b[b * 2 + (0 * 3 + c2) * 2 + 0] += tmp.real();
                    //     send_t_b[b * 2 + (0 * 3 + c2) * 2 + 1] += tmp.imag();
                    //     tmp = -(srcO[1 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half;
                    //     send_t_b[b * 2 + (1 * 3 + c2) * 2 + 0] += tmp.real();
                    //     send_t_b[b * 2 + (1 * 3 + c2) * 2 + 1] += tmp.imag();
                    // }
                }
            }
        }

        MPI_Isend(send_t_b, len_t_f * 6 * 2, MPI_DOUBLE, nodenum_t_b, 8 * rank + 6, MPI_COMM_WORLD,
                  &reqs[8 * rank + 6]);
        MPI_Irecv(resv_t_f, len_t_f * 6 * 2, MPI_DOUBLE, nodenum_t_f, 8 * nodenum_t_f + 6,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_t_f + 6]);
    }

    int len_t_b = subgrid[0] * subgrid[1] * subgrid[2];

    double *resv_t_b = new double[len_t_b * 6 * 2];
    double *send_t_f = new double[len_t_b * 6 * 2];
    if (N_sub[3] != 1) {

        // for (int i = 0; i < len_t_b * 6 * 2; i++) {
        //     send_t_f[i] = 0;
        // }

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++) {
            for (int y = 0; y < subgrid[1]; y++) {
                for (int z = 0; z < subgrid[2]; z++) {
                    complex<double> tmp;

                    int t = subgrid[3] - 1;

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    complex<double> *AO = U.A[3] + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                    subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                    x + (1 - cb) * subgrid_vol_cb) *
                                                       9;

                    int b = cont * 6;
                    cont += 1;

                    U33_P8((fast_complex *) AO, (fast_complex *) srcO, send_t_f, flag, b);

                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {

                    //         tmp = -(srcO[0 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         send_t_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
                    //         send_t_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();
                    //         tmp = -(srcO[1 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         send_t_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
                    //         send_t_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
                    //     }
                    // }
                }
            }
        }

        MPI_Isend(send_t_f, len_t_b * 6 * 2, MPI_DOUBLE, nodenum_t_f, 8 * rank + 7, MPI_COMM_WORLD,
                  &reqs[8 * rank + 7]);
        MPI_Irecv(resv_t_b, len_t_b * 6 * 2, MPI_DOUBLE, nodenum_t_b, 8 * nodenum_t_b + 7,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_t_b + 7]);
    }

    //////////////////////////////////////////////////////// no comunication
    /////////////////////////////////////////////////////////

    for (int y = 0; y < subgrid[1]; y++) {
        for (int z = 0; z < subgrid[2]; z++) {
            for (int t = 0; t < subgrid[3]; t++) {
                int x_u =
                    ((y + z + t + x_p) % 2 == cb || N_sub[0] == 1) ? subgrid[0] : subgrid[0] - 1;

                for (int x = 0; x < x_u; x++) {

                    complex<double> *destE;
                    complex<double> *AE;
                    complex<double> tmp;
                    int f_x;
                    if ((y + z + t + x_p) % 2 == cb) {
                        f_x = x;
                    } else {
                        f_x = (x + 1) % subgrid[0];
                    }

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     f_x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[0] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {
                    //         {
                    //             tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half *
                    //                   AE[c1 * 3 + c2];
                    //             destE[0 * 3 + c1] += tmp;
                    //             destE[3 * 3 + c1] += flag * (I * tmp);
                    //             tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half *
                    //                   AE[c1 * 3 + c2];
                    //             destE[1 * 3 + c1] += tmp;
                    //             destE[2 * 3 + c1] += flag * (I * tmp);
                    //         }
                    //     }
                    // }

                    U33_P9((fast_complex *) AE, (fast_complex *) srcO, (fast_complex *) destE,
                           flag);
                }
            }
        }
    }

    for (int y = 0; y < subgrid[1]; y++) {
        for (int z = 0; z < subgrid[2]; z++) {
            for (int t = 0; t < subgrid[3]; t++) {
                int x_d = (((y + z + t + x_p) % 2) != cb || N_sub[0] == 1) ? 0 : 1;

                for (int x = x_d; x < subgrid[0]; x++) {
                    complex<double> *destE;
                    complex<double> *AO;
                    complex<double> tmp;

                    int b_x;

                    if ((t + z + y + x_p) % 2 == cb) {
                        b_x = (x - 1 + subgrid[0]) % subgrid[0];
                    } else {
                        b_x = x;
                    }

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     b_x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AO = U.A[0] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + b_x + (1 - cb) * subgrid_vol_cb) *
                             9;

                    U33_P10((fast_complex *) AO, (fast_complex *) srcO, (fast_complex *) destE,
                            flag);

                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {
                    //         tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);

                    //         destE[0 * 3 + c1] += tmp;
                    //         destE[3 * 3 + c1] += flag * (-I * tmp);

                    //         tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);

                    //         destE[1 * 3 + c1] += tmp;
                    //         destE[2 * 3 + c1] += flag * (-I * tmp);
                    //     }
                    // }
                }
            }
        }
    }

    int y_u = (N_sub[1] == 1) ? subgrid[1] : subgrid[1] - 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = 0; y < y_u; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AE;

                    int f_y = (y + 1) % subgrid[1];

                    complex<double> *srcO =
                        src.A +
                        (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                         subgrid[0] * f_y + x + (1 - cb) * subgrid_vol_cb) *
                            12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[1] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    U33_P11((fast_complex *) AE, (fast_complex *) srcO, (fast_complex *) destE,
                            flag);
                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {
                    //         tmp = -(srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
                    //               AE[c1 * 3 + c2];
                    //         destE[0 * 3 + c1] += tmp;
                    //         destE[3 * 3 + c1] += flag * (tmp);
                    //         tmp = -(srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half *
                    //               AE[c1 * 3 + c2];
                    //         destE[1 * 3 + c1] += tmp;
                    //         destE[2 * 3 + c1] -= flag * (tmp);
                    //     }
                    // }
                }
            }
        }
    }

    int y_d = (N_sub[1] == 1) ? 0 : 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = y_d; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    complex<double> *destE;
                    complex<double> *AO;
                    complex<double> tmp;

                    int b_y = (y - 1 + subgrid[1]) % subgrid[1];

                    complex<double> *srcO =
                        src.A +
                        (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                         subgrid[0] * b_y + x + (1 - cb) * subgrid_vol_cb) *
                            12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AO = U.A[1] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * b_y + x + (1 - cb) * subgrid_vol_cb) *
                             9;

                    U33_P12((fast_complex *) AO, (fast_complex *) srcO, (fast_complex *) destE,
                            flag);

                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {
                    //         tmp = -(srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         destE[0 * 3 + c1] += tmp;
                    //         destE[3 * 3 + c1] -= flag * (tmp);
                    //         tmp = -(srcO[1 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         destE[1 * 3 + c1] += tmp;
                    //         destE[2 * 3 + c1] += flag * (tmp);
                    //     }
                    // }
                }
            }
        }
    }

    int z_u = (N_sub[2] == 1) ? subgrid[2] : subgrid[2] - 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < z_u; z++) {
                for (int t = 0; t < subgrid[3]; t++) {

                    int f_z = (z + 1) % subgrid[2];

                    complex<double> tmp;

                    complex<double> *srcO =
                        src.A +
                        (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * f_z +
                         subgrid[0] * y + x + (1 - cb) * subgrid_vol_cb) *
                            12;

                    complex<double> *destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                       subgrid[0] * subgrid[1] * z +
                                                       subgrid[0] * y + x + cb * subgrid_vol_cb) *
                                                          12;

                    complex<double> *AE = U.A[2] + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                    subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                    x + cb * subgrid_vol_cb) *
                                                       9;

                    U33_P13((fast_complex *) AE, (fast_complex *) srcO, (fast_complex *) destE,
                            flag);

                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {

                    //         tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half *
                    //               AE[c1 * 3 + c2];
                    //         destE[0 * 3 + c1] += tmp;
                    //         destE[2 * 3 + c1] += flag * (I * tmp);
                    //         tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half *
                    //               AE[c1 * 3 + c2];
                    //         destE[1 * 3 + c1] += tmp;
                    //         destE[3 * 3 + c1] += flag * (-I * tmp);
                    //     }
                    // }
                }
            }
        }
    }

    int z_d = (N_sub[2] == 1) ? 0 : 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = z_d; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AO;

                    int b_z = (z - 1 + subgrid[2]) % subgrid[2];

                    complex<double> *srcO =
                        src.A +
                        (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * b_z +
                         subgrid[0] * y + x + (1 - cb) * subgrid_vol_cb) *
                            12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AO = U.A[2] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * b_z +
                          subgrid[0] * y + x + (1 - cb) * subgrid_vol_cb) *
                             9;

                    U33_P14((fast_complex *) AO, (fast_complex *) srcO, (fast_complex *) destE,
                            flag);

                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {
                    //         tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         destE[0 * 3 + c1] += tmp;
                    //         destE[2 * 3 + c1] += flag * (-I * tmp);
                    //         tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         destE[1 * 3 + c1] += tmp;
                    //         destE[3 * 3 + c1] += flag * (I * tmp);
                    //     }
                    // }
                }
            }
        }
    }

    int t_u = (N_sub[3] == 1) ? subgrid[3] : subgrid[3] - 1;

    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < t_u; t++) {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AE;

                    int f_t = (t + 1) % subgrid[3];

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * f_t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[3] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    U33_P15((fast_complex *) AE, (fast_complex *) srcO, (fast_complex *) destE,
                            flag);

                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {
                    //         tmp = -(srcO[0 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half *
                    //               AE[c1 * 3 + c2];
                    //         destE[0 * 3 + c1] += tmp;
                    //         destE[2 * 3 + c1] -= flag * (tmp);
                    //         tmp = -(srcO[1 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half *
                    //               AE[c1 * 3 + c2];
                    //         destE[1 * 3 + c1] += tmp;
                    //         destE[3 * 3 + c1] -= flag * (tmp);
                    //     }
                    // }
                }
            }
        }
    }

    int t_d = (N_sub[3] == 1) ? 0 : 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = t_d; t < subgrid[3]; t++) {

                    complex<double> *destE;
                    complex<double> *AO;

                    int b_t = (t - 1 + subgrid[3]) % subgrid[3];

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * b_t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AO = U.A[3] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * b_t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + (1 - cb) * subgrid_vol_cb) *
                             9;

                    complex<double> tmp;

                    U33_P16((fast_complex *) AO, (fast_complex *) srcO, (fast_complex *) destE,
                            flag);

                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {
                    //         tmp = -(srcO[0 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         destE[0 * 3 + c1] += tmp;
                    //         destE[2 * 3 + c1] += flag * (tmp);
                    //         tmp = -(srcO[1 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
                    //               conj(AO[c2 * 3 + c1]);
                    //         destE[1 * 3 + c1] += tmp;
                    //         destE[3 * 3 + c1] += flag * (tmp);
                    //     }
                    // }
                }
            }
        }
    }

    //    printf(" rank =%i  ghost  \n ", rank);

    //////////////////////////////////////////////////////////////////////////////////////ghost//////////////////////////////////////////////////////////////////

    if (N_sub[0] != 1) {

        MPI_Wait(&reqr[8 * nodenum_x_f], &star[8 * nodenum_x_f]);

        int cont = 0;
        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    if ((y + z + t + x_p) % 2 == cb) {
                        continue;
                    }

                    complex<double> *destE;
                    complex<double> *AE;
                    complex<double> tmp;

                    int x = subgrid[0] - 1;

                    complex<double> *srcO = (complex<double> *) (&resv_x_f[cont * 6 * 2]);

                    cont += 1;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[0] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;
                    U33_P17((fast_complex *) AE, (fast_complex *) srcO, (fast_complex *) destE,
                            flag);
                    // for (int c1 = 0; c1 < 3; c1++) {
                    //     for (int c2 = 0; c2 < 3; c2++) {
                    //         tmp = srcO[0 * 3 + c2] * AE[c1 * 3 + c2];
                    //         destE[0 * 3 + c1] += tmp;
                    //         destE[3 * 3 + c1] += flag * (I * tmp);
                    //         tmp = srcO[1 * 3 + c2] * AE[c1 * 3 + c2];
                    //         destE[1 * 3 + c1] += tmp;
                    //         destE[2 * 3 + c1] += flag * (I * tmp);
                    //     }
                    // }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank], &stas[8 * rank]);

    } // if(N_sub[0]!=1)

    //    delete[] send_x_b;
    //    delete[] resv_x_f;

    if (N_sub[0] != 1) {

        MPI_Wait(&reqr[8 * nodenum_x_b + 1], &star[8 * nodenum_x_b + 1]);

        int cont = 0;

        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    if (((y + z + t + x_p) % 2) != cb) {
                        continue;
                    }

                    int x = 0;

                    complex<double> *srcO = (complex<double> *) (&resv_x_b[cont * 6 * 2]);
                    cont += 1;

                    complex<double> *destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                       subgrid[0] * subgrid[1] * z +
                                                       subgrid[0] * y + x + cb * subgrid_vol_cb) *
                                                          12;

                    for (int c1 = 0; c1 < 3; c1++) {
                        destE[0 * 3 + c1] += srcO[0 * 3 + c1];
                        destE[3 * 3 + c1] += flag * (-I * srcO[0 * 3 + c1]);
                        destE[1 * 3 + c1] += srcO[1 * 3 + c1];
                        destE[2 * 3 + c1] += flag * (-I * srcO[1 * 3 + c1]);
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 1], &stas[8 * rank + 1]);

    } // if(N_sub[0]!=1)

    //    delete[] send_x_f;
    //    delete[] resv_x_b;

    if (N_sub[1] != 1) {

        MPI_Wait(&reqr[8 * nodenum_y_f + 2], &star[8 * nodenum_y_f + 2]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AE;

                    int y = subgrid[1] - 1;

                    complex<double> *srcO = (complex<double> *) (&resv_y_f[cont * 6 * 2]);

                    cont += 1;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[1] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = srcO[0 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (tmp);
                            tmp = srcO[1 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] -= flag * (tmp);
                        }
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 2], &stas[8 * rank + 2]);

    } // if(N_sub[1]!=1)

    //    delete[] send_y_b;
    //    delete[] resv_y_f;

    if (N_sub[1] != 1) {

        MPI_Wait(&reqr[8 * nodenum_y_b + 3], &star[8 * nodenum_y_b + 3]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    complex<double> *srcO = (complex<double> *) (&resv_y_b[cont * 6 * 2]);

                    cont += 1;

                    int y = 0;
                    complex<double> *destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                       subgrid[0] * subgrid[1] * z +
                                                       subgrid[0] * y + x + cb * subgrid_vol_cb) *
                                                          12;

                    for (int c1 = 0; c1 < 3; c1++) {
                        destE[0 * 3 + c1] += srcO[0 * 3 + c1];
                        destE[3 * 3 + c1] -= flag * srcO[0 * 3 + c1];
                        destE[1 * 3 + c1] += srcO[1 * 3 + c1];
                        destE[2 * 3 + c1] += flag * srcO[1 * 3 + c1];
                    }
                    //                    for (int i = 0; i < 12; i++) {
                    //                        destE[i] += srcO[i];
                    //                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 3], &stas[8 * rank + 3]);
    }

    //    delete[] send_y_f;
    //    delete[] resv_y_b;

    if (N_sub[2] != 1) {

        MPI_Wait(&reqr[8 * nodenum_z_f + 4], &star[8 * nodenum_z_f + 4]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++) {
            for (int y = 0; y < subgrid[1]; y++) {
                for (int t = 0; t < subgrid[3]; t++) {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AE;

                    int z = subgrid[2] - 1;

                    complex<double> *srcO = (complex<double> *) (&resv_z_f[cont * 6 * 2]);

                    cont += 1;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[2] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = srcO[0 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] += flag * (I * tmp);
                            tmp = srcO[1 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (-I * tmp);
                        }
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 4], &stas[8 * rank + 4]);

    } // if(N_sub[2]!=1)

    //    delete[] send_z_b;
    //    delete[] resv_z_f;

    if (N_sub[2] != 1) {

        MPI_Wait(&reqr[8 * nodenum_z_b + 5], &star[8 * nodenum_z_b + 5]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++) {
            for (int y = 0; y < subgrid[1]; y++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    complex<double> *srcO = (complex<double> *) (&resv_z_b[cont * 6 * 2]);

                    cont += 1;

                    int z = 0;
                    complex<double> *destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                       subgrid[0] * subgrid[1] * z +
                                                       subgrid[0] * y + x + cb * subgrid_vol_cb) *
                                                          12;

                    for (int c1 = 0; c1 < 3; c1++) {
                        destE[0 * 3 + c1] += srcO[0 * 3 + c1];
                        destE[2 * 3 + c1] += flag * (-I * srcO[0 * 3 + c1]);
                        destE[1 * 3 + c1] += srcO[1 * 3 + c1];
                        destE[3 * 3 + c1] += flag * (I * srcO[1 * 3 + c1]);
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 5], &stas[8 * rank + 5]);

    } // if (N_sub[2] != 1)

    // delete[] send_z_f;
    // delete[] resv_z_b;

    if (N_sub[3] != 1) {

        MPI_Wait(&reqr[8 * nodenum_t_f + 6], &star[8 * nodenum_t_f + 6]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++) {
            for (int y = 0; y < subgrid[1]; y++) {
                for (int z = 0; z < subgrid[2]; z++) {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AE;
                    int t = subgrid[3] - 1;

                    complex<double> *srcO = (complex<double> *) (&resv_t_f[cont * 6 * 2]);

                    cont += 1;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[3] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = srcO[0 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] -= flag * (tmp);
                            tmp = srcO[1 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] -= flag * (tmp);
                        }
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 6], &stas[8 * rank + 6]);
    }

    if (N_sub[3] != 1) {

        MPI_Wait(&reqr[8 * nodenum_t_b + 7], &star[8 * nodenum_t_b + 7]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++) {
            for (int y = 0; y < subgrid[1]; y++) {
                for (int z = 0; z < subgrid[2]; z++) {
                    complex<double> *srcO = (complex<double> *) (&resv_t_b[cont * 6 * 2]);

                    cont += 1;
                    int t = 0;
                    complex<double> *destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                       subgrid[0] * subgrid[1] * z +
                                                       subgrid[0] * y + x + cb * subgrid_vol_cb) *
                                                          12;

                    for (int c1 = 0; c1 < 3; c1++) {
                        destE[0 * 3 + c1] += srcO[0 * 3 + c1];
                        destE[2 * 3 + c1] += flag * (srcO[0 * 3 + c1]);
                        destE[1 * 3 + c1] += srcO[1 * 3 + c1];
                        destE[3 * 3 + c1] += flag * (srcO[1 * 3 + c1]);
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 7], &stas[8 * rank + 7]);

    } // if (N_sub[3] != 1)

    MPI_Barrier(MPI_COMM_WORLD);

    delete[] send_x_b;
    delete[] resv_x_f;
    delete[] send_x_f;
    delete[] resv_x_b;

    delete[] send_y_b;
    delete[] resv_y_f;
    delete[] send_y_f;
    delete[] resv_y_b;

    delete[] send_z_b;
    delete[] resv_z_f;
    delete[] send_z_f;
    delete[] resv_z_b;

    delete[] send_t_b;
    delete[] resv_t_f;
    delete[] send_t_f;
    delete[] resv_t_b;
}
