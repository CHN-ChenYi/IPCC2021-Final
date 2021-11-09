/**
** @file:  operator_mpi.h
** @brief:
**/

#ifndef LATTICE_OPERATOR_MPI_H
#define LATTICE_OPERATOR_MPI_H

#include <mpi.h>
#include <complex>
#include "fast_complex.hpp"

const __m256d odd_vec = {1.0, -1.0, 1.0, -1.0};

template <typename T>
double norm_2(const T &s)
{
    //    int rank;
    //    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //if (rank==0){
    // std::complex<double> s1(0.0, 0.0);
    __m256d s1 = _mm256_setzero_pd();
    // for (int i = 0; i < s.size; i++) {
    //     s1 += s.A[i] * conj(s.A[i]);
    // }
    for (int i = 0; i < s.size; i += 2) {
        __m256d vec1 = _mm256_load_pd((double *) &s.A[i]);
        __m256d vec2 = _mm256_mul_pd(vec1, odd_vec);
        s1 += complex_256_mul(vec1, vec2);
    }
    __m128d res1 = _mm256_extractf128_pd(s1, 0);
    __m128d res2 = _mm256_extractf128_pd(s1, 1);
    res1 = _mm_add_pd(res1, res2);
    double sum_n = _mm_cvtsd_f64(res1);
    // double sum_n = s1.real();
    double sum;
    MPI_Reduce(&sum_n, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    return sum;

    //    return s1.real();
    //    }else
    //    {return 0;}
}

template <typename T>
std::complex<double> vector_p(const T &r1, const T &r2)
{
    // std::complex<double> s1(0.0, 0.0);
    // for (int i = 0; i < r1.size; i++) {
    //     s1 += (conj(r1.A[i]) * r2.A[i]);
    // }
    __m256d s1 = _mm256_setzero_pd();
    for (int i = 0; i < r1.size; i += 2) {
        __m256d vec2 = _mm256_load_pd((double *) &r2.A[i]);
        __m256d vec1 = _mm256_mul_pd(_mm256_load_pd((double *) &r1.A[i]), odd_vec);
        s1 += complex_256_mul(vec1, vec2);
    }
    __m128d res1 = _mm256_extractf128_pd(s1, 0);
    __m128d res2 = _mm256_extractf128_pd(s1, 1);
    res1 = _mm_add_pd(res1, res2);
    fast_complex sum_inter;
    _mm_store_pd((double *) &sum_inter, res1);
    // double sum_r = s1.real(); //fix
    // double sum_i = s1.imag();
    double sumr;
    double sumi;
    MPI_Reduce(&sum_inter.r, &sumr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_inter.i, &sumi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sumr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sumi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    std::complex<double> sum(sumr, sumi);
    //sum.real()= sumr;
    //sum.imag()= sumi;
    return sum;
};

#endif //LATTICE_OPERATOR_MPI_H
