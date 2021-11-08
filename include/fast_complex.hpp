#ifndef FAST_COMPLEX_H
#define FAST_COMPLEX_H

#include <iostream>
#include <complex>

struct fast_complex {
    double r, i;
    inline fast_complex() {}
    inline fast_complex(const double r) : r(r), i(0) {}
    inline fast_complex(const double r, const double i) : r(r), i(i) {}
    inline double real() { return r; }
    inline double imag() { return i; }
    inline void real(double r) { this->r = r; }
    inline void imag(double i) { this->i = i; }
    // inline fast_complex operator=(const int &b)
    // {
    //     r = b, i = 0;
    //     return *this;
    // }
    // inline fast_complex operator=(const double &b)
    // {
    //     r = b, i = 0;
    //     return *this;
    // }
    // inline fast_complex operator=(const fast_complex &b)
    // {
    //     r = b.r, i = b.i;
    //     return *this;
    // }
    inline fast_complex operator+=(const fast_complex &b)
    {
        r += b.r, i += b.i;
        return *this;
    }
    inline fast_complex operator-=(const fast_complex &b)
    {
        r -= b.r, i -= b.i;
        return *this;
    }
    inline friend std::ostream &operator<<(std::ostream &out, const fast_complex &a)
    {
        out << "(" << a.r << "," << a.i << ")";
        return out;
    }
};

inline fast_complex conj(const fast_complex &a)
{
    return fast_complex(a.r, -a.i);
}

inline fast_complex operator+(const fast_complex &a, const fast_complex &b)
{
    return fast_complex(a.r + b.r, a.i + b.i);
}
inline fast_complex operator+(const double &a, const fast_complex &b)
{
    return fast_complex(a + b.r, b.i);
}
inline fast_complex operator+(const fast_complex &a, const double &b)
{
    return fast_complex(a.r + b, a.i);
}

inline fast_complex operator-(const fast_complex &a, const fast_complex &b)
{
    return fast_complex(a.r - b.r, a.i - b.i);
}
inline fast_complex operator-(const double &a, const fast_complex &b)
{
    return fast_complex(a - b.r, -b.i);
}
inline fast_complex operator-(const fast_complex &a, const double &b)
{
    return fast_complex(a.r - b, a.i);
}
inline fast_complex operator-(const fast_complex &a)
{
    return fast_complex(-a.r, -a.i);
}

inline fast_complex operator*(const fast_complex &a, const fast_complex &b)
{
    return fast_complex(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}
inline fast_complex operator*(const fast_complex &a, const double &b)
{
    return fast_complex(a.r * b, a.i * b);
}
inline fast_complex operator*(const double &a, const fast_complex &b)
{
    return fast_complex(a * b.r, a * b.i);
}

inline fast_complex operator/(const fast_complex &a, const fast_complex &b)
{
    const double t = b.r * b.r + b.i * b.i;
    return fast_complex((a.r * b.r + a.i * b.i) / t, (a.i * b.r - a.r * b.i) / t);
}

#include <immintrin.h>

inline __m256d complex_256_add(__m256d vec1, __m256d vec2)
{
    return _mm256_add_pd(vec1, vec2);
}

inline __m256d complex_256_sub(__m256d vec1, __m256d vec2)
{
    return _mm256_sub_pd(vec1, vec2);
}
// vec1 = ( a, b, x, y )
// vec2 = ( c, d, z, w )
// out  = ( ac-bd, ad+bc, xz-yw, xw+yz )
inline __m256d complex_256_mul(const __m256d vec1, const __m256d vec2)
{
    // std::complex<double> out[2];

    // (a, b, x, y) => (b, b, y, y)
    __m256d tmp = _mm256_permute_pd(vec1, 0b1111);

    // _mm256_storeu_pd((double *) out, tmp);
    // std::cout << out[0] << out[1] << std::endl;

    // res = ( bc, bd, yz, yw )
    __m256d res = _mm256_mul_pd(tmp, vec2);

    // _mm256_storeu_pd((double *) out, res);
    // std::cout << out[0] << out[1] << std::endl;

    // res = ( bd, bc, yw, yz )
    res = _mm256_permute_pd(res, 0b0101);

    // _mm256_storeu_pd((double *) out, res);
    // std::cout << out[0] << out[1] << std::endl;

    // (a, b, x, y) => ( a, a, x, x )
    tmp = _mm256_permute_pd(vec1, 0b0000);

    // _mm256_storeu_pd((double *) out, tmp);
    // std::cout << out[0] << out[1] << std::endl;
    
    // tmp * vec2 = ( ac, ad, xz, xw )
    // out  = ( ac-bd, ad+bc, xz-yw, xw+yz )
    return _mm256_fmaddsub_pd(tmp, vec2, res);
}

#endif
