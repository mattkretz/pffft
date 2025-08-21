/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright © 2025      GSI Helmholtzzentrum fuer Schwerionenforschung GmbH
 *                       Matthias Kretz <m.kretz@gsi.de>
 */

#ifndef PF_SIMD_COMPLEX_H_
#define PF_SIMD_COMPLEX_H_

#include <vir/simd.h>

namespace stdx = vir::stdx;

namespace pffft
{
  struct imaginary_constant
  {};

  inline constexpr imaginary_constant I = {};

  template <typename T>
    class complex
    {
      T re, im;

    public:
      constexpr T
      real() const
      { return re; }

      constexpr T
      imag() const
      { return im; }

      constexpr void
      real(T x)
      { re = x; }

      constexpr void
      imag(T x)
      { im = x; }

      constexpr
      complex() = default;

      constexpr
      complex(T r, T i = {})
      : re(r), im(i)
      {}

      friend constexpr complex
      operator+(const complex& a, const complex& b)
      { return {a.re + b.re, a.im + b.im}; }

      friend constexpr complex
      operator-(const complex& a, const complex& b)
      { return {a.re - b.re, a.im - b.im}; }

      friend constexpr complex
      operator*(const complex& a, const complex& b)
      { return {a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im}; }

      friend constexpr complex
      operator*(const complex& a, imaginary_constant)
      { return {-a.im, a.re}; }

      friend constexpr complex
      operator*(imaginary_constant, const complex& a)
      { return {-a.im, a.re}; }

      friend constexpr complex&
      operator+=(complex& a, const complex& b)
      { a = a + b; return a; }

      friend constexpr complex&
      operator-=(complex& a, const complex& b)
      { a = a - b; return a; }

      friend constexpr complex&
      operator*=(complex& a, const complex& b)
      { a = a * b; return a; }

      friend constexpr complex
      conj(const complex& a)
      { return {a.re, -a.im}; }
    };

  template <typename T, typename Abi>
    constexpr T*
    store_unchecked(const complex<stdx::simd<T, Abi>>& c, T* addr,
                    auto flags = stdx::element_aligned)
    {
      c.real().copy_to(addr, flags);
      c.imag().copy_to(addr + stdx::simd_size_v<T, Abi>, flags);
      return addr + 2 * stdx::simd_size_v<T, Abi>;
    }

  template <typename V>
    void
    transform_as_complex(size_t N, const typename V::value_type* in0, typename V::value_type* out,
                         auto&& fun)
    {
      size_t off = 0;
      while (off + V::size() <= N)
        {
          complex<V> x0;
          std::memcpy(&x0, in0 + off, sizeof(x0));
          complex<V> r = fun(x0);
          std::memcpy(out0 + off, &r, sizeof(r));
          off += V::size();
        }
    }

  template <typename V>
    void
    transform_as_complex(size_t N, const typename V::value_type* in0,
                         const typename V::value_type* in1, typename V::value_type* out, auto&& fun)
    {
      size_t off = 0;
      while (off + V::size() <= N)
        {
          complex<V> x0, x1;
          std::memcpy(&x0, in0 + off, sizeof(x0));
          std::memcpy(&x1, in1 + off, sizeof(x0));
          complex<V> r = fun(x0, x1);
          std::memcpy(out0 + off, &r, sizeof(r));
          off += V::size();
        }
    }
}

#endif  // PF_SIMD_COMPLEX_H_
