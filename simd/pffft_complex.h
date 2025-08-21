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

      constexpr
      complex(const std::array<T, 2>& x)
      : re(x[0]), im(x[1])
      {}

      constexpr
      complex(const std::tuple<T, T>& x)
      : re(std::get<0>(x)), im(std::get<1>(x))
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
      operator*(const complex& a, const T& b)
      { return {a.re * b, a.im * b}; }

      friend constexpr complex
      operator*(const T& a, const complex& b)
      { return {a * b.re, a * b.im}; }

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

      friend constexpr stdx::resize_simd_t<T::size() * 2, T>
      flatten(const complex& a)
      { return stdx::concat(a.re, a.im); }
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

  template <typename T>
    struct in
    {
      const T* ptr;

      const complex<v4sf>&
      operator[](size_t i) const
      {
        const complex<v4sf>* [[gnu::may_alias]] vptr
          = reinterpret_cast<const complex<v4sf>*>(ptr);
        return vptr[i];
      }
    };

  template <typename T>
    struct rev_in
    {
      const T* ptr;

      const complex<v4sf>&
      operator[](size_t i) const
      {
        const complex<v4sf>* [[gnu::may_alias]] vptr
          = reinterpret_cast<const complex<v4sf>*>(ptr);
        return *(vptr - i);
      }
    };

  template <typename V>
    void
    transform_as_complex(size_t N, const typename V::value_type* _in0, typename V::value_type* _out,
                         auto&& fun)
    {
      assert(VALIGNED(_in0) && VALIGNED(_out));
      const complex<V>* [[gnu::may_alias]] in0 = reinterpret_cast<const complex<V>*>(_in0);
      complex<V>* [[gnu::may_alias]] out = reinterpret_cast<complex<V>*>(_out);
      for (size_t i = 0; i < N / V::size(); ++i)
        out[i] = fun(in0[i]);
    }

  template <typename V>
    void
    transform_as_complex(size_t N, const typename V::value_type* _in0,
                         const typename V::value_type* _in1, typename V::value_type* _out,
                         auto&& fun)
    {
      assert(VALIGNED(_in0) && VALIGNED(_in1) && VALIGNED(_out));
      const complex<V>* [[gnu::may_alias]] in0 = reinterpret_cast<const complex<V>*>(_in0);
      const complex<V>* [[gnu::may_alias]] in1 = reinterpret_cast<const complex<V>*>(_in1);
      complex<V>* [[gnu::may_alias]] out = reinterpret_cast<complex<V>*>(_out);
      for (size_t i = 0; i < N / V::size(); ++i)
        out[i] = fun(in0[i], in1[i]);
    }

  template <typename V>
    void
    transform_as_complex(size_t N, const typename V::value_type* _in0,
                         const typename V::value_type* _in1, const typename V::value_type* _in2,
                         typename V::value_type* _out, auto&& fun)
    {
      assert(VALIGNED(_in0) && VALIGNED(_in1) && VALIGNED(_in2) && VALIGNED(_out));
      const complex<V>* [[gnu::may_alias]] in0 = reinterpret_cast<const complex<V>*>(_in0);
      const complex<V>* [[gnu::may_alias]] in1 = reinterpret_cast<const complex<V>*>(_in1);
      const complex<V>* [[gnu::may_alias]] in2 = reinterpret_cast<const complex<V>*>(_in2);
      complex<V>* [[gnu::may_alias]] out = reinterpret_cast<complex<V>*>(_out);
      for (size_t i = 0; i < N / V::size(); ++i)
        out[i] = fun(in0[i], in1[i], in2[i]);
    }

  template <typename V>
    void
    transform_as_complex(size_t N, const V* _in0, const V* _in1, const typename V::value_type* _in2,
                         typename V::value_type* _out, auto&& fun)
    {
      assert(VALIGNED(_in0) && VALIGNED(_in1) && VALIGNED(_in2) && VALIGNED(_out));
      const complex<V>* [[gnu::may_alias]] in0 = reinterpret_cast<const complex<V>*>(_in0);
      const complex<V>* [[gnu::may_alias]] in1 = reinterpret_cast<const complex<V>*>(_in1);
      const complex<V>* [[gnu::may_alias]] in2 = reinterpret_cast<const complex<V>*>(_in2);
      complex<V>* [[gnu::may_alias]] out = reinterpret_cast<complex<V>*>(_out);
      for (size_t i = 0; i < N / V::size(); ++i)
        out[i] = fun(in0[i], in1[i], in2[i]);
    }
}

#endif  // PF_SIMD_COMPLEX_H_
