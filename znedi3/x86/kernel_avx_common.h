#pragma once

#ifdef ZNEDI3_X86

#ifndef ZNEDI3_X86_KERNEL_AVX_COMMON_H_
#define ZNEDI3_X86_KERNEL_AVX_COMMON_H_

#ifdef KERNEL_IMPL_INCLUDED
  #error Must not include multiple impl headers
#endif

#define KERNEL_IMPL_INCLUDED

#ifndef USE_FMA
  #error Must define USE_FMA
#endif

#include <cassert>
#include <cfloat>
#include <cstddef>
#include "alloc.h"
#include "ccdep.h"
#include "kernel.h"
#include "kernel_x86.h"

#if USE_FMA
  #define mm_fmadd_ps _mm_fmadd_ps
  #define mm_fnmadd_ps _mm_fnmadd_ps
  #define mm256_fmadd_ps _mm256_fmadd_ps
  #define mm256_fnmadd_ps _mm256_fnmadd_ps
  #define mm256_fmadd_pd _mm256_fmadd_pd
  #define mm256_fnmadd_pd _mm256_fnmadd_pd
#else
  #define mm_fmadd_ps(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
  #define mm_fnmadd_ps(a, b, c) _mm_sub_ps(c, _mm_mul_ps(a, b))

  #define mm256_fmadd_ps(a, b, c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
  #define mm256_fnmadd_ps(a, b, c) _mm256_sub_ps(c, _mm256_mul_ps(a, b))

  #define mm256_fmadd_pd(a, b, c) _mm256_add_pd(_mm256_mul_pd(a, b), c)
  #define mm256_fnmadd_pd(a, b, c) _mm256_sub_pd(c, _mm256_mul_pd(a, b))
#endif

namespace znedi3 {
namespace {

// Applies a 4x4 transpose to each 128-bit lane.
inline FORCE_INLINE void mm256_transpose2_4x4_ps(__m256 &a, __m256 &b, __m256 &c, __m256 &d)
{
	__m256 t0 = _mm256_shuffle_ps(a, b, 0x44);
	__m256 t1 = _mm256_shuffle_ps(c, d, 0x44);
	__m256 t2 = _mm256_shuffle_ps(a, b, 0xEE);
	__m256 t3 = _mm256_shuffle_ps(c, d, 0xEE);
	a = _mm256_shuffle_ps(t0, t1, 0x88);
	b = _mm256_shuffle_ps(t0, t1, 0xDD);
	c = _mm256_shuffle_ps(t2, t3, 0x88);
	d = _mm256_shuffle_ps(t2, t3, 0xDD);
}

inline FORCE_INLINE void mm256_transpose4_pd(__m256d &a, __m256d &b, __m256d &c, __m256d &d)
{
	__m256d t0 = _mm256_unpacklo_pd(a, b);
	__m256d t1 = _mm256_unpacklo_pd(c, d);
	__m256d t2 = _mm256_unpackhi_pd(a, b);
	__m256d t3 = _mm256_unpackhi_pd(c, d);
	a = _mm256_permute2f128_pd(t0, t1, 0x20);
	b = _mm256_permute2f128_pd(t2, t3, 0x20);
	c = _mm256_permute2f128_pd(t0, t1, 0x31);
	d = _mm256_permute2f128_pd(t2, t3, 0x31);
}

inline FORCE_INLINE void mm256_transpose2_ps128(__m256 &a, __m256 &b)
{
	__m256 t0 = _mm256_permute2f128_ps(a, b, 0x20);
	__m256 t1 = _mm256_permute2f128_ps(a, b, 0x31);
	a = t0;
	b = t1;
}

inline FORCE_INLINE __m128 mm_rsqrt24_ps(__m128 x)
{
	__m128 tmp0 = _mm_rsqrt_ps(x);
	__m128 tmp1 = _mm_mul_ps(x, tmp0);
	__m128 tmp2 = _mm_mul_ps(_mm_set_ps1(0.5f), tmp0);
	__m128 tmp3 = mm_fnmadd_ps(tmp1, tmp0, _mm_set_ps1(3.0f));
	return _mm_mul_ps(tmp2, tmp3);
}

inline FORCE_INLINE __m256 mm256_rcp24_ps(__m256 x)
{
	__m256 tmp0 = _mm256_rcp_ps(x);
	__m256 tmp1 = mm256_fnmadd_ps(x, tmp0, _mm256_set1_ps(1.0f));
	__m256 tmp2 = mm256_fmadd_ps(tmp0, tmp1, tmp0);
	return tmp2;
}

inline FORCE_INLINE __m256 mm256_expf_ps(__m256 x)
{
	__m256 i, f;

	x = mm256_fmadd_ps(_mm256_set1_ps(EXPF_LN2_INV_SCALED), x, _mm256_set1_ps(EXPF_ONE_SCALED));
	// x = _mm256_min_ps(x, _mm256_set1_ps(inf_scaled));
	// x = _mm256_max_ps(x, _mm256_setzero_ps());
	x = _mm256_castsi256_ps(_mm256_cvttps_epi32(x));

	// Clear the mantissa. This represents exp2(floor(x)).
	i = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7F800000UL)));
	// Reset the exponent to zero. This represents exp2(x - floor(x)).
	f = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x007FFFFFUL)));
	f = _mm256_or_ps(f, _mm256_castsi256_ps(_mm256_set1_epi32(0x3F800000UL)));

	x = _mm256_set1_ps(EXP2F_X_PLUS1_REMEZ[4]);
	x = mm256_fmadd_ps(f, x, _mm256_set1_ps(EXP2F_X_PLUS1_REMEZ[3]));
	x = mm256_fmadd_ps(f, x, _mm256_set1_ps(EXP2F_X_PLUS1_REMEZ[2]));
	x = mm256_fmadd_ps(f, x, _mm256_set1_ps(EXP2F_X_PLUS1_REMEZ[1]));
	x = mm256_fmadd_ps(f, x, _mm256_set1_ps(EXP2F_X_PLUS1_REMEZ[0]));

	return _mm256_mul_ps(i, x);
}

inline FORCE_INLINE __m256 mm256_elliott_ps(__m256 x)
{
	const __m256i mask = _mm256_set1_epi32(UINT32_MAX >> 1);

	__m256 den = _mm256_and_ps(x, _mm256_castsi256_ps(mask));
	den = _mm256_add_ps(den, _mm256_set1_ps(1.0f));

	return _mm256_mul_ps(x, mm256_rcp24_ps(den));
}


inline FORCE_INLINE void prescreener_old_layer0_avx(const float kernel[4][48], const float bias[4], const float *window, ptrdiff_t src_stride,
                                                    float *activation, ptrdiff_t activation_stride, unsigned n)
{
	ptrdiff_t activation_stride_f = activation_stride / sizeof(float);

	for (unsigned i = 0; i < n; i += 8) {
		_mm256_store_ps(activation + 0 * activation_stride_f + i, _mm256_setzero_ps());
		_mm256_store_ps(activation + 1 * activation_stride_f + i, _mm256_setzero_ps());
		_mm256_store_ps(activation + 2 * activation_stride_f + i, _mm256_setzero_ps());
		_mm256_store_ps(activation + 3 * activation_stride_f + i, _mm256_setzero_ps());
	}

	// Compute 48x4 convolution.
	for (unsigned k = 0; k < 4; ++k) {
		const float *window_p = window + k * (src_stride / sizeof(float));

		for (unsigned kk = 0; kk < 12; kk += 2) {
			const __m256 n0_c0 = _mm256_broadcast_ss(&kernel[0][12 * k + kk + 0]);
			const __m256 n0_c1 = _mm256_broadcast_ss(&kernel[0][12 * k + kk + 1]);

			const __m256 n1_c0 = _mm256_broadcast_ss(&kernel[1][12 * k + kk + 0]);
			const __m256 n1_c1 = _mm256_broadcast_ss(&kernel[1][12 * k + kk + 1]);

			const __m256 n2_c0 = _mm256_broadcast_ss(&kernel[2][12 * k + kk + 0]);
			const __m256 n2_c1 = _mm256_broadcast_ss(&kernel[2][12 * k + kk + 1]);

			const __m256 n3_c0 = _mm256_broadcast_ss(&kernel[3][12 * k + kk + 0]);
			const __m256 n3_c1 = _mm256_broadcast_ss(&kernel[3][12 * k + kk + 1]);

			for (unsigned i = 0; i < n; i += 8) {
				__m256 x0 = _mm256_loadu_ps(window_p + i + kk);
				__m256 x1 = _mm256_loadu_ps(window_p + i + kk + 1);

				__m256 accum0 = _mm256_load_ps(activation + 0 * activation_stride_f + i);
				__m256 accum1 = _mm256_load_ps(activation + 1 * activation_stride_f + i);
				__m256 accum2 = _mm256_load_ps(activation + 2 * activation_stride_f + i);
				__m256 accum3 = _mm256_load_ps(activation + 3 * activation_stride_f + i);

				accum0 = mm256_fmadd_ps(n0_c0, x0, accum0);
				accum0 = mm256_fmadd_ps(n0_c1, x1, accum0);

				accum1 = mm256_fmadd_ps(n1_c0, x0, accum1);
				accum1 = mm256_fmadd_ps(n1_c1, x1, accum1);

				accum2 = mm256_fmadd_ps(n2_c0, x0, accum2);
				accum2 = mm256_fmadd_ps(n2_c1, x1, accum2);

				accum3 = mm256_fmadd_ps(n3_c0, x0, accum3);
				accum3 = mm256_fmadd_ps(n3_c1, x1, accum3);

				_mm256_store_ps(activation + 0 * activation_stride_f + i, accum0);
				_mm256_store_ps(activation + 1 * activation_stride_f + i, accum1);
				_mm256_store_ps(activation + 2 * activation_stride_f + i, accum2);
				_mm256_store_ps(activation + 3 * activation_stride_f + i, accum3);
			}
		}
	}

	// Add bias and apply elliott function.
	const __m256 bias0 = _mm256_broadcast_ss(bias + 0);
	const __m256 bias1 = _mm256_broadcast_ss(bias + 1);
	const __m256 bias2 = _mm256_broadcast_ss(bias + 2);
	const __m256 bias3 = _mm256_broadcast_ss(bias + 3);

	for (unsigned i = 0; i < n; i += 8) {
		__m256 n0 = _mm256_load_ps(activation + 0 * activation_stride_f + i);
		__m256 n1 = _mm256_load_ps(activation + 1 * activation_stride_f + i);
		__m256 n2 = _mm256_load_ps(activation + 2 * activation_stride_f + i);
		__m256 n3 = _mm256_load_ps(activation + 3 * activation_stride_f + i);

		n0 = _mm256_add_ps(n0, bias0);
		n1 = _mm256_add_ps(n1, bias1);
		n2 = _mm256_add_ps(n2, bias2);
		n3 = _mm256_add_ps(n3, bias3);

		n1 = mm256_elliott_ps(n1);
		n2 = mm256_elliott_ps(n2);
		n3 = mm256_elliott_ps(n3);

		_mm256_store_ps(activation + 0 * activation_stride_f + i, n0);
		_mm256_store_ps(activation + 1 * activation_stride_f + i, n1);
		_mm256_store_ps(activation + 2 * activation_stride_f + i, n2);
		_mm256_store_ps(activation + 3 * activation_stride_f + i, n3);
	}
}

inline FORCE_INLINE void prescreener_old_layer1_avx(const float kernel[4][4], const float bias[4], float *activation, ptrdiff_t activation_stride, unsigned n)
{
	const ptrdiff_t activation_stride_f = activation_stride / sizeof(float);

	for (unsigned k = 0; k < 4; k += 2) {
		const __m256 n0_c0 = _mm256_broadcast_ss(&kernel[k + 0][0]);
		const __m256 n0_c1 = _mm256_broadcast_ss(&kernel[k + 0][1]);
		const __m256 n0_c2 = _mm256_broadcast_ss(&kernel[k + 0][2]);
		const __m256 n0_c3 = _mm256_broadcast_ss(&kernel[k + 0][3]);

		const __m256 n1_c0 = _mm256_broadcast_ss(&kernel[k + 1][0]);
		const __m256 n1_c1 = _mm256_broadcast_ss(&kernel[k + 1][1]);
		const __m256 n1_c2 = _mm256_broadcast_ss(&kernel[k + 1][2]);
		const __m256 n1_c3 = _mm256_broadcast_ss(&kernel[k + 1][3]);

		const __m256 bias0 = _mm256_broadcast_ss(bias + k + 0);
		const __m256 bias1 = _mm256_broadcast_ss(bias + k + 1);

		// Compute 2x4 convolution.
		for (unsigned i = 0; i < n; i += 8) {
			__m256 x0 = _mm256_load_ps(activation + 0 * activation_stride_f + i);
			__m256 x1 = _mm256_load_ps(activation + 1 * activation_stride_f + i);
			__m256 x2 = _mm256_load_ps(activation + 2 * activation_stride_f + i);
			__m256 x3 = _mm256_load_ps(activation + 3 * activation_stride_f + i);

			__m256 accum0 = mm256_fmadd_ps(n0_c0, x0, bias0);
			__m256 accum1 = mm256_fmadd_ps(n1_c0, x0, bias1);

			accum0 = mm256_fmadd_ps(n0_c1, x1, accum0);
			accum1 = mm256_fmadd_ps(n1_c1, x1, accum1);

			accum0 = mm256_fmadd_ps(n0_c2, x2, accum0);
			accum1 = mm256_fmadd_ps(n1_c2, x2, accum1);

			accum0 = mm256_fmadd_ps(n0_c3, x3, accum0);
			accum1 = mm256_fmadd_ps(n1_c3, x3, accum1);

			accum0 = mm256_elliott_ps(accum0);
			accum1 = mm256_elliott_ps(accum1);

			_mm256_store_ps(activation + (4 + k) * activation_stride_f + i, accum0);
			_mm256_store_ps(activation + (5 + k) * activation_stride_f + i, accum1);
		}
	}
}

inline FORCE_INLINE void prescreener_old_layer2_avx(const float kernel[4][8], const float bias[4], float *activation, ptrdiff_t activation_stride,
                                                    unsigned char *prescreen, unsigned n)
{
	const ptrdiff_t activation_stride_f = activation_stride / sizeof(float);

	for (unsigned k = 0; k < 4; ++k) {
		__m256 c0 = _mm256_broadcast_ss(&kernel[k][0]);
		__m256 c1 = _mm256_broadcast_ss(&kernel[k][1]);
		__m256 c2 = _mm256_broadcast_ss(&kernel[k][2]);
		__m256 c3 = _mm256_broadcast_ss(&kernel[k][3]);
		__m256 c4 = _mm256_broadcast_ss(&kernel[k][4]);
		__m256 c5 = _mm256_broadcast_ss(&kernel[k][5]);
		__m256 c6 = _mm256_broadcast_ss(&kernel[k][6]);
		__m256 c7 = _mm256_broadcast_ss(&kernel[k][7]);
		__m256 bias_ps = _mm256_broadcast_ss(bias + k);

		// Compute 1x8 convolution.
		for (unsigned i = 0; i < n; i += 8) {
			__m256 x0 = _mm256_load_ps(activation + 0 * activation_stride_f + i);
			__m256 x1 = _mm256_load_ps(activation + 1 * activation_stride_f + i);
			__m256 x2 = _mm256_load_ps(activation + 2 * activation_stride_f + i);
			__m256 x3 = _mm256_load_ps(activation + 3 * activation_stride_f + i);
			__m256 x4 = _mm256_load_ps(activation + 4 * activation_stride_f + i);
			__m256 x5 = _mm256_load_ps(activation + 5 * activation_stride_f + i);
			__m256 x6 = _mm256_load_ps(activation + 6 * activation_stride_f + i);
			__m256 x7 = _mm256_load_ps(activation + 7 * activation_stride_f + i);
			__m256 accum0, accum1;

			accum0 = mm256_fmadd_ps(c0, x0, bias_ps);
			accum1 = _mm256_mul_ps(c1, x1);
			accum0 = mm256_fmadd_ps(c2, x2, accum0);
			accum1 = mm256_fmadd_ps(c3, x3, accum1);
			accum0 = mm256_fmadd_ps(c4, x4, accum0);
			accum1 = mm256_fmadd_ps(c5, x5, accum1);
			accum0 = mm256_fmadd_ps(c6, x6, accum0);
			accum1 = mm256_fmadd_ps(c7, x7, accum1);

			accum0 = _mm256_add_ps(accum0, accum1);
			_mm256_store_ps(activation + (8 + k) * activation_stride_f + i, accum0);
		}
	}

	// Collapse neurons.
	for (unsigned i = 0; i < n; i += 8) {
		__m256 activation8 = _mm256_load_ps(activation + 8 * activation_stride_f + i);
		__m256 activation9 = _mm256_load_ps(activation + 9 * activation_stride_f + i);
		__m256 activation10 = _mm256_load_ps(activation + 10 * activation_stride_f + i);
		__m256 activation11 = _mm256_load_ps(activation + 11 * activation_stride_f + i);

		activation8 = _mm256_max_ps(activation8, activation9);
		activation10 = _mm256_max_ps(activation10, activation11);

		__m256 mask = _mm256_cmp_ps(activation10, activation8, _CMP_LE_OQ);
		__m128i mask_lo = _mm_castps_si128(_mm256_castps256_ps128(mask));
		__m128i mask_hi = _mm_castps_si128(_mm256_extractf128_ps(mask, 1));

		__m128i prescreen_mask = _mm_packs_epi16(_mm_packs_epi32(mask_lo, mask_hi), _mm_setzero_si128());
		_mm_storel_epi64((__m128i *)(prescreen + i), prescreen_mask);
	}
}


class PrescreenerOldAVX final : public Prescreener {
	PrescreenerOldCoefficients m_data;
public:
	PrescreenerOldAVX(const PrescreenerOldCoefficients &data, double half) :
		m_data(data)
	{
		subtract_mean(m_data, half);
	}

	size_t get_tmp_size() const override
	{
		return 12 * 512 * sizeof(float);
	}

	void process(const void *src, ptrdiff_t src_stride, unsigned char *prescreen, void *tmp, unsigned n) const override
	{
		float *activation = static_cast<float *>(tmp);
		ptrdiff_t activation_stride = 512 * sizeof(float);

		const float *src_p = static_cast<const float *>(src);
		ptrdiff_t src_stride_f = src_stride / sizeof(float);

		// Adjust source pointer to point to top-left of filter window.
		const float *window = src_p - 2 * src_stride_f - 5;

		for (unsigned i = 0; i < n; i += 512) {
			unsigned nn = i + 512 > n ? n - i : 512;

			prescreener_old_layer0_avx(m_data.kernel_l0, m_data.bias_l0, window + i, src_stride, activation, activation_stride, nn);
			prescreener_old_layer1_avx(m_data.kernel_l1, m_data.bias_l1, activation, activation_stride, nn);
			prescreener_old_layer2_avx(m_data.kernel_l2, m_data.bias_l2, activation, activation_stride, prescreen + i, nn);
		}
	}
};


class PrescreenerNewAVX final : public Prescreener {
	AlignedVector<PrescreenerNewCoefficients> m_data;
public:
	PrescreenerNewAVX(const PrescreenerNewCoefficients &data, double half) :
		m_data(1, data)
	{
		subtract_mean(m_data[0], half);
	}

	size_t get_tmp_size() const override { return 0; }

	void process(const void *src, ptrdiff_t src_stride, unsigned char *prescreen, void *, unsigned n) const override
	{
		const PrescreenerNewCoefficients &data = m_data.front();

		const float *src_p = static_cast<const float *>(src);
		ptrdiff_t src_stride_f = src_stride / sizeof(float);

		// Adjust source pointer to point to top-left of filter window.
		const float *window = src_p - 2 * src_stride_f - 6;

		for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(n); j += 8) {
			// Layer 1.
			__m256 x0a, x1a, x2a, x3a;
			__m256 x0b, x1b, x2b, x3b;
			__m256 partial0, partial1;
			__m256 tmp0, tmp1, tmp2, tmp3;

			// Pixels [0-3].
			x0a = _mm256_loadu_ps(window + 0 * src_stride_f + j + 0);
			x1a = _mm256_loadu_ps(window + 1 * src_stride_f + j + 0);
			x2a = _mm256_loadu_ps(window + 2 * src_stride_f + j + 0);
			x3a = _mm256_loadu_ps(window + 3 * src_stride_f + j + 0);

			x0b = _mm256_loadu_ps(window + 0 * src_stride_f + j + 8);
			x1b = _mm256_loadu_ps(window + 1 * src_stride_f + j + 8);
			x2b = _mm256_loadu_ps(window + 2 * src_stride_f + j + 8);
			x3b = _mm256_loadu_ps(window + 3 * src_stride_f + j + 8);

			// x0a-x3a.
			tmp0 = _mm256_mul_ps(_mm256_load_ps(data.kernel_l0[0] + 0), x0a);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 16), x1a, tmp0);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 32), x2a, tmp0);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 48), x3a, tmp0);

			tmp1 = _mm256_mul_ps(_mm256_load_ps(data.kernel_l0[1] + 0), x0a);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 16), x1a, tmp1);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 32), x2a, tmp1);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 48), x3a, tmp1);

			tmp2 = _mm256_mul_ps(_mm256_load_ps(data.kernel_l0[2] + 0), x0a);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 16), x1a, tmp2);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 32), x2a, tmp2);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 48), x3a, tmp2);

			tmp3 = _mm256_mul_ps(_mm256_load_ps(data.kernel_l0[3] + 0), x0a);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 16), x1a, tmp3);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 32), x2a, tmp3);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 48), x3a, tmp3);

			// x0b-x3b.
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 8), x0b, tmp0);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 24), x1b, tmp0);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 40), x2b, tmp0);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 56), x3b, tmp0);

			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 8), x0b, tmp1);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 24), x1b, tmp1);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 40), x2b, tmp1);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 56), x3b, tmp1);

			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 8), x0b, tmp2);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 24), x1b, tmp2);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 40), x2b, tmp2);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 56), x3b, tmp2);

			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 8), x0b, tmp3);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 24), x1b, tmp3);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 40), x2b, tmp3);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 56), x3b, tmp3);

			mm256_transpose2_4x4_ps(tmp0, tmp1, tmp2, tmp3);
			tmp0 = _mm256_add_ps(tmp0, tmp1);
			tmp2 = _mm256_add_ps(tmp2, tmp3);
			partial0 = _mm256_add_ps(tmp0, tmp2);

			// Pixels [4-7].
			x0a = _mm256_loadu_ps(window + 0 * src_stride_f + j + 4);
			x1a = _mm256_loadu_ps(window + 1 * src_stride_f + j + 4);
			x2a = _mm256_loadu_ps(window + 2 * src_stride_f + j + 4);
			x3a = _mm256_loadu_ps(window + 3 * src_stride_f + j + 4);

			x0b = _mm256_loadu_ps(window + 0 * src_stride_f + j + 12);
			x1b = _mm256_loadu_ps(window + 1 * src_stride_f + j + 12);
			x2b = _mm256_loadu_ps(window + 2 * src_stride_f + j + 12);
			x3b = _mm256_loadu_ps(window + 3 * src_stride_f + j + 12);

			// x0a-x3a.
			tmp0 = _mm256_mul_ps(_mm256_load_ps(data.kernel_l0[0] + 0), x0a);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 16), x1a, tmp0);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 32), x2a, tmp0);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 48), x3a, tmp0);

			tmp1 = _mm256_mul_ps(_mm256_load_ps(data.kernel_l0[1] + 0), x0a);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 16), x1a, tmp1);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 32), x2a, tmp1);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 48), x3a, tmp1);

			tmp2 = _mm256_mul_ps(_mm256_load_ps(data.kernel_l0[2] + 0), x0a);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 16), x1a, tmp2);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 32), x2a, tmp2);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 48), x3a, tmp2);

			tmp3 = _mm256_mul_ps(_mm256_load_ps(data.kernel_l0[3] + 0), x0a);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 16), x1a, tmp3);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 32), x2a, tmp3);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 48), x3a, tmp3);

			// x0b-x3b.
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 8), x0b, tmp0);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 24), x1b, tmp0);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 40), x2b, tmp0);
			tmp0 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[0] + 56), x3b, tmp0);

			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 8), x0b, tmp1);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 24), x1b, tmp1);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 40), x2b, tmp1);
			tmp1 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[1] + 56), x3b, tmp1);

			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 8), x0b, tmp2);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 24), x1b, tmp2);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 40), x2b, tmp2);
			tmp2 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[2] + 56), x3b, tmp2);

			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 8), x0b, tmp3);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 24), x1b, tmp3);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 40), x2b, tmp3);
			tmp3 = mm256_fmadd_ps(_mm256_load_ps(data.kernel_l0[3] + 56), x3b, tmp3);

			mm256_transpose2_4x4_ps(tmp0, tmp1, tmp2, tmp3);
			tmp0 = _mm256_add_ps(tmp0, tmp1);
			tmp2 = _mm256_add_ps(tmp2, tmp3);
			partial1 = _mm256_add_ps(tmp0, tmp2);

			// Finish summing neurons.
			mm256_transpose2_ps128(partial0, partial1);
			partial0 = _mm256_add_ps(partial0, partial1);

			__m256 activation_l0 = _mm256_add_ps(partial0, _mm256_broadcast_ps((const __m128 *)data.bias_l0));
			activation_l0 = mm256_elliott_ps(activation_l0);

			// Layer 2.
			tmp0 = _mm256_mul_ps(_mm256_broadcast_ps((const __m128 *)data.kernel_l1[0]), activation_l0);
			tmp1 = _mm256_mul_ps(_mm256_broadcast_ps((const __m128 *)data.kernel_l1[1]), activation_l0);
			tmp2 = _mm256_mul_ps(_mm256_broadcast_ps((const __m128 *)data.kernel_l1[2]), activation_l0);
			tmp3 = _mm256_mul_ps(_mm256_broadcast_ps((const __m128 *)data.kernel_l1[3]), activation_l0);

			mm256_transpose2_4x4_ps(tmp0, tmp1, tmp2, tmp3);
			tmp0 = _mm256_add_ps(tmp0, tmp1);
			tmp2 = _mm256_add_ps(tmp2, tmp3);
			tmp0 = _mm256_add_ps(tmp0, tmp2);

			__m256 activation_l1 = _mm256_add_ps(tmp0, _mm256_broadcast_ps((const __m128 *)data.bias_l1));
			__m256 mask = _mm256_cmp_ps(activation_l1, _mm256_setzero_ps(), _CMP_GT_OQ);

			__m128i mask_lo = _mm_castps_si128(_mm256_castps256_ps128(mask));
			__m128i mask_hi = _mm_castps_si128(_mm256_extractf128_ps(mask, 1));

			__m128i prescreen_mask = _mm_packs_epi16(_mm_packs_epi32(mask_lo, mask_hi), _mm_setzero_si128());
			_mm_storel_epi64((__m128i *)(prescreen + j), prescreen_mask);
		}
	}
};


inline FORCE_INLINE void gather_pixels_avx(const float *src, ptrdiff_t src_stride, ptrdiff_t xdim, ptrdiff_t ydim, float *buf, __m256d *partial_sum_sumsq)
{
	ptrdiff_t src_stride_f = src_stride / sizeof(float);

	__m256d sum0 = _mm256_setzero_pd();
	__m256d sum1 = _mm256_setzero_pd();
	__m256d sumsq0 = _mm256_setzero_pd();
	__m256d sumsq1 = _mm256_setzero_pd();

	for (ptrdiff_t i = 0; i < ydim; ++i) {
		for (ptrdiff_t j = 0; j < xdim; j += 8) {
			__m128 val0 = _mm_loadu_ps(src + j + 0);
			__m128 val1 = _mm_loadu_ps(src + j + 4);

			__m256d vald0 = _mm256_cvtps_pd(val0);
			__m256d vald1 = _mm256_cvtps_pd(val1);

			sum0 = _mm256_add_pd(sum0, vald0);
			sum1 = _mm256_add_pd(sum1, vald1);

			sumsq0 = mm256_fmadd_pd(vald0, vald0, sumsq0);
			sumsq1 = mm256_fmadd_pd(vald1, vald1, sumsq1);

			_mm_store_ps(buf + j + 0, val0);
			_mm_store_ps(buf + j + 4, val1);
		}
		src += src_stride_f;
		buf += xdim;
	}

	partial_sum_sumsq[0] = _mm256_add_pd(sum0, sum1);
	partial_sum_sumsq[1] = _mm256_add_pd(sumsq0, sumsq1);
}

inline FORCE_INLINE void input_stddev_x4_avx(const __m256d *partial_sum_sumsq, float *mstd, double inv_size)
{
	__m256d partial_sum0 = partial_sum_sumsq[0];
	__m256d partial_sum1 = partial_sum_sumsq[2];
	__m256d partial_sum2 = partial_sum_sumsq[4];
	__m256d partial_sum3 = partial_sum_sumsq[6];

	__m256d partial_sumsq0 = partial_sum_sumsq[1];
	__m256d partial_sumsq1 = partial_sum_sumsq[3];
	__m256d partial_sumsq2 = partial_sum_sumsq[5];
	__m256d partial_sumsq3 = partial_sum_sumsq[7];

	mm256_transpose4_pd(partial_sum0, partial_sum1, partial_sum2, partial_sum3);
	mm256_transpose4_pd(partial_sumsq0, partial_sumsq1, partial_sumsq2, partial_sumsq3);

	partial_sum0 = _mm256_add_pd(partial_sum0, partial_sum1);
	partial_sum2 = _mm256_add_pd(partial_sum2, partial_sum3);
	partial_sum0 = _mm256_add_pd(partial_sum0, partial_sum2);

	partial_sumsq0 = _mm256_add_pd(partial_sumsq0, partial_sumsq1);
	partial_sumsq2 = _mm256_add_pd(partial_sumsq2, partial_sumsq3);
	partial_sumsq0 = _mm256_add_pd(partial_sumsq0, partial_sumsq2);

	__m256d sum = partial_sum0;
	__m256d sumsq = partial_sumsq0;

	sum = _mm256_mul_pd(sum, _mm256_set1_pd(inv_size));
	sumsq = _mm256_mul_pd(sumsq, _mm256_set1_pd(inv_size));

	__m256d variance = mm256_fnmadd_pd(sum, sum, sumsq);
	__m128 epislon_mask = _mm256_cvtpd_ps(_mm256_cmp_pd(variance, _mm256_set1_pd(FLT_EPSILON), _CMP_GE_OQ));

	__m128 variance_f32 = _mm256_cvtpd_ps(variance);
	__m128 stddev_inv = mm_rsqrt24_ps(variance_f32);
	__m128 stddev = _mm_mul_ps(stddev_inv, variance_f32);

	stddev_inv = _mm_blendv_ps(_mm_setzero_ps(), stddev_inv, epislon_mask);
	stddev = _mm_blendv_ps(_mm_setzero_ps(), stddev, epislon_mask);

	__m128 mstd0 = _mm256_cvtpd_ps(sum);
	__m128 mstd1 = stddev;
	__m128 mstd2 = stddev_inv;
	__m128 mstd3 = _mm_setzero_ps();

	_mm_store_ps(mstd + 0 * 4, mstd0);
	_mm_store_ps(mstd + 1 * 4, mstd1);
	_mm_store_ps(mstd + 2 * 4, mstd2);
	_mm_store_ps(mstd + 3 * 4, mstd3);
}

inline FORCE_INLINE void sgemv_x4_avx(const float *matrix, const float *vector, const float *bias, unsigned matrix_rows, unsigned matrix_cols,
                                      float *result, unsigned nns, const float *scale)
{
	float *activation_softmax = result;
	float *activation_elliott = result + 4 * static_cast<ptrdiff_t>(nns);

	for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(matrix_rows); i += 16) {
		__m256 accum0a = _mm256_setzero_ps();
		__m256 accum1a = _mm256_setzero_ps();
		__m256 accum2a = _mm256_setzero_ps();
		__m256 accum3a = _mm256_setzero_ps();

		__m256 accum0b = _mm256_setzero_ps();
		__m256 accum1b = _mm256_setzero_ps();
		__m256 accum2b = _mm256_setzero_ps();
		__m256 accum3b = _mm256_setzero_ps();

		for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(matrix_cols); ++j) {
			__m256 x0 = _mm256_broadcast_ss(vector + 0 * static_cast<ptrdiff_t>(matrix_cols) + j);
			__m256 x1 = _mm256_broadcast_ss(vector + 1 * static_cast<ptrdiff_t>(matrix_cols) + j);
			__m256 x2 = _mm256_broadcast_ss(vector + 2 * static_cast<ptrdiff_t>(matrix_cols) + j);
			__m256 x3 = _mm256_broadcast_ss(vector + 3 * static_cast<ptrdiff_t>(matrix_cols) + j);
			__m256 coeffs;

			coeffs = _mm256_load_ps(matrix + j * matrix_rows + i + 0);

			accum0a = mm256_fmadd_ps(coeffs, x0, accum0a);
			accum1a = mm256_fmadd_ps(coeffs, x1, accum1a);
			accum2a = mm256_fmadd_ps(coeffs, x2, accum2a);
			accum3a = mm256_fmadd_ps(coeffs, x3, accum3a);

			coeffs = _mm256_load_ps(matrix + j * matrix_rows + i + 8);

			accum0b = mm256_fmadd_ps(coeffs, x0, accum0b);
			accum1b = mm256_fmadd_ps(coeffs, x1, accum1b);
			accum2b = mm256_fmadd_ps(coeffs, x2, accum2b);
			accum3b = mm256_fmadd_ps(coeffs, x3, accum3b);
		}

		__m256 scale0 = _mm256_broadcast_ss(scale + 0);
		__m256 scale1 = _mm256_broadcast_ss(scale + 1);
		__m256 scale2 = _mm256_broadcast_ss(scale + 2);
		__m256 scale3 = _mm256_broadcast_ss(scale + 3);
		__m256 bias_ps;
		float *dst;

		bias_ps = _mm256_load_ps(bias + i + 0);
		accum0a = mm256_fmadd_ps(scale0, accum0a, bias_ps);
		accum1a = mm256_fmadd_ps(scale1, accum1a, bias_ps);
		accum2a = mm256_fmadd_ps(scale2, accum2a, bias_ps);
		accum3a = mm256_fmadd_ps(scale3, accum3a, bias_ps);

		dst = i + 0 >= static_cast<ptrdiff_t>(nns) ? activation_elliott + (i + 0) - static_cast<ptrdiff_t>(nns) : activation_softmax + i + 0;
		_mm256_store_ps(dst + 0 * static_cast<ptrdiff_t>(nns), accum0a);
		_mm256_store_ps(dst + 1 * static_cast<ptrdiff_t>(nns), accum1a);
		_mm256_store_ps(dst + 2 * static_cast<ptrdiff_t>(nns), accum2a);
		_mm256_store_ps(dst + 3 * static_cast<ptrdiff_t>(nns), accum3a);

		bias_ps = _mm256_load_ps(bias + i + 8);
		accum0b = mm256_fmadd_ps(scale0, accum0b, bias_ps);
		accum1b = mm256_fmadd_ps(scale1, accum1b, bias_ps);
		accum2b = mm256_fmadd_ps(scale2, accum2b, bias_ps);
		accum3b = mm256_fmadd_ps(scale3, accum3b, bias_ps);

		dst = i + 8 >= static_cast<ptrdiff_t>(nns) ? activation_elliott + (i + 8) - static_cast<ptrdiff_t>(nns) : activation_softmax + i + 8;
		_mm256_store_ps(dst + 0 * static_cast<ptrdiff_t>(nns), accum0b);
		_mm256_store_ps(dst + 1 * static_cast<ptrdiff_t>(nns), accum1b);
		_mm256_store_ps(dst + 2 * static_cast<ptrdiff_t>(nns), accum2b);
		_mm256_store_ps(dst + 3 * static_cast<ptrdiff_t>(nns), accum3b);
	}
}

inline FORCE_INLINE void softmax_exp_avx(float *ptr, unsigned n)
{
	const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(UINT32_MAX >> 1));
	const __m256 exp_max = _mm256_set1_ps(80.0f);

	for (unsigned i = 0; i < n; i += 8) {
		__m256 x = _mm256_load_ps(ptr + i);
		__m256 xabs = _mm256_and_ps(abs_mask, x);
		__m256 xsign = _mm256_andnot_ps(abs_mask, x);
		x = _mm256_min_ps(xabs, exp_max);
		x = _mm256_or_ps(xsign, x);
		x = mm256_expf_ps(x);
		_mm256_store_ps(ptr + i, x);
	}
}

inline FORCE_INLINE void wae5_x4_avx(const float *softmax, const float *elliott, unsigned n, float *mstd)
{
	__m256 vsum0 = _mm256_setzero_ps();
	__m256 vsum1 = _mm256_setzero_ps();
	__m256 vsum2 = _mm256_setzero_ps();
	__m256 vsum3 = _mm256_setzero_ps();

	__m256 wsum0 = _mm256_setzero_ps();
	__m256 wsum1 = _mm256_setzero_ps();
	__m256 wsum2 = _mm256_setzero_ps();
	__m256 wsum3 = _mm256_setzero_ps();

	for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); i += 8) {
		__m256 s0 = _mm256_load_ps(softmax + 0 * static_cast<ptrdiff_t>(n) + i);
		__m256 s1 = _mm256_load_ps(softmax + 1 * static_cast<ptrdiff_t>(n) + i);
		__m256 s2 = _mm256_load_ps(softmax + 2 * static_cast<ptrdiff_t>(n) + i);
		__m256 s3 = _mm256_load_ps(softmax + 3 * static_cast<ptrdiff_t>(n) + i);

		__m256 e0 = _mm256_load_ps(elliott + 0 * static_cast<ptrdiff_t>(n) + i);
		__m256 e1 = _mm256_load_ps(elliott + 1 * static_cast<ptrdiff_t>(n) + i);
		__m256 e2 = _mm256_load_ps(elliott + 2 * static_cast<ptrdiff_t>(n) + i);
		__m256 e3 = _mm256_load_ps(elliott + 3 * static_cast<ptrdiff_t>(n) + i);

		e0 = mm256_elliott_ps(e0);
		e1 = mm256_elliott_ps(e1);
		e2 = mm256_elliott_ps(e2);
		e3 = mm256_elliott_ps(e3);

		vsum0 = mm256_fmadd_ps(s0, e0, vsum0);
		vsum1 = mm256_fmadd_ps(s1, e1, vsum1);
		vsum2 = mm256_fmadd_ps(s2, e2, vsum2);
		vsum3 = mm256_fmadd_ps(s3, e3, vsum3);

		wsum0 = _mm256_add_ps(wsum0, s0);
		wsum1 = _mm256_add_ps(wsum1, s1);
		wsum2 = _mm256_add_ps(wsum2, s2);
		wsum3 = _mm256_add_ps(wsum3, s3);
	}

	mm256_transpose2_4x4_ps(vsum0, vsum1, vsum2, vsum3);
	vsum0 = _mm256_add_ps(vsum0, vsum1);
	vsum2 = _mm256_add_ps(vsum2, vsum3);
	vsum0 = _mm256_add_ps(vsum0, vsum2);

	__m128 vsum_reduced = _mm_add_ps(_mm256_castps256_ps128(vsum0), _mm256_extractf128_ps(vsum0, 1));

	mm256_transpose2_4x4_ps(wsum0, wsum1, wsum2, wsum3);
	wsum0 = _mm256_add_ps(wsum0, wsum1);
	wsum2 = _mm256_add_ps(wsum2, wsum3);
	wsum0 = _mm256_add_ps(wsum0, wsum2);

	__m128 wsum_reduced = _mm_add_ps(_mm256_castps256_ps128(wsum0), _mm256_extractf128_ps(wsum0, 1));

	__m128 mask = _mm_cmp_ps(wsum_reduced, _mm_set_ps1(1e-10f), _CMP_GT_OQ);

	// Gather mstd[0] and mstd[1].
	__m128 mstd0 = _mm_load_ps(mstd + 0 * 4);
	__m128 mstd1 = _mm_load_ps(mstd + 1 * 4);
	__m128 mstd3 = _mm_load_ps(mstd + 3 * 4);

	vsum_reduced = _mm_mul_ps(vsum_reduced, _mm_set_ps1(5.0f));
	vsum_reduced = _mm_div_ps(vsum_reduced, wsum_reduced);
	vsum_reduced = mm_fmadd_ps(mstd1, vsum_reduced, mstd0);
	vsum_reduced = _mm_blendv_ps(mstd0, vsum_reduced, mask);

	mstd3 = _mm_add_ps(mstd3, vsum_reduced);
	_mm_store_ps(mstd + 3 * 4, mstd3);
}


class PredictorAVX final : public Predictor {
	InterleavedPredictorModel m_model;
	double m_inv_filter_size;
	bool m_use_q2;

	void apply_model(const float *input, float *activation, float *mstd, const __m256d *partial_sum_sumsq) const
	{
		unsigned filter_size = m_model.xdim * m_model.ydim;
		unsigned nns = m_model.nns;

		float *activation_softmax = activation;
		float *activation_elliott = activation + 4 * nns;

		for (unsigned q = 0; q < (m_use_q2 ? 2U : 1U); ++q) {
			const float *neurons = q ? m_model.neurons_q2 : m_model.neurons_q1;
			const float *bias = q ? m_model.bias_q2 : m_model.bias_q1;

			input_stddev_x4_avx(partial_sum_sumsq, mstd, m_inv_filter_size);
			sgemv_x4_avx(neurons, input, bias, nns * 2, filter_size, activation, nns, mstd + 2 * 4);
			softmax_exp_avx(activation_softmax, 4 * nns);
			wae5_x4_avx(activation_softmax, activation_elliott, nns, mstd);
		}
	}
public:
	PredictorAVX(const PredictorModel &model, bool use_q2) :
		m_model(create_interleaved_predictor_model(model)),
		m_inv_filter_size{ 1.0 / (m_model.xdim * m_model.ydim) },
		m_use_q2{ use_q2 }
	{
		assert(model.first.xdim * model.first.ydim <= 48 * 6);
	}

	size_t get_tmp_size() const override
	{
		FakeAllocator alloc;

		alloc.allocate_n<float>(48 * 6 * 4);
		alloc.allocate_n<float>(256 * 2 * 4);
		alloc.allocate_n<float>(4 * 4);

		return alloc.count();
	}

	void process(const void *src, ptrdiff_t src_stride, void *dst, const unsigned char *prescreen, void *tmp, unsigned n) const override
	{
		LinearAllocator alloc{ tmp };

		const float *src_p = static_cast<const float *>(src);
		float *dst_p = static_cast<float *>(dst);
		ptrdiff_t src_stride_f = src_stride / sizeof(float);

		// Adjust source pointer to point to top-left of filter window.
		const float *window = src_p - static_cast<ptrdiff_t>(m_model.ydim / 2) * src_stride_f - (m_model.xdim / 2 - 1);
		unsigned filter_size = m_model.xdim * m_model.ydim;

		float *input = alloc.allocate_n<float>(48 * 6 * 4);
		float *activation = alloc.allocate_n<float>(256 * 2 * 4);
		float *mstd = alloc.allocate_n<float>(4 * 4);

		__m256d partial_sum_sumsq[8];
		unsigned gathered_idx[4];
		unsigned num_gathered = 0;

		for (unsigned i = 0; i < n; ++i) {
			if (prescreen[i])
				continue;

			gather_pixels_avx(window + i, src_stride, m_model.xdim, m_model.ydim, input + num_gathered * filter_size, partial_sum_sumsq + num_gathered * 2);
			gathered_idx[num_gathered] = i;
			++num_gathered;

			if (num_gathered == 4) {
				apply_model(input, activation, mstd, partial_sum_sumsq);

				dst_p[gathered_idx[0]] = mstd[3 * 4 + 0];
				dst_p[gathered_idx[1]] = mstd[3 * 4 + 1];
				dst_p[gathered_idx[2]] = mstd[3 * 4 + 2];
				dst_p[gathered_idx[3]] = mstd[3 * 4 + 3];

				num_gathered = 0;
			}
		}
		if (num_gathered) {
			apply_model(input, activation, mstd, partial_sum_sumsq);

			for (unsigned idx = 0; idx < num_gathered; ++idx) {
				dst_p[gathered_idx[idx]] = mstd[3 * 4 + idx];
			}
		}
	}
};


void cubic_interpolation_avx_impl(const void *src, ptrdiff_t src_stride, void *dst, const unsigned char *prescreen, unsigned n)
{
	const float *src_p = static_cast<const float *>(src);
	float *dst_p = static_cast<float *>(dst);
	ptrdiff_t src_stride_f = src_stride / sizeof(float);

	const float *src_p0 = src_p - 2 * src_stride_f;
	const float *src_p1 = src_p - 1 * src_stride_f;
	const float *src_p2 = src_p + 0 * src_stride_f;
	const float *src_p3 = src_p + 1 * src_stride_f;

	const __m256 k0 = _mm256_set1_ps(-3.0f / 32.0f);
	const __m256 k1 = _mm256_set1_ps(19.0f / 32.0f);

	for (unsigned i = 0; i < n - (n % 8); i += 8) {
		__m128i masklo = _mm_cvtsi32_si128(*(const uint32_t *)(prescreen + i + 0));
		__m128i maskhi = _mm_cvtsi32_si128(*(const uint32_t *)(prescreen + i + 4));
		masklo = _mm_unpacklo_epi8(masklo, masklo);
		masklo = _mm_unpacklo_epi16(masklo, masklo);
		maskhi = _mm_unpacklo_epi8(maskhi, maskhi);
		maskhi = _mm_unpacklo_epi16(maskhi, maskhi);

		__m256i mask = _mm256_insertf128_si256(_mm256_castsi128_si256(masklo), maskhi, 1);
		__m256 orig = _mm256_load_ps(dst_p + i);
		orig = _mm256_andnot_ps(_mm256_castsi256_ps(mask), orig);

		__m256 accum = _mm256_mul_ps(k0, _mm256_load_ps(src_p0 + i));
		accum = mm256_fmadd_ps(k1, _mm256_load_ps(src_p1 + i), accum);
		accum = mm256_fmadd_ps(k1, _mm256_load_ps(src_p2 + i), accum);
		accum = mm256_fmadd_ps(k0, _mm256_load_ps(src_p3 + i), accum);

		accum = _mm256_and_ps(_mm256_castsi256_ps(mask), accum);
		accum = _mm256_or_ps(orig, accum);

		_mm256_store_ps(dst_p + i, accum);
	}
	for (unsigned i = n - (n % 8); i < n; ++i) {
		if (!prescreen[i])
			continue;

		float accum = 0.0f;
		accum += (-3.0f / 32.0f) * src_p0[i];
		accum += (19.0f / 32.0f) * src_p1[i];
		accum += (19.0f / 32.0f) * src_p2[i];
		accum += (-3.0f / 32.0f) * src_p3[i];

		dst_p[i] = accum;
	}
}

} // namespace
} // namespace znedi3

#endif // ZNEDI3_X86_KERNEL_AVX_COMMON_H_

#endif // ZNEDI3_X86
