#ifdef ZNEDI3_ARM

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <memory>
#include "sse2neon.h"
#include "kernel.h"
#include "kernel_arm.h"
#include "kernel_neon_common.h"

namespace znedi3 {
namespace {

inline FORCE_INLINE __m128d mm_horizontal_sum2_pd(__m128d a, __m128d b)
{
	__m128d lo = _mm_unpacklo_pd(a, b);
	__m128d hi = _mm_unpackhi_pd(a, b);
	return _mm_add_pd(lo, hi);
}

inline FORCE_INLINE __m128 mm_expf_ps(__m128 x)
{
	const __m128 ln2_inv_scaled_ps = _mm_set_ps1(EXPF_LN2_INV_SCALED);
	const __m128 one_scaled_ps = _mm_set_ps1(EXPF_ONE_SCALED);

	__m128 i, f;

	x = _mm_add_ps(_mm_mul_ps(ln2_inv_scaled_ps, x), one_scaled_ps);
	// x = _mm_min_ps(x, _mm_set_ps1(inf_scaled));
	// x = _mm_max_ps(x, _mm_setzero_ps());
	x = _mm_castsi128_ps(_mm_cvttps_epi32(x));

	// Clear the mantissa. This represents exp2(floor(x)).
	i = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(0x7F800000UL)));
	// Reset the exponent to zero. This represents exp2(x - floor(x)).
	f = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(0x007FFFFFUL)));
	f = _mm_or_ps(f, _mm_castsi128_ps(_mm_set1_epi32(0x3F800000UL)));

	x = _mm_set_ps1(EXP2F_X_PLUS1_REMEZ[4]);
	x = _mm_add_ps(_mm_mul_ps(f, x), _mm_set_ps1(EXP2F_X_PLUS1_REMEZ[3]));
	x = _mm_add_ps(_mm_mul_ps(f, x), _mm_set_ps1(EXP2F_X_PLUS1_REMEZ[2]));
	x = _mm_add_ps(_mm_mul_ps(f, x), _mm_set_ps1(EXP2F_X_PLUS1_REMEZ[1]));
	x = _mm_add_ps(_mm_mul_ps(f, x), _mm_set_ps1(EXP2F_X_PLUS1_REMEZ[0]));

	return _mm_mul_ps(i, x);
}

inline FORCE_INLINE __m128 mm_elliott_ps(__m128 x)
{
	__m128 den = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(UINT32_MAX >> 1)));
	den = _mm_add_ps(den, _mm_set_ps1(1.0f));

	return _mm_mul_ps(x, mm_rcp24_ps(den));
}

inline FORCE_INLINE void gather_input_sse2(const float * const *src, ptrdiff_t offset_x, ptrdiff_t xdim, ptrdiff_t ydim, float *buf, float mstd[4], double inv_size)
{
	__m128d sum0 = _mm_setzero_pd();
	__m128d sum1 = _mm_setzero_pd();
	__m128d sumsq0 = _mm_setzero_pd();
	__m128d sumsq1 = _mm_setzero_pd();

	for (ptrdiff_t i = 0; i < ydim; ++i) {
		const float *srcp = src[i];

		for (ptrdiff_t j = 0; j < xdim; j += 4) {
			__m128 val = _mm_loadu_ps(srcp + offset_x + j);
			__m128d vald0 = _mm_cvtps_pd(val);
			__m128d vald1 = _mm_cvtps_pd(_mm_shuffle_ps(val, val, _MM_SHUFFLE(1, 0, 3, 2)));

			sum0 = _mm_add_pd(sum0, vald0);
			sum1 = _mm_add_pd(sum1, vald1);
			sumsq0 = _mm_add_pd(sumsq0, _mm_mul_pd(vald0, vald0));
			sumsq1 = _mm_add_pd(sumsq1, _mm_mul_pd(vald1, vald1));

			_mm_store_ps(buf + j, val);
		}
		buf += xdim;
	}

	sum0 = _mm_add_pd(sum0, sum1);
	sumsq0 = _mm_add_pd(sumsq0, sumsq1);

	__m128d reduced = mm_horizontal_sum2_pd(sum0, sumsq0);
	reduced = _mm_mul_pd(reduced, _mm_set1_pd(inv_size));

	double sum_reduced = _mm_cvtsd_f64(reduced);
	double sumsq_reduced = _mm_cvtsd_f64(_mm_unpackhi_pd(reduced, reduced));

	mstd[0] = static_cast<float>(sum_reduced);
	mstd[3] = 0.0f;

	double tmp = sumsq_reduced - static_cast<double>(mstd[0]) * mstd[0];
	if (tmp < FLT_EPSILON) {
		mstd[1] = 0.0f;
		mstd[2] = 0.0f;
	} else {
		float rsqrt = _mm_cvtss_f32(mm_rsqrt24_ss(_mm_set_ss(static_cast<float>(tmp))));

		mstd[1] = rsqrt * static_cast<float>(tmp);
		mstd[2] = rsqrt;
	}
}

inline FORCE_INLINE void softmax_exp(float *ptr, unsigned n)
{
	const uint32_t abs_mask_val = UINT32_MAX >> 1;
	const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(abs_mask_val));
	const __m128 exp_max = _mm_set_ps1(80.0f);

	for (unsigned i = 0; i < n; i += 4) {
		__m128 x = _mm_load_ps(ptr + i);
		__m128 xabs = _mm_and_ps(abs_mask, x);
		__m128 xsign = _mm_andnot_ps(abs_mask, x);
		x = _mm_min_ps(xabs, exp_max);
		x = _mm_or_ps(xsign, x);
		x = mm_expf_ps(x);
		_mm_store_ps(ptr + i, x);
	}
}

inline FORCE_INLINE void wae5(const float *softmax, const float *elliott, unsigned n, float *mstd)
{
	__m128 vsum = _mm_setzero_ps();
	__m128 wsum = _mm_setzero_ps();

	for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); i += 4) {
		__m128 s = _mm_load_ps(softmax + i);
		__m128 e = _mm_load_ps(elliott + i);
		e = mm_elliott_ps(e);

		vsum = _mm_add_ps(vsum, _mm_mul_ps(s, e));
		wsum = _mm_add_ps(wsum, s);
	}

	__m128 v0 = vsum;
	__m128 v1 = wsum;
	__m128 v2 = _mm_setzero_ps();
	__m128 v3 = _mm_setzero_ps();

	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	v0 = _mm_add_ps(v0, v1);
	v2 = _mm_add_ps(v2, v3);
	v0 = _mm_add_ps(v0, v2);

	float vsum_reduced = _mm_cvtss_f32(v0);
	float wsum_reduced = _mm_cvtss_f32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1)));

	if (wsum_reduced > 1e-10f)
		mstd[3] += (5.0f * vsum_reduced) / wsum_reduced * mstd[1] + mstd[0];
	else
		mstd[3] += mstd[0];
}


struct PredictorSSE2Traits {
	static inline FORCE_INLINE void gather_input(const float * const *src, ptrdiff_t offset_x, ptrdiff_t xdim, ptrdiff_t ydim, float *buf, float mstd[4], double inv_size)
	{
		gather_input_sse2(src, offset_x, xdim, ydim, buf, mstd, inv_size);
	}

	static inline FORCE_INLINE void softmax_exp(float *ptr, unsigned n)
	{
		::znedi3::softmax_exp(ptr, n);
	}

	static inline FORCE_INLINE void wae5(const float *softmax, const float *elliott, unsigned n, float *mstd)
	{
		::znedi3::wae5(softmax, elliott, n, mstd);
	}
};

class PredictorSSE2 final : public PredictorSSEBase<PredictorSSE2Traits> {
public:
	using PredictorSSEBase<PredictorSSE2Traits>::PredictorSSEBase;
};

class PrescreenerOldSSE : public Prescreener {
	struct InterleavedPrescreenerOldCoefficients {
		float kernel_l0[48][4];
		float bias_l0[4];

		float kernel_l1[4][4];
		float bias_l1[4];

		float kernel_l2[8][4];
		float bias_l2[4];
	};

	AlignedVector<InterleavedPrescreenerOldCoefficients> m_data;
public:
	PrescreenerOldSSE(const PrescreenerOldCoefficients &data, double half) :
		m_data(1)
	{
		PrescreenerOldCoefficients d = data;
		subtract_mean(d, half);

		for (unsigned i = 0; i < 48; ++i) {
			for (unsigned n = 0; n < 4; ++n) {
				m_data[0].kernel_l0[i][n] = d.kernel_l0[n][i];
			}
		}
		for (unsigned i = 0; i < 4; ++i) {
			for (unsigned n = 0; n < 4; ++n) {
				m_data[0].kernel_l1[i][n] = d.kernel_l1[n][i];
			}
		}
		for (unsigned i = 0; i < 8; ++i) {
			for (unsigned n = 0; n < 4; ++n) {
				m_data[0].kernel_l2[i][n] = d.kernel_l2[n][i];
			}
		}

		std::copy_n(d.bias_l0, 4, m_data[0].bias_l0);
		std::copy_n(d.bias_l1, 4, m_data[0].bias_l1);
		std::copy_n(d.bias_l2, 4, m_data[0].bias_l2);
	}

	size_t get_tmp_size() const noexcept override { return 0; }

	void process(const float * const src[4], unsigned char *prescreen, void *tmp, unsigned n) const noexcept override
	{
		const InterleavedPrescreenerOldCoefficients &data = m_data.front();
		ptrdiff_t window_offset = 6;

		for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(n); ++j) {
			__m128 accum0 = _mm_setzero_ps();
			__m128 accum1 = _mm_setzero_ps();
			__m128 accum2 = _mm_setzero_ps();
			__m128 accum3 = _mm_setzero_ps();

			// Layer 0.
			for (ptrdiff_t ki = 0; ki < 4; ++ki) {
				const float *srcp = src[ki];

				for (ptrdiff_t kj = 0; kj < 12; kj += 4) {
					__m128 xtmp = _mm_loadu_ps(srcp - window_offset + j + kj);
					__m128 x, coeffs;

					coeffs = _mm_load_ps(data.kernel_l0[ki * 12 + kj + 0]);
					x = _mm_shuffle_ps(xtmp, xtmp, _MM_SHUFFLE(0, 0, 0, 0));
					accum0 = _mm_add_ps(accum0, _mm_mul_ps(coeffs, x));

					coeffs = _mm_load_ps(data.kernel_l0[ki * 12 + kj + 1]);
					x = _mm_shuffle_ps(xtmp, xtmp, _MM_SHUFFLE(1, 1, 1, 1));
					accum1 = _mm_add_ps(accum1, _mm_mul_ps(coeffs, x));

					coeffs = _mm_load_ps(data.kernel_l0[ki * 12 + kj + 2]);
					x = _mm_shuffle_ps(xtmp, xtmp, _MM_SHUFFLE(2, 2, 2, 2));
					accum2 = _mm_add_ps(accum2, _mm_mul_ps(coeffs, x));

					coeffs = _mm_load_ps(data.kernel_l0[ki * 12 + kj + 3]);
					x = _mm_shuffle_ps(xtmp, xtmp, _MM_SHUFFLE(3, 3, 3, 3));
					accum3 = _mm_add_ps(accum3, _mm_mul_ps(coeffs, x));
				}
			}

			accum0 = _mm_add_ps(accum0, accum1);
			accum2 = _mm_add_ps(accum2, accum3);
			accum0 = _mm_add_ps(accum0, accum2);

			alignas(16) constexpr uint32_t l0_mask[4] = { 0, UINT32_MAX, UINT32_MAX, UINT32_MAX };
			__m128 l0 = _mm_add_ps(accum0, _mm_load_ps(data.bias_l0));
			__m128 l0_elliott = _mm_and_ps(_mm_load_ps((const float *)l0_mask), mm_elliott_ps(l0));
			l0 = _mm_andnot_ps(_mm_load_ps((const float *)l0_mask), l0);
			l0 = _mm_or_ps(l0, l0_elliott);

			// Layer 1.
			accum0 = _mm_mul_ps(_mm_load_ps(data.kernel_l1[0]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(0, 0, 0, 0)));
			accum1 = _mm_mul_ps(_mm_load_ps(data.kernel_l1[1]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(1, 1, 1, 1)));
			accum2 = _mm_mul_ps(_mm_load_ps(data.kernel_l1[2]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(2, 2, 2, 2)));
			accum3 = _mm_mul_ps(_mm_load_ps(data.kernel_l1[3]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(3, 3, 3, 3)));

			accum0 = _mm_add_ps(accum0, accum1);
			accum2 = _mm_add_ps(accum2, accum3);
			accum0 = _mm_add_ps(accum0, accum2);

			__m128 l1 = _mm_add_ps(accum0, _mm_load_ps(data.bias_l1));
			l1 = mm_elliott_ps(l1);

			// Layer 2.
			accum0 = _mm_mul_ps(_mm_load_ps(data.kernel_l2[0]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(0, 0, 0, 0)));
			accum1 = _mm_mul_ps(_mm_load_ps(data.kernel_l2[1]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(1, 1, 1, 1)));
			accum2 = _mm_mul_ps(_mm_load_ps(data.kernel_l2[2]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(2, 2, 2, 2)));
			accum3 = _mm_mul_ps(_mm_load_ps(data.kernel_l2[3]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(3, 3, 3, 3)));

			accum0 = _mm_add_ps(accum0, _mm_mul_ps(_mm_load_ps(data.kernel_l2[4]), _mm_shuffle_ps(l1, l1, _MM_SHUFFLE(0, 0, 0, 0))));
			accum1 = _mm_add_ps(accum1, _mm_mul_ps(_mm_load_ps(data.kernel_l2[5]), _mm_shuffle_ps(l1, l1, _MM_SHUFFLE(1, 1, 1, 1))));
			accum2 = _mm_add_ps(accum2, _mm_mul_ps(_mm_load_ps(data.kernel_l2[6]), _mm_shuffle_ps(l1, l1, _MM_SHUFFLE(2, 2, 2, 2))));
			accum3 = _mm_add_ps(accum3, _mm_mul_ps(_mm_load_ps(data.kernel_l2[7]), _mm_shuffle_ps(l1, l1, _MM_SHUFFLE(3, 3, 3, 3))));

			accum0 = _mm_add_ps(accum0, accum1);
			accum2 = _mm_add_ps(accum2, accum3);
			accum0 = _mm_add_ps(accum0, accum2);

			__m128 l2 = _mm_add_ps(accum0, _mm_load_ps(data.bias_l2));
			__m128 l2_swap = _mm_shuffle_ps(l2, l2, _MM_SHUFFLE(2, 3, 0, 1));
			l2 = _mm_max_ps(l2, l2_swap); // max(l2[0], l2[1]) ... max(l2[2], l2[3]) ...

			prescreen[j] = _mm_comile_ss(_mm_shuffle_ps(l2, l2, _MM_SHUFFLE(1, 0, 3, 2)), l2) ? UCHAR_MAX : 0;
		}
	}
};


class PrescreenerNewSSE : public Prescreener {
	struct InterleavedPrescreenerNewCoefficients {
		float kernel_l0[64][4];
		float bias_l0[4];

		float kernel_l1[4][4];
		float bias_l1[4];
	};

	AlignedVector<InterleavedPrescreenerNewCoefficients> m_data;
public:
	PrescreenerNewSSE(const PrescreenerNewCoefficients &data, double half) :
		m_data(1)
	{
		PrescreenerNewCoefficients d = data;
		subtract_mean(d, half);

		for (unsigned i = 0; i < 64; ++i) {
			for (unsigned n = 0; n < 4; ++n) {
				m_data[0].kernel_l0[i][n] = d.kernel_l0[n][i];
			}
		}
		for (unsigned i = 0; i < 4; ++i) {
			for (unsigned n = 0; n < 4; ++n) {
				m_data[0].kernel_l1[i][n] = d.kernel_l1[n][i];
			}
		}

		std::copy_n(d.bias_l0, 4, m_data[0].bias_l0);
		std::copy_n(d.bias_l1, 4, m_data[0].bias_l1);
	}

	size_t get_tmp_size() const noexcept override { return 0; }

	void process(const float * const src[4], unsigned char *prescreen, void *tmp, unsigned n) const noexcept override
	{
		const InterleavedPrescreenerNewCoefficients &data = m_data.front();
		ptrdiff_t window_offset = 6;

		for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(n); j += 4) {
			__m128 accum0 = _mm_setzero_ps();
			__m128 accum1 = _mm_setzero_ps();
			__m128 accum2 = _mm_setzero_ps();
			__m128 accum3 = _mm_setzero_ps();

			// Layer 0.
			for (ptrdiff_t ki = 0; ki < 4; ++ki) {
				const float *srcp = src[ki];

				for (ptrdiff_t kj = 0; kj < 16; kj += 4) {
					__m128 xtmp = _mm_loadu_ps(srcp - window_offset + j + kj);
					__m128 x, coeffs;

					coeffs = _mm_load_ps(data.kernel_l0[ki * 16 + kj + 0]);
					x = _mm_shuffle_ps(xtmp, xtmp, _MM_SHUFFLE(0, 0, 0, 0));
					accum0 = _mm_add_ps(accum0, _mm_mul_ps(coeffs, x));

					coeffs = _mm_load_ps(data.kernel_l0[ki * 16 + kj + 1]);
					x = _mm_shuffle_ps(xtmp, xtmp, _MM_SHUFFLE(1, 1, 1, 1));
					accum1 = _mm_add_ps(accum1, _mm_mul_ps(coeffs, x));

					coeffs = _mm_load_ps(data.kernel_l0[ki * 16 + kj + 2]);
					x = _mm_shuffle_ps(xtmp, xtmp, _MM_SHUFFLE(2, 2, 2, 2));
					accum2 = _mm_add_ps(accum2, _mm_mul_ps(coeffs, x));

					coeffs = _mm_load_ps(data.kernel_l0[ki * 16 + kj + 3]);
					x = _mm_shuffle_ps(xtmp, xtmp, _MM_SHUFFLE(3, 3, 3, 3));
					accum3 = _mm_add_ps(accum3, _mm_mul_ps(coeffs, x));
				}
			}

			accum0 = _mm_add_ps(accum0, accum1);
			accum2 = _mm_add_ps(accum2, accum3);
			accum0 = _mm_add_ps(accum0, accum2);

			__m128 l0 = _mm_add_ps(accum0, _mm_load_ps(data.bias_l0));
			l0 = mm_elliott_ps(l0);

			// Layer1
			accum0 = _mm_mul_ps(_mm_load_ps(data.kernel_l1[0]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(0, 0, 0, 0)));
			accum1 = _mm_mul_ps(_mm_load_ps(data.kernel_l1[1]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(1, 1, 1, 1)));
			accum2 = _mm_mul_ps(_mm_load_ps(data.kernel_l1[2]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(2, 2, 2, 2)));
			accum3 = _mm_mul_ps(_mm_load_ps(data.kernel_l1[3]), _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(3, 3, 3, 3)));

			accum0 = _mm_add_ps(accum0, accum1);
			accum2 = _mm_add_ps(accum2, accum3);
			accum0 = _mm_add_ps(accum0, accum2);

			__m128 l1 = _mm_add_ps(accum0, _mm_load_ps(data.bias_l1));
			l1 = mm_elliott_ps(l1);

			alignas(16) uint32_t prescreen_mask_tmp[4];
			_mm_store_ps((float *)prescreen_mask_tmp, _mm_cmple_ps(_mm_setzero_ps(), l1));

			prescreen[j + 0] = static_cast<uint8_t>(prescreen_mask_tmp[0]);
			prescreen[j + 1] = static_cast<uint8_t>(prescreen_mask_tmp[1]);
			prescreen[j + 2] = static_cast<uint8_t>(prescreen_mask_tmp[2]);
			prescreen[j + 3] = static_cast<uint8_t>(prescreen_mask_tmp[3]);
		}
	}
};

} // namespace


void byte_to_float_sse2(const void *src, void *dst, size_t n)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	for (size_t i = 0; i < n - n % 16; i += 16) {
		__m128i tmp = _mm_load_si128((const __m128i *)(src_p + i));
		__m128i lo = _mm_unpacklo_epi8(tmp, _mm_setzero_si128());
		__m128i hi = _mm_unpackhi_epi8(tmp, _mm_setzero_si128());
		__m128i lolo = _mm_unpacklo_epi16(lo, _mm_setzero_si128());
		__m128i lohi = _mm_unpackhi_epi16(lo, _mm_setzero_si128());
		__m128i hilo = _mm_unpacklo_epi16(hi, _mm_setzero_si128());
		__m128i hihi = _mm_unpackhi_epi16(hi, _mm_setzero_si128());

		_mm_store_ps(dst_p + i + 0, _mm_cvtepi32_ps(lolo));
		_mm_store_ps(dst_p + i + 4, _mm_cvtepi32_ps(lohi));
		_mm_store_ps(dst_p + i + 8, _mm_cvtepi32_ps(hilo));
		_mm_store_ps(dst_p + i + 12, _mm_cvtepi32_ps(hihi));
	}
	for (size_t i = n - n % 16; i < n; ++i) {
		dst_p[i] = src_p[i];
	}
}

void word_to_float_sse2(const void *src, void *dst, size_t n)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		__m128i tmp = _mm_load_si128((const __m128i *)(src_p + i));
		__m128i lo = _mm_unpacklo_epi16(tmp, _mm_setzero_si128());
		__m128i hi = _mm_unpackhi_epi16(tmp, _mm_setzero_si128());

		_mm_store_ps(dst_p + i + 0, _mm_cvtepi32_ps(lo));
		_mm_store_ps(dst_p + i + 4, _mm_cvtepi32_ps(hi));
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		dst_p[i] = src_p[i];
	}
}

void float_to_byte_sse2(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	for (size_t i = 0; i < n - n % 16; i += 16) {
		__m128i lolo = _mm_cvtps_epi32(_mm_load_ps(src_p + i + 0));
		__m128i lohi = _mm_cvtps_epi32(_mm_load_ps(src_p + i + 4));
		__m128i hilo = _mm_cvtps_epi32(_mm_load_ps(src_p + i + 8));
		__m128i hihi = _mm_cvtps_epi32(_mm_load_ps(src_p + i + 12));

		__m128i lo = _mm_packs_epi32(lolo, lohi);
		__m128i hi = _mm_packs_epi32(hilo, hihi);
		__m128i x = _mm_packus_epi16(lo, hi);
		_mm_store_si128((__m128i *)(dst_p + i), x);
	}
	for (size_t i = n - n % 16; i < n; ++i) {
		int32_t x = _mm_cvtss_si32(_mm_set_ss(src_p[i]));
		x = std::min(std::max(x, static_cast<int32_t>(0)), static_cast<int32_t>(UINT8_MAX));
		dst_p[i] = static_cast<uint8_t>(x);
	}
}

void float_to_word_sse2(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		__m128i lo = _mm_cvtps_epi32(_mm_load_ps(src_p + i + 0));
		__m128i hi = _mm_cvtps_epi32(_mm_load_ps(src_p + i + 4));

		lo = _mm_add_epi32(lo, _mm_set1_epi32(INT16_MIN));
		hi = _mm_add_epi32(hi, _mm_set1_epi32(INT16_MIN));

		__m128i x = _mm_packs_epi32(lo, hi);
		x = _mm_sub_epi16(x, _mm_set1_epi16(INT16_MIN));

		_mm_store_si128((__m128i *)(dst_p + i), x);
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		int32_t x = _mm_cvtss_si32(_mm_set_ss(src_p[i]));
		x = std::min(std::max(x, static_cast<int32_t>(0)), static_cast<int32_t>(UINT16_MAX));
		dst_p[i] = static_cast<uint16_t>(x);
	}
}

void cubic_interpolation_sse2(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n)
{
	const float *srcp0 = src[0];
	const float *srcp1 = src[1];
	const float *srcp2 = src[2];
	const float *srcp3 = src[3];

	const __m128 k0 = _mm_set_ps1(-3.0f / 32.0f);
	const __m128 k1 = _mm_set_ps1(19.0f / 32.0f);

	for (unsigned i = 0; i < n; i += 4) {
		__m128i mask = _mm_cvtsi32_si128(*(const uint32_t *)(prescreen + i));
		mask = _mm_unpacklo_epi8(mask, mask);
		mask = _mm_unpacklo_epi16(mask, mask);

		__m128 orig = _mm_load_ps(dst + i);
		orig = _mm_andnot_ps(_mm_castsi128_ps(mask), orig);

		__m128 accum = _mm_mul_ps(k0, _mm_load_ps(srcp0 + i));
		accum = _mm_add_ps(accum, _mm_mul_ps(k1, _mm_load_ps(srcp1 + i)));
		accum = _mm_add_ps(accum, _mm_mul_ps(k1, _mm_load_ps(srcp2 + i)));
		accum = _mm_add_ps(accum, _mm_mul_ps(k0, _mm_load_ps(srcp3 + i)));

		accum = _mm_and_ps(_mm_castsi128_ps(mask), accum);
		accum = _mm_or_ps(orig, accum);

		_mm_store_ps(dst + i, accum);
	}
}


std::unique_ptr<Predictor> create_predictor_sse2(const PredictorModel &model, bool use_q2)
{
	return std::make_unique<PredictorSSE2>(model, use_q2);
}

std::unique_ptr<Prescreener> create_prescreener_old_sse(const PrescreenerOldCoefficients &coeffs, double pixel_half)
{
	return std::make_unique<PrescreenerOldSSE>(coeffs, pixel_half);
}

std::unique_ptr<Prescreener> create_prescreener_new_sse(const PrescreenerNewCoefficients &coeffs, double pixel_half)
{
	return std::make_unique<PrescreenerNewSSE>(coeffs, pixel_half);
}

} // namespace znedi3

#endif // ZNEDI3_ARM
