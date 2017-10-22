#ifdef ZNEDI3_X86

#include <cstdint>
#include <memory>
#include <emmintrin.h>
#include "kernel.h"
#include "kernel_x86.h"

#include "kernel_sse_commmon.h"

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
	constexpr uint32_t mask[4] = { UINT32_MAX >> 1, UINT32_MAX >> 1, UINT32_MAX >> 1, UINT32_MAX >> 1 };

	__m128 den = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(UINT32_MAX >> 1)));
	den = _mm_add_ps(den, _mm_set_ps1(1.0f));

	return _mm_mul_ps(x, mm_rcp24_ps(den));
}

inline FORCE_INLINE void gather_input_sse2(const float *src, ptrdiff_t src_stride, ptrdiff_t xdim, ptrdiff_t ydim, float *buf, float mstd[4], double inv_size)
{
	ptrdiff_t src_stride_f = src_stride / sizeof(float);

	__m128d sum0 = _mm_setzero_pd();
	__m128d sum1 = _mm_setzero_pd();
	__m128d sumsq0 = _mm_setzero_pd();
	__m128d sumsq1 = _mm_setzero_pd();

	for (unsigned i = 0; i < ydim; ++i) {
		for (unsigned j = 0; j < xdim; j += 4) {
			__m128 val = _mm_loadu_ps(src + j);
			__m128d vald0 = _mm_cvtps_pd(val);
			__m128d vald1 = _mm_cvtps_pd(_mm_shuffle_ps(val, val, _MM_SHUFFLE(1, 0, 3, 2)));

			sum0 = _mm_add_pd(sum0, vald0);
			sum1 = _mm_add_pd(sum1, vald1);
			sumsq0 = _mm_add_pd(sumsq0, _mm_mul_pd(vald0, vald0));
			sumsq1 = _mm_add_pd(sumsq1, _mm_mul_pd(vald1, vald1));

			_mm_store_ps(buf + j, val);
		}
		src += src_stride_f;
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
	const __m128 abs_mask = _mm_set_ps1(*(const float *)&abs_mask_val);
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
	static inline FORCE_INLINE void gather_input(const float *src, ptrdiff_t src_stride, ptrdiff_t xdim, ptrdiff_t ydim, float *buf, float mstd[4], double inv_size)
	{
		gather_input_sse2(src, src_stride, xdim, ydim, buf, mstd, inv_size);
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

} // namespace


std::unique_ptr<Predictor> create_predictor_sse2(const PredictorModel &model, bool use_q2)
{
	return std::make_unique<PredictorSSE2>(model, use_q2);
}

} // namespace znedi3

#endif // ZNEDI3_X86
