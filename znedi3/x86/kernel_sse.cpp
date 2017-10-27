#ifdef ZNEDI3_X86

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <memory>
#include <xmmintrin.h>
#include "alloc.h"
#include "ccdep.h"
#include "kernel.h"
#include "kernel_x86.h"

#include "kernel_sse_commmon.h"

namespace znedi3 {
namespace {

inline FORCE_INLINE __m128 mm_expf_ps(__m128 x)
{
	const __m128 ln2_inv_scaled_ps = _mm_set_ps1(EXPF_LN2_INV_SCALED);
	const __m128 one_scaled_ps = _mm_set_ps1(EXPF_ONE_SCALED);

	__m128 i, f;

	x = _mm_add_ps(_mm_mul_ps(ln2_inv_scaled_ps, x), one_scaled_ps);
	// x = _mm_min_ps(x, _mm_set_ps1(inf_scaled));
	// x = _mm_max_ps(x, _mm_setzero_ps());

	alignas(16) int32_t tmpi[4];
	tmpi[0] = _mm_cvtt_ss2si(_mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 0, 0, 0)));
	tmpi[1] = _mm_cvtt_ss2si(_mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 1, 1, 1)));
	tmpi[2] = _mm_cvtt_ss2si(_mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 2, 2, 2)));
	tmpi[3] = _mm_cvtt_ss2si(_mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 3, 3)));
	x = _mm_load_ps((const float *)tmpi);

	alignas(16) constexpr uint32_t exp_mask[4] = { 0x7F800000UL, 0x7F800000UL, 0x7F800000UL, 0x7F800000UL };
	alignas(16) constexpr uint32_t mant_mask[4] = { 0x007FFFFFUL, 0x007FFFFFUL, 0x007FFFFFUL, 0x007FFFFFUL };
	alignas(16) constexpr uint32_t one_mask[4] = { 0x3F800000UL, 0x3F800000UL, 0x3F800000UL, 0x3F800000UL };
	// Clear the mantissa. This represents exp2(floor(x)).
	i = _mm_and_ps(x, _mm_load_ps((const float *)exp_mask));
	// Reset the exponent to zero. This represents exp2(x - floor(x)).
	f = _mm_and_ps(x, _mm_load_ps((const float *)mant_mask));
	f = _mm_or_ps(f, _mm_load_ps((const float *)one_mask));

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

	__m128 den = _mm_and_ps(x, _mm_load_ps((const float *)mask));
	den = _mm_add_ps(den, _mm_set_ps1(1.0f));

	return _mm_mul_ps(x, mm_rcp24_ps(den));
}


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

	size_t get_tmp_size() const override { return 0; }

	void process(const void *src, ptrdiff_t src_stride, unsigned char *prescreen, void *tmp, unsigned n) const override
	{
		const InterleavedPrescreenerOldCoefficients &data = m_data.front();

		float *activation = static_cast<float *>(tmp);
		ptrdiff_t activation_stride = 512 * sizeof(float);

		const float *src_p = static_cast<const float *>(src);
		ptrdiff_t src_stride_f = src_stride / sizeof(float);

		// Adjust source pointer to point to top-left of filter window.
		const float *window = src_p - 2 * src_stride_f - 5;

		for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(n); ++j) {
			__m128 accum0 = _mm_setzero_ps();
			__m128 accum1 = _mm_setzero_ps();
			__m128 accum2 = _mm_setzero_ps();
			__m128 accum3 = _mm_setzero_ps();

			// Layer 0.
			for (ptrdiff_t ki = 0; ki < 4; ++ki) {
				for (ptrdiff_t kj = 0; kj < 12; kj += 4) {
					__m128 xtmp = _mm_loadu_ps(window + ki * src_stride_f + j + kj);
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

	size_t get_tmp_size() const override { return 0; }

	void process(const void *src, ptrdiff_t src_stride, unsigned char *prescreen, void *tmp, unsigned n) const override
	{
		const InterleavedPrescreenerNewCoefficients &data = m_data.front();

		const float *src_p = static_cast<const float *>(src);
		ptrdiff_t src_stride_f = src_stride / sizeof(float);

		// Adjust source pointer to point to top-left of filter window.
		const float *window = src_p - 2 * src_stride_f - 6;

		for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(n); j += 4) {
			__m128 accum0 = _mm_setzero_ps();
			__m128 accum1 = _mm_setzero_ps();
			__m128 accum2 = _mm_setzero_ps();
			__m128 accum3 = _mm_setzero_ps();

			// Layer 0.
			for (ptrdiff_t ki = 0; ki < 4; ++ki) {
				for (ptrdiff_t kj = 0; kj < 16; kj += 4) {
					__m128 xtmp = _mm_loadu_ps(window + ki * src_stride_f + j + kj);
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


inline FORCE_INLINE void gather_input_sse(const float *src, ptrdiff_t src_stride, ptrdiff_t xdim, ptrdiff_t ydim, float *buf, float mstd[4], double inv_size)
{
	ptrdiff_t src_stride_f = src_stride / sizeof(float);

	double sum = 0;
	double sum_sq = 0;

	for (ptrdiff_t i = 0; i < ydim; ++i) {
		for (ptrdiff_t j = 0; j < xdim; ++j) {
			float val = src[i * src_stride_f + j];

			buf[i * xdim + j] = val;
			sum += val;
			sum_sq += static_cast<double>(val) * val;
		}
	}

	mstd[0] = static_cast<float>(sum * inv_size);
	mstd[3] = 0.0f;

	double tmp = sum_sq * inv_size - static_cast<double>(mstd[0]) * mstd[0];
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


struct PredictorSSETraits {
	static inline FORCE_INLINE void gather_input(const float *src, ptrdiff_t src_stride, ptrdiff_t xdim, ptrdiff_t ydim, float *buf, float mstd[4], double inv_size)
	{
		gather_input_sse(src, src_stride, xdim, ydim, buf, mstd, inv_size);
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

class PredictorSSE final : public PredictorSSEBase<PredictorSSETraits> {
public:
	using PredictorSSEBase<PredictorSSETraits>::PredictorSSEBase;
};

} // namespace


std::unique_ptr<Prescreener> create_prescreener_old_sse(const PrescreenerOldCoefficients &coeffs, double pixel_half)
{
	return std::make_unique<PrescreenerOldSSE>(coeffs, pixel_half);
}

std::unique_ptr<Prescreener> create_prescreener_new_sse(const PrescreenerNewCoefficients &coeffs, double pixel_half)
{
	return std::make_unique<PrescreenerNewSSE>(coeffs, pixel_half);
}

std::unique_ptr<Predictor> create_predictor_sse(const PredictorModel &model, bool use_q2)
{
	return std::make_unique<PredictorSSE>(model, use_q2);
}

} // namespace znedi3

#endif // ZNEDI3_X86
