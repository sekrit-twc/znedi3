#ifdef ZNEDI3_X86_AVX512

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include "ccdep.h"
#include "kernel.h"
#include "kernel_x86.h"

namespace znedi3 {
namespace {

inline FORCE_INLINE float mm512_horizontal_sum_ps(__m512 x)
{
	__m256 stage1 = _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
	__m256 stage2 = _mm256_hadd_ps(stage1, stage1);
	__m256 stage3 = _mm256_hadd_ps(stage2, stage2);
	__m128 stage4 = _mm_add_ss(_mm256_castps256_ps128(stage3), _mm256_extractf128_ps(stage3, 1));
	return _mm_cvtss_f32(stage4);
}

inline FORCE_INLINE double mm512d_horizontal_sum_pd(__m512d x)
{
	__m256d stage1 = _mm256_add_pd(_mm512_castpd512_pd256(x), _mm512_extractf64x4_pd(x, 1));
	__m256d stage2 = _mm256_hadd_pd(stage1, stage1);
	__m128d stage3 = _mm_add_sd(_mm256_castpd256_pd128(stage2), _mm256_extractf128_pd(stage2, 1));
	return _mm_cvtsd_f64(stage3);
}

inline FORCE_INLINE __m512 mm512_elliott_ps(__m512 x)
{
	const __m512i mask = _mm512_set1_epi32(UINT32_MAX >> 1);

	__m512 den = _mm512_and_ps(x, _mm512_castsi512_ps(mask));
	den = _mm512_add_ps(den, _mm512_set1_ps(1.0f));

	return _mm512_div_ps(x, den);
}

inline FORCE_INLINE void gather_pixels_avx512(const float *src, ptrdiff_t src_stride, ptrdiff_t xdim, ptrdiff_t ydim, float *buf, double inv_size, float mstd[4])
{
	ptrdiff_t src_stride_f = src_stride / sizeof(float);

	__m512d sum = _mm512_setzero_pd();
	__m512d sumsq = _mm512_setzero_pd();

	for (ptrdiff_t i = 0; i < ydim; ++i) {
		for (ptrdiff_t j = 0; j < xdim; j += 8) {
			__m256 val = _mm256_load_ps(src + j);

			__m512d vald = _mm512_cvtps_pd(val);
			sum = _mm512_add_pd(sum, vald);
			sumsq = _mm512_fmadd_pd(vald, vald, sumsq);

			_mm256_store_ps(buf + j, val);
		}
		src += src_stride_f;
		buf += xdim;
	}

	// Get horizontal sums.
	double sum_reduced = mm512d_horizontal_sum_pd(sum);
	double sumsq_reduced = mm512d_horizontal_sum_pd(sumsq);

	mstd[0] = static_cast<float>(sum_reduced * inv_size);
	mstd[3] = 0.0f;

	double tmp = sumsq_reduced * inv_size - static_cast<double>(mstd[0]) * mstd[0];
	if (tmp < FLT_EPSILON) {
		mstd[1] = 0.0f;
		mstd[2] = 0.0f;
	} else {
		mstd[1] = static_cast<float>(_mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(tmp))));
		mstd[2] = 1.0f / mstd[1];
	}
}

inline FORCE_INLINE void zero_memory_avx512(float *ptr, unsigned n)
{
	for (unsigned i = 0; i < n; i += 32) {
		_mm512_store_ps(ptr + i + 0, _mm512_setzero_ps());
		_mm512_store_ps(ptr + i + 16, _mm512_setzero_ps());
	}
}

void interleaved_convolution_avx512(const float *kernels, const float *input, unsigned nns, unsigned n, float *output, float scale, const float *bias)
{
	const float *kptr0 = kernels;
	const float *kptr1 = kernels + 1 * static_cast<ptrdiff_t>(nns) * 2;
	const float *kptr2 = kernels + 2 * static_cast<ptrdiff_t>(nns) * 2;
	const float *kptr3 = kernels + 3 * static_cast<ptrdiff_t>(nns) * 2;
	const float *kptr4 = kernels + 4 * static_cast<ptrdiff_t>(nns) * 2;
	const float *kptr5 = kernels + 5 * static_cast<ptrdiff_t>(nns) * 2;
	const float *kptr6 = kernels + 6 * static_cast<ptrdiff_t>(nns) * 2;
	const float *kptr7 = kernels + 7 * static_cast<ptrdiff_t>(nns) * 2;

	for (ptrdiff_t k = 0; k < n; k += 8) {
		__m512 x0 = _mm512_set1_ps(input[k + 0]);
		__m512 x1 = _mm512_set1_ps(input[k + 1]);
		__m512 x2 = _mm512_set1_ps(input[k + 2]);
		__m512 x3 = _mm512_set1_ps(input[k + 3]);
		__m512 x4 = _mm512_set1_ps(input[k + 4]);
		__m512 x5 = _mm512_set1_ps(input[k + 5]);
		__m512 x6 = _mm512_set1_ps(input[k + 6]);
		__m512 x7 = _mm512_set1_ps(input[k + 7]);

		for (ptrdiff_t nn = 0; nn < nns * 2; nn += 32) {
			__m512 n00_15_a = _mm512_load_ps(output + nn);
			__m512 n16_31_a = _mm512_load_ps(output + nn + 16);
			__m512 n00_15_b = _mm512_setzero_ps();
			__m512 n16_31_b = _mm512_setzero_ps();
			__m512 c;

			c = _mm512_load_ps(kptr0 + nn + 0);
			n00_15_a = _mm512_fmadd_ps(c, x0, n00_15_a);

			c = _mm512_load_ps(kptr0 + nn + 16);
			n16_31_a = _mm512_fmadd_ps(c, x0, n16_31_a);

			c = _mm512_load_ps(kptr1 + nn + 0);
			n00_15_b = _mm512_fmadd_ps(c, x1, n00_15_b);

			c = _mm512_load_ps(kptr1 + nn + 16);
			n16_31_b = _mm512_fmadd_ps(c, x1, n16_31_b);

			c = _mm512_load_ps(kptr2 + nn + 0);
			n00_15_a = _mm512_fmadd_ps(c, x2, n00_15_a);

			c = _mm512_load_ps(kptr2 + nn + 16);
			n16_31_a = _mm512_fmadd_ps(c, x2, n16_31_a);

			c = _mm512_load_ps(kptr3 + nn + 0);
			n00_15_b = _mm512_fmadd_ps(c, x3, n00_15_b);

			c = _mm512_load_ps(kptr3 + nn + 16);
			n16_31_b = _mm512_fmadd_ps(c, x3, n16_31_b);

			c = _mm512_load_ps(kptr4 + nn + 0);
			n00_15_a = _mm512_fmadd_ps(c, x4, n00_15_a);

			c = _mm512_load_ps(kptr4 + nn + 16);
			n16_31_a = _mm512_fmadd_ps(c, x4, n16_31_a);

			c = _mm512_load_ps(kptr5 + nn + 0);
			n00_15_b = _mm512_fmadd_ps(c, x5, n00_15_b);

			c = _mm512_load_ps(kptr5 + nn + 16);
			n16_31_b = _mm512_fmadd_ps(c, x5, n16_31_b);

			c = _mm512_load_ps(kptr6 + nn + 0);
			n00_15_a = _mm512_fmadd_ps(c, x6, n00_15_a);

			c = _mm512_load_ps(kptr6 + nn + 16);
			n16_31_a = _mm512_fmadd_ps(c, x6, n16_31_a);

			c = _mm512_load_ps(kptr7 + nn + 0);
			n00_15_b = _mm512_fmadd_ps(c, x7, n00_15_b);

			c = _mm512_load_ps(kptr7 + nn + 16);
			n16_31_b = _mm512_fmadd_ps(c, x7, n16_31_b);

			n00_15_a = _mm512_add_ps(n00_15_a, n00_15_b);
			n16_31_a = _mm512_add_ps(n16_31_a, n16_31_b);

			_mm512_store_ps(output + nn, n00_15_a);
			_mm512_store_ps(output + nn + 16, n16_31_a);
		}

		kptr0 += 8 * static_cast<ptrdiff_t>(nns) * 2;
		kptr1 += 8 * static_cast<ptrdiff_t>(nns) * 2;
		kptr2 += 8 * static_cast<ptrdiff_t>(nns) * 2;
		kptr3 += 8 * static_cast<ptrdiff_t>(nns) * 2;
		kptr4 += 8 * static_cast<ptrdiff_t>(nns) * 2;
		kptr5 += 8 * static_cast<ptrdiff_t>(nns) * 2;
		kptr6 += 8 * static_cast<ptrdiff_t>(nns) * 2;
		kptr7 += 8 * static_cast<ptrdiff_t>(nns) * 2;
	}
	for (ptrdiff_t nn = 0; nn < nns * 2; nn += 16) {
		__m512 accum = _mm512_load_ps(output + nn);
		accum = _mm512_fmadd_ps(_mm512_set1_ps(scale), accum, _mm512_load_ps(bias + nn));
		_mm512_store_ps(output + nn, accum);
	}
}

inline FORCE_INLINE __m512 mm512_expf_ps(__m512 x)
{
	constexpr float exp2f_x_plus1_remez[5] = {
		0.509871020343597804469416f,
		0.312146713032169896138863f,
		0.166617139319965966118107f,
		-2.19061993049215080032874e-3f,
		1.3555747234758484073940937e-2f
	};
	constexpr float ln2_inv_scaled = 12102203.1615614f;
	constexpr float one_scaled = 1065353216.f;
	// constexpr float inf_scaled = 2139095040.f;

	__m512 i, f;

	x = _mm512_fmadd_ps(_mm512_set1_ps(EXPF_LN2_INV_SCALED), x, _mm512_set1_ps(EXPF_ONE_SCALED));
	// x = _mm512_min_ps(x, _mm512_set1_ps(inf_scaled));
	// x = _mm512_max_ps(x, _mm512_setzero_ps());
	x = _mm512_castsi512_ps(_mm512_cvttps_epi32(x));

	// Clear the mantissa. This represents exp2(floor(x)).
	i = _mm512_and_ps(x, _mm512_castsi512_ps(_mm512_set1_epi32(0x7F800000UL)));
	// Reset the exponent to zero. This represents exp2(x - floor(x)).
	f = _mm512_and_ps(x, _mm512_castsi512_ps(_mm512_set1_epi32(0x007FFFFFUL)));
	f = _mm512_or_ps(f, _mm512_castsi512_ps(_mm512_set1_epi32(0x3F800000UL)));

	x = _mm512_set1_ps(EXP2F_X_PLUS1_REMEZ[4]);
	x = _mm512_fmadd_ps(f, x, _mm512_set1_ps(EXP2F_X_PLUS1_REMEZ[3]));
	x = _mm512_fmadd_ps(f, x, _mm512_set1_ps(EXP2F_X_PLUS1_REMEZ[2]));
	x = _mm512_fmadd_ps(f, x, _mm512_set1_ps(EXP2F_X_PLUS1_REMEZ[1]));
	x = _mm512_fmadd_ps(f, x, _mm512_set1_ps(EXP2F_X_PLUS1_REMEZ[0]));

	return _mm512_mul_ps(i, x);
}

inline FORCE_INLINE void softmax_exp(float *ptr, unsigned n)
{
	const __m512 exp_min = _mm512_set1_ps(-80.0f);
	const __m512 exp_max = _mm512_set1_ps(80.0f);

	for (unsigned i = 0; i < n; i += 16) {
		__m512 x = _mm512_load_ps(ptr + i);
		x = _mm512_max_ps(x, exp_min);
		x = _mm512_min_ps(x, exp_max);
		x = mm512_expf_ps(x);
		_mm512_store_ps(ptr + i, x);
	}
}

inline FORCE_INLINE void wae5(const float *softmax, const float *elliott, unsigned n, float mstd[4])
{
	__m512 vsum = _mm512_setzero_ps();
	__m512 wsum = _mm512_setzero_ps();

	for (unsigned i = 0; i < n; i += 16) {
		__m512 s = _mm512_load_ps(softmax + i);
		__m512 e = _mm512_load_ps(elliott + i);
		__m512 ee = mm512_elliott_ps(e);

		vsum = _mm512_fmadd_ps(s, ee, vsum);
		wsum = _mm512_add_ps(wsum, s);
	}

	float vsum_reduced = mm512_horizontal_sum_ps(vsum);
	float wsum_rediced = mm512_horizontal_sum_ps(wsum);

	if (wsum_rediced > 1e-10f)
		mstd[3] += (5.0f * vsum_reduced) / wsum_rediced * mstd[1] + mstd[0];
	else
		mstd[3] += mstd[0];
}

class PredictorAVX512 final : public Predictor {
	InterleavedPredictorModel m_model;
	bool m_use_q2;
public:
	PredictorAVX512(const PredictorModel &model, bool use_q2) :
		m_model(create_interleaved_predictor_model(model)),
		m_use_q2{ use_q2 }
	{
		assert(model.first.xdim * model.first.ydim <= 48 * 6);
	}

	void process(const void *src, ptrdiff_t src_stride, void *dst, const unsigned char *prescreen, unsigned n) const override
	{
		const float *src_p = static_cast<const float *>(src);
		float *dst_p = static_cast<float *>(dst);
		ptrdiff_t src_stride_f = src_stride / sizeof(float);

		// Adjust source pointer to point to top-left of filter window.
		const float *window = src_p - static_cast<ptrdiff_t>(m_model.ydim / 2) * src_stride_f - (m_model.xdim / 2 - 1);
		unsigned filter_size = m_model.xdim * m_model.ydim;
		unsigned nns = m_model.nns;

		double inv_filter_size = 1.0 / filter_size;

		for (unsigned i = 0; i < n; ++i) {
			if (prescreen[i])
				continue;

			alignas(64) float input[48 * 6];
			alignas(64) float activation[256 * 2];
			alignas(64) float mstd[4];

			// gather_input(window + i, src_stride, input, mstd);
			gather_pixels_avx512(window + i, src_stride, m_model.xdim, m_model.ydim, input, inv_filter_size, mstd);
			float scale = mstd[2];

			for (unsigned q = 0; q < (m_use_q2 ? 2U : 1U); ++q) {
				const float *neurons = q ? m_model.neurons_q2 : m_model.neurons_q1;
				const float *bias = q ? m_model.bias_q2 : m_model.bias_q1;

				zero_memory_avx512(activation, nns * 2);
				interleaved_convolution_avx512(neurons, input, nns, filter_size, activation, scale, bias);

				softmax_exp(activation, nns);
				wae5(activation, activation + nns, nns, mstd);
			}

			dst_p[i] = mstd[3] * (m_use_q2 ? 0.5f : 1.0f);
		}
	}
};

} // namespace


void byte_to_float_avx512f(const void *src, void *dst, size_t n)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	float *dst_p = static_cast<float *>(dst);
	__m512i x;

	for (size_t i = 0; i < n - n % 16; i += 16) {
		x = _mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(src_p + i)));
		_mm512_store_ps(dst_p + i, _mm512_cvtepi32_ps(x));
	}

	x = _mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(src_p + (n - n % 16))));
	_mm512_mask_store_ps(dst_p + (n - n % 16), UINT16_MAX >> (16 - n % 16), _mm512_cvtepi32_ps(x));
}

void word_to_float_avx512f(const void *src, void *dst, size_t n)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);
	__m512i x;

	for (size_t i = 0; i < n - n % 16; i += 16) {
		x = _mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i *)(src_p + i)));
		_mm512_store_ps(dst_p + i, _mm512_cvtepi32_ps(x));
	}

	x = _mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i *)(src_p + (n - n % 16))));
	_mm512_mask_store_ps(dst_p + (n - n % 16), UINT16_MAX >> (16 - n % 16), _mm512_cvtepi32_ps(x));
}

void half_to_float_avx512f(const void *src, void *dst, size_t n)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);
	__m512 x;

	for (size_t i = 0; i < n - n % 16; i += 16) {
		x = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)(src_p + i)));
		_mm512_store_ps(dst_p + i, x);
	}

	x = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)(src_p + (n - n % 16))));
	_mm512_mask_store_ps(dst_p + (n - n % 16), UINT16_MAX >> (16 - n % 16), x);
}

void float_to_byte_avx512f(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);
	__m512i x;

	for (size_t i = 0; i < n - n % 16; i += 16) {
		x = _mm512_cvtps_epu32(_mm512_load_ps(src_p + i));
		_mm_store_si128((__m128i *)(dst_p + i), _mm512_cvtusepi32_epi8(x));
	}

	// 8-bit mask granularity requires AVX-512 BW.
	alignas(16) uint8_t tmp[16];
	x = _mm512_cvtps_epu32(_mm512_load_ps(src_p + (n - n % 16)));
	_mm_store_si128((__m128i *)tmp, _mm512_cvtusepi32_epi8(x));

	for (size_t i = n - n % 16; i < n; ++i) {
		dst_p[i] = tmp[i % 16];
	}
}

void float_to_word_avx512f(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);
	__m512i x;

	for (size_t i = 0; i < n - n % 16; i += 16) {
		x = _mm512_cvtps_epu32(_mm512_load_ps(src_p + i));
		_mm256_store_si256((__m256i *)(dst_p + i), _mm512_cvtusepi32_epi16(x));
	}

	// 16-bit mask granularity requires AVX-512 BW.
	alignas(32) uint16_t tmp[16];
	x = _mm512_cvtps_epu32(_mm512_load_ps(src_p + (n - n % 16)));
	_mm256_store_si256((__m256i *)tmp, _mm512_cvtusepi32_epi16(x));

	for (size_t i = n - n % 16; i < n; ++i) {
		dst_p[i] = tmp[i % 16];
	}
}

void float_to_half_avx512f(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);
	__m256i x;

	for (size_t i = 0; i < n - n % 16; i += 16) {
		x = _mm512_cvtps_ph(_mm512_load_ps(src_p + i), 0);
		_mm256_store_si256((__m256i *)(dst_p + i), x);
	}

	// 16-bit mask granularity requires AVX-512 BW.
	alignas(32) uint16_t tmp[16];
	x = _mm512_cvtps_ph(_mm512_load_ps(src_p + (n - n % 16)), 0);
	_mm256_store_si256((__m256i *)tmp, x);

	for (size_t i = n - n % 16; i < n; ++i) {
		dst_p[i] = tmp[i % 16];
	}
}


std::unique_ptr<Predictor> create_predictor_avx512f(const std::pair<const PredictorTraits, PredictorCoefficients> &model, bool use_q2)
{
	return std::make_unique<PredictorAVX512>(model, use_q2);
}

} // namespace znedi3

#endif // ZNEDI3_X86_AVX512
