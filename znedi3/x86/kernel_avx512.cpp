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

inline FORCE_INLINE __m128 mm512_horizontal_sum2_ps(__m512 x, __m512 y)
{
	__m256 stage1x = _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
	__m256 stage1y = _mm256_add_ps(_mm512_castps512_ps256(y), _mm512_extractf32x8_ps(y, 1));
	__m256 stage2 = _mm256_hadd_ps(stage1x, stage1y);
	__m256 stage3 = _mm256_hadd_ps(stage2, stage2);
	__m128 stage4 = _mm_add_ps(_mm256_castps256_ps128(stage3), _mm256_extractf128_ps(stage3, 1));
	return stage4;
}

inline FORCE_INLINE __m128d mm512_horizontal_sum2_pd(__m512d x, __m512d y)
{
	__m256d stage1x = _mm256_add_pd(_mm512_castpd512_pd256(x), _mm512_extractf64x4_pd(x, 1));
	__m256d stage1y = _mm256_add_pd(_mm512_castpd512_pd256(y), _mm512_extractf64x4_pd(y, 1));
	__m256d stage2 = _mm256_hadd_pd(stage1x, stage1y);
	__m128d stage3 = _mm_add_pd(_mm256_castpd256_pd128(stage2), _mm256_extractf128_pd(stage2, 1));
	return stage3;
}

inline FORCE_INLINE __m128 mm_rsqrt24_ss(__m128 x)
{
	__m128 tmp0 = _mm_rsqrt_ss(x);
	__m128 tmp1 = _mm_mul_ss(x, tmp0);
	__m128 tmp2 = _mm_mul_ss(_mm_set_ss(0.5f), tmp0);
	__m128 tmp3 = _mm_fnmadd_ss(tmp1, tmp0, _mm_set_ss(3.0f));
	return _mm_mul_ss(tmp2, tmp3);
}

inline FORCE_INLINE __m512 mm512_expf_ps(__m512 x)
{
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
			__m256 val = _mm256_loadu_ps(src + j);

			__m512d vald = _mm512_cvtps_pd(val);
			sum = _mm512_add_pd(sum, vald);
			sumsq = _mm512_fmadd_pd(vald, vald, sumsq);

			_mm256_store_ps(buf + j, val);
		}
		src += src_stride_f;
		buf += xdim;
	}

	// Get horizontal sums.
	__m128d hsum = mm512_horizontal_sum2_pd(sum, sumsq);
	hsum = _mm_mul_pd(hsum, _mm_set1_pd(inv_size));

	__m128d sum_reduced = hsum;
	__m128d sumsq_reduced = _mm_permute_pd(hsum, 1);

	__m128d variance = _mm_fnmadd_pd(sum_reduced, sum_reduced, sumsq_reduced);
	__m128d epsilon_mask = _mm_cmp_sd(variance, _mm_set_sd(FLT_EPSILON), _CMP_GE_OQ);

	__m128 variance_f32 = _mm_cvtsd_ss(_mm_undefined_ps(), variance);
	__m128 stddev_inv = mm_rsqrt24_ss(variance_f32);
	__m128 stddev = _mm_mul_ss(stddev_inv, variance_f32);

	stddev_inv = _mm_and_ps(_mm_castpd_ps(epsilon_mask), stddev_inv);
	stddev = _mm_and_ps(_mm_castpd_ps(epsilon_mask), stddev);

	mstd[0] = static_cast<float>(_mm_cvtsd_f64(sum_reduced));
	mstd[1] = _mm_cvtss_f32(stddev);
	mstd[2] = _mm_cvtss_f32(stddev_inv);
	mstd[3] = 0.0f;
}

template <unsigned NNS>
inline FORCE_INLINE void interleaved_convolution_avx512(const float *kernels, const float *input, unsigned nns, unsigned n, float *output, float scale, const float *bias)
{
	nns = NNS ? NNS : nns;

	const float *kptr0_base = kernels;
	const float *kptr1_base = kernels + 16;
	ptrdiff_t kstride = static_cast<ptrdiff_t>(nns) * 2;

	for (unsigned nn = 0; nn < nns * 2; nn += 32) {
		_mm512_store_ps(output + nn + 0, _mm512_setzero_ps());
		_mm512_store_ps(output + nn + 16, _mm512_setzero_ps());
	}
	for (ptrdiff_t k = 0; k < n; k += 8) {
		__m512 x0 = _mm512_set1_ps(input[k + 0]);
		__m512 x1 = _mm512_set1_ps(input[k + 1]);
		__m512 x2 = _mm512_set1_ps(input[k + 2]);
		__m512 x3 = _mm512_set1_ps(input[k + 3]);
		__m512 x4 = _mm512_set1_ps(input[k + 4]);
		__m512 x5 = _mm512_set1_ps(input[k + 5]);
		__m512 x6 = _mm512_set1_ps(input[k + 6]);
		__m512 x7 = _mm512_set1_ps(input[k + 7]);

		const float *kptr0 = kptr0_base;
		const float *kptr1 = kptr1_base;

		for (ptrdiff_t nn = 0; nn < nns * 2; nn += 32) {
			__m512 n00_15_a = _mm512_load_ps(output + nn);
			__m512 n16_31_a = _mm512_load_ps(output + nn + 16);
			__m512 n00_15_b = _mm512_setzero_ps();
			__m512 n16_31_b = _mm512_setzero_ps();
			__m512 c;

			c = _mm512_load_ps(kptr0 + 0 * kstride);
			n00_15_a = _mm512_fmadd_ps(c, x0, n00_15_a);

			c = _mm512_load_ps(kptr1 + 0 * kstride);
			n16_31_a = _mm512_fmadd_ps(c, x0, n16_31_a);

			c = _mm512_load_ps(kptr0 + 1 * kstride);
			n00_15_b = _mm512_fmadd_ps(c, x1, n00_15_b);

			c = _mm512_load_ps(kptr1 + 1 * kstride);
			n16_31_b = _mm512_fmadd_ps(c, x1, n16_31_b);

			c = _mm512_load_ps(kptr0 + 2 * kstride);
			n00_15_a = _mm512_fmadd_ps(c, x2, n00_15_a);

			c = _mm512_load_ps(kptr1 + 2 * kstride);
			n16_31_a = _mm512_fmadd_ps(c, x2, n16_31_a);

			c = _mm512_load_ps(kptr0 + 3 * kstride);
			n00_15_b = _mm512_fmadd_ps(c, x3, n00_15_b);

			c = _mm512_load_ps(kptr1 + 3 * kstride);
			n16_31_b = _mm512_fmadd_ps(c, x3, n16_31_b);

			c = _mm512_load_ps(kptr0 + 4 * kstride);
			n00_15_a = _mm512_fmadd_ps(c, x4, n00_15_a);

			c = _mm512_load_ps(kptr1 + 4 * kstride);
			n16_31_a = _mm512_fmadd_ps(c, x4, n16_31_a);

			c = _mm512_load_ps(kptr0 + 5 * kstride);
			n00_15_b = _mm512_fmadd_ps(c, x5, n00_15_b);

			c = _mm512_load_ps(kptr1 + 5 * kstride);
			n16_31_b = _mm512_fmadd_ps(c, x5, n16_31_b);

			c = _mm512_load_ps(kptr0 + 6 * kstride);
			n00_15_a = _mm512_fmadd_ps(c, x6, n00_15_a);

			c = _mm512_load_ps(kptr1 + 6 * kstride);
			n16_31_a = _mm512_fmadd_ps(c, x6, n16_31_a);

			c = _mm512_load_ps(kptr0 + 7 * kstride);
			n00_15_b = _mm512_fmadd_ps(c, x7, n00_15_b);

			c = _mm512_load_ps(kptr1 + 7 * kstride);
			n16_31_b = _mm512_fmadd_ps(c, x7, n16_31_b);

			n00_15_a = _mm512_add_ps(n00_15_a, n00_15_b);
			n16_31_a = _mm512_add_ps(n16_31_a, n16_31_b);

			_mm512_store_ps(output + nn, n00_15_a);
			_mm512_store_ps(output + nn + 16, n16_31_a);

			kptr0 += 32;
			kptr1 += 32;
		}
		kptr0_base += 8 * kstride;
		kptr1_base += 8 * kstride;
	}
	for (ptrdiff_t nn = 0; nn < nns * 2; nn += 16) {
		__m512 accum = _mm512_load_ps(output + nn);
		accum = _mm512_fmadd_ps(_mm512_set1_ps(scale), accum, _mm512_load_ps(bias + nn));
		_mm512_store_ps(output + nn, accum);
	}
}

template <>
inline FORCE_INLINE void interleaved_convolution_avx512<16>(const float *kernels, const float *input, unsigned, unsigned n, float *output, float scale, const float *bias)
{
	__m512 accum0a = _mm512_setzero_ps();
	__m512 accum1a = _mm512_setzero_ps();
	__m512 accum0b = _mm512_setzero_ps();
	__m512 accum1b = _mm512_setzero_ps();

	for (ptrdiff_t k = 0; k < static_cast<ptrdiff_t>(n); k += 8) {
		__m512 x0 = _mm512_set1_ps(input[k + 0]);
		__m512 x1 = _mm512_set1_ps(input[k + 1]);
		__m512 x2 = _mm512_set1_ps(input[k + 2]);
		__m512 x3 = _mm512_set1_ps(input[k + 3]);
		__m512 x4 = _mm512_set1_ps(input[k + 4]);
		__m512 x5 = _mm512_set1_ps(input[k + 5]);
		__m512 x6 = _mm512_set1_ps(input[k + 6]);
		__m512 x7 = _mm512_set1_ps(input[k + 7]);

		accum0a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 0), x0, accum0a);
		accum1a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 16), x0, accum1a);

		accum0b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 32), x1, accum0b);
		accum1b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 48), x1, accum1b);

		accum0a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 64), x2, accum0a);
		accum1a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 80), x2, accum1a);

		accum0b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 96), x3, accum0b);
		accum1b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 112), x3, accum1b);

		accum0a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 128), x4, accum0a);
		accum1a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 144), x4, accum1a);

		accum0b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 160), x5, accum0b);
		accum1b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 176), x5, accum1b);

		accum0a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 192), x6, accum0a);
		accum1a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 208), x6, accum1a);

		accum0b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 224), x7, accum0b);
		accum1b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 240), x7, accum1b);

		kernels += 256;
	}

	accum0a = _mm512_add_ps(accum0a, accum0b);
	accum1a = _mm512_add_ps(accum1a, accum1b);

	__m512 scale_ps = _mm512_set1_ps(scale);
	accum0a = _mm512_fmadd_ps(scale_ps, accum0a, _mm512_load_ps(bias + 0));
	accum1a = _mm512_fmadd_ps(scale_ps, accum1a, _mm512_load_ps(bias + 16));

	_mm512_store_ps(output + 0, accum0a);
	_mm512_store_ps(output + 16, accum1a);
}

template <>
inline FORCE_INLINE void interleaved_convolution_avx512<32>(const float *kernels, const float *input, unsigned, unsigned n, float *output, float scale, const float *bias)
{
	__m512 accum0a = _mm512_setzero_ps();
	__m512 accum1a = _mm512_setzero_ps();
	__m512 accum2a = _mm512_setzero_ps();
	__m512 accum3a = _mm512_setzero_ps();

	__m512 accum0b = _mm512_setzero_ps();
	__m512 accum1b = _mm512_setzero_ps();
	__m512 accum2b = _mm512_setzero_ps();
	__m512 accum3b = _mm512_setzero_ps();

	for (ptrdiff_t k = 0; k < static_cast<ptrdiff_t>(n); k += 8) {
		__m512 x0 = _mm512_set1_ps(input[k + 0]);
		__m512 x1 = _mm512_set1_ps(input[k + 1]);
		__m512 x2 = _mm512_set1_ps(input[k + 2]);
		__m512 x3 = _mm512_set1_ps(input[k + 3]);
		__m512 x4 = _mm512_set1_ps(input[k + 4]);
		__m512 x5 = _mm512_set1_ps(input[k + 5]);
		__m512 x6 = _mm512_set1_ps(input[k + 6]);
		__m512 x7 = _mm512_set1_ps(input[k + 7]);

		accum0a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 0), x0, accum0a);
		accum1a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 16), x0, accum1a);
		accum2a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 32), x0, accum2a);
		accum3a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 48), x0, accum3a);

		accum0b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 64), x1, accum0b);
		accum1b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 80), x1, accum1b);
		accum2b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 96), x1, accum2b);
		accum3b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 112), x1, accum3b);

		accum0a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 128), x2, accum0a);
		accum1a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 144), x2, accum1a);
		accum2a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 160), x2, accum2a);
		accum3a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 176), x2, accum3a);

		accum0b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 192), x3, accum0b);
		accum1b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 208), x3, accum1b);
		accum2b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 224), x3, accum2b);
		accum3b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 240), x3, accum3b);

		accum0a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 256), x4, accum0a);
		accum1a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 272), x4, accum1a);
		accum2a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 288), x4, accum2a);
		accum3a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 304), x4, accum3a);

		accum0b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 320), x5, accum0b);
		accum1b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 336), x5, accum1b);
		accum2b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 352), x5, accum2b);
		accum3b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 368), x5, accum3b);

		accum0a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 384), x6, accum0a);
		accum1a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 400), x6, accum1a);
		accum2a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 416), x6, accum2a);
		accum3a = _mm512_fmadd_ps(_mm512_load_ps(kernels + 432), x6, accum3a);

		accum0b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 448), x7, accum0b);
		accum1b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 464), x7, accum1b);
		accum2b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 480), x7, accum2b);
		accum3b = _mm512_fmadd_ps(_mm512_load_ps(kernels + 496), x7, accum3b);

		kernels += 512;
	}

	accum0a = _mm512_add_ps(accum0a, accum0b);
	accum1a = _mm512_add_ps(accum1a, accum1b);
	accum2a = _mm512_add_ps(accum2a, accum2b);
	accum3a = _mm512_add_ps(accum3a, accum3b);

	__m512 scale_ps = _mm512_set1_ps(scale);
	accum0a = _mm512_fmadd_ps(scale_ps, accum0a, _mm512_load_ps(bias + 0));
	accum1a = _mm512_fmadd_ps(scale_ps, accum1a, _mm512_load_ps(bias + 16));
	accum2a = _mm512_fmadd_ps(scale_ps, accum2a, _mm512_load_ps(bias + 32));
	accum3a = _mm512_fmadd_ps(scale_ps, accum3a, _mm512_load_ps(bias + 48));

	_mm512_store_ps(output + 0, accum0a);
	_mm512_store_ps(output + 16, accum1a);
	_mm512_store_ps(output + 32, accum2a);
	_mm512_store_ps(output + 48, accum3a);
}

template <unsigned NNS>
inline FORCE_INLINE void softmax_exp(float *ptr, unsigned n)
{
	n = NNS ? NNS : n;

	const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(UINT32_MAX >> 1));
	const __m512 exp_max = _mm512_set1_ps(80.0f);

	for (unsigned i = 0; i < n; i += 16) {
		__m512 x = _mm512_load_ps(ptr + i);
		__m512 xabs = _mm512_and_ps(abs_mask, x);
		__m512 xsign = _mm512_andnot_ps(abs_mask, x);
		x = _mm512_min_ps(xabs, exp_max);
		x = _mm512_or_ps(xsign, x);
		x = mm512_expf_ps(x);
		_mm512_store_ps(ptr + i, x);
	}
}

template <unsigned NNS>
inline FORCE_INLINE void wae5(const float *softmax, const float *elliott, unsigned n, float mstd[4])
{
	n = NNS ? NNS : n;

	__m512 vsum = _mm512_setzero_ps();
	__m512 wsum = _mm512_setzero_ps();

	for (unsigned i = 0; i < n; i += 16) {
		__m512 s = _mm512_load_ps(softmax + i);
		__m512 e = _mm512_load_ps(elliott + i);
		__m512 ee = mm512_elliott_ps(e);

		vsum = _mm512_fmadd_ps(s, ee, vsum);
		wsum = _mm512_add_ps(wsum, s);
	}

	__m128 packed = mm512_horizontal_sum2_ps(vsum, wsum);

	__m128 vsum_reduced = packed;
	__m128 wsum_reduced = _mm_permute_ps(packed, _MM_SHUFFLE(2, 3, 0, 1));
	__m128 mask = _mm_cmp_ss(wsum_reduced, _mm_set_ss(1e-10f), _CMP_GT_OQ);

	vsum_reduced = _mm_mul_ss(vsum_reduced, _mm_set_ss(5.0f));
	vsum_reduced = _mm_div_ss(vsum_reduced, wsum_reduced);

	vsum_reduced = _mm_fmadd_ss(_mm_set_ss(mstd[1]), vsum_reduced, _mm_set_ss(mstd[0]));
	vsum_reduced = _mm_blendv_ps(_mm_set_ss(mstd[0]), vsum_reduced, mask);

	mstd[3] += _mm_cvtss_f32(vsum_reduced);
}

template <unsigned NNS>
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

				interleaved_convolution_avx512<NNS>(neurons, input, nns, filter_size, activation, scale, bias);

				softmax_exp<NNS>(activation, nns);
				wae5<NNS>(activation, activation + nns, nns, mstd);
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
	switch (model.first.nns) {
	case 16:
		return std::make_unique<PredictorAVX512<16>>(model, use_q2);
	case 32:
		return std::make_unique<PredictorAVX512<32>>(model, use_q2);
	default:
		return std::make_unique<PredictorAVX512<0>>(model, use_q2);
	}
}

} // namespace znedi3

#endif // ZNEDI3_X86_AVX512
