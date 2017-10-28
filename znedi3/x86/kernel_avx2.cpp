#ifdef ZNEDI3_X86

#include <algorithm>
#include <immintrin.h>
#include "kernel.h"
#include "kernel_x86.h"

#define USE_FMA 1
#include "kernel_avx_common.h"

namespace znedi3 {

void byte_to_float_avx2(const void *src, void *dst, size_t n)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		__m256i x = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src_p + i)));
		_mm256_store_ps(dst_p + i, _mm256_cvtepi32_ps(x));
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		dst_p[i] = src_p[i];
	}
}

void word_to_float_avx2(const void *src, void *dst, size_t n)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		__m256i x = _mm256_cvtepu16_epi32(_mm_load_si128((const __m128i *)(src_p + i)));
		_mm256_store_ps(dst_p + i, _mm256_cvtepi32_ps(x));
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		dst_p[i] = src_p[i];
	}
}
void float_to_byte_avx2(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	for (size_t i = 0; i < n - n % 16; i += 16) {
		__m256i lo = _mm256_cvtps_epi32(_mm256_load_ps(src_p + i + 0));
		__m256i hi = _mm256_cvtps_epi32(_mm256_load_ps(src_p + i + 8));
		__m256i x = _mm256_packus_epi32(lo, hi);
		x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3, 1, 2, 0));
		x = _mm256_packus_epi16(x, x);
		x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3, 1, 2, 0));
		_mm_store_si128((__m128i *)(dst_p + i), _mm256_castsi256_si128(x));
	}
	for (size_t i = n - n % 16; i < n; ++i) {
		int32_t x = _mm_cvtss_si32(_mm_set_ss(src_p[i]));
		x = std::min(std::max(x, static_cast<int32_t>(0)), static_cast<int32_t>(UINT8_MAX));
		dst_p[i] = static_cast<uint8_t>(x);
	}
}

void float_to_word_avx2(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	for (size_t i = 0; i < n - n % 16; i += 16) {
		__m256i lo = _mm256_cvtps_epi32(_mm256_load_ps(src_p + i + 0));
		__m256i hi = _mm256_cvtps_epi32(_mm256_load_ps(src_p + i + 8));
		__m256i x = _mm256_packus_epi32(lo, hi);
		x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3, 1, 2, 0));
		_mm256_store_si256((__m256i *)(dst_p + i), x);
	}
	for (size_t i = n - n % 16; i < n; ++i) {
		int32_t x = _mm_cvtss_si32(_mm_set_ss(src_p[i]));
		x = std::min(std::max(x, static_cast<int32_t>(0)), static_cast<int32_t>(UINT16_MAX));
		dst_p[i] = static_cast<uint16_t>(x);
	}
}

void cubic_interpolation_avx2(const void *src, ptrdiff_t src_stride, void *dst, const unsigned char *prescreen, unsigned n)
{
	cubic_interpolation_avx_impl(src, src_stride, dst, prescreen, n);
}


std::unique_ptr<Prescreener> create_prescreener_old_avx2(const PrescreenerOldCoefficients &coeffs, double pixel_half)
{
	return std::make_unique<PrescreenerOldAVX>(coeffs, pixel_half);
}

std::unique_ptr<Prescreener> create_prescreener_new_avx2(const PrescreenerNewCoefficients &coeffs, double pixel_half)
{
	return std::make_unique<PrescreenerNewAVX>(coeffs, pixel_half);
}

std::unique_ptr<Predictor> create_predictor_avx2(const PredictorModel &model, bool use_q2)
{
	return std::make_unique<PredictorAVX>(model, use_q2);
}

} // namespace znedi3

#endif // ZNEDI3_X86
