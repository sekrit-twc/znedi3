#ifdef ZNEDI3_X86

#include <immintrin.h>
#include "kernel.h"
#include "kernel_x86.h"

namespace znedi3 {

void half_to_float_f16c(const void *src, void *dst, size_t n)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		__m256 x = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(src_p + i)));
		_mm256_store_ps(dst_p + i, x);
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		dst_p[i] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(src_p[i])));
	}
}

void float_to_half_f16c(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		__m128i x = _mm256_cvtps_ph(_mm256_load_ps(src_p + i), 0);
		_mm_store_si128((__m128i *)(dst_p + i), x);
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		dst_p[i] = _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(src_p[i]), 0));
	}
}

} // namespace znedi3

#endif // ZNEDI3_X86