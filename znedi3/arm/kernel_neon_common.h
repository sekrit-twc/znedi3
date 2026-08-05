#pragma once

#ifdef ZNEDI3_ARM

#ifndef ZNEDI3_ARM_KERNEL_NEON_COMMON_H_
#define ZNEDI3_ARM_KERNEL_NEON_COMMON_H_

// NEON helpers shared by the NEON, SVE, and SME-driver predictors. Inline
// only; each including TU compiles these under its own target features.
//
// This header must never be included in a unit compiled with SME target
// features (kernel_sme.cpp). With the SME feature enabled, the
// auto-vectorizer may emit streaming-compatible SVE instructions in regular
// functions, which segfault outside streaming mode.
#ifdef __ARM_FEATURE_SME
  #error "kernel_neon_common.h must not be included in a TU compiled with SME target features"
#endif

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <arm_neon.h>
#include "ccdep.h"
#include "kernel_interleaved.h"

namespace znedi3 {

inline FORCE_INLINE float32x4_t neon_rcp24_f32(float32x4_t x)
{
	float32x4_t r = vrecpeq_f32(x);
	r = vmulq_f32(r, vrecpsq_f32(x, r));
	r = vmulq_f32(r, vrecpsq_f32(x, r));
	return r;
}

inline FORCE_INLINE float32x4_t neon_expf_f32(float32x4_t x)
{
	x = vfmaq_f32(vdupq_n_f32(EXPF_ONE_SCALED), x, vdupq_n_f32(EXPF_LN2_INV_SCALED));
	int32x4_t xi = vcvtq_s32_f32(x);

	// Clear the mantissa. This represents exp2(floor(x)).
	float32x4_t i = vreinterpretq_f32_s32(vandq_s32(xi, vdupq_n_s32(0x7F800000L)));
	// Reset the exponent to zero. This represents exp2(x - floor(x)).
	float32x4_t f = vreinterpretq_f32_s32(vorrq_s32(vandq_s32(xi, vdupq_n_s32(0x007FFFFFL)), vdupq_n_s32(0x3F800000L)));

	x = vdupq_n_f32(EXP2F_X_PLUS1_REMEZ[4]);
	x = vfmaq_f32(vdupq_n_f32(EXP2F_X_PLUS1_REMEZ[3]), f, x);
	x = vfmaq_f32(vdupq_n_f32(EXP2F_X_PLUS1_REMEZ[2]), f, x);
	x = vfmaq_f32(vdupq_n_f32(EXP2F_X_PLUS1_REMEZ[1]), f, x);
	x = vfmaq_f32(vdupq_n_f32(EXP2F_X_PLUS1_REMEZ[0]), f, x);

	return vmulq_f32(i, x);
}

inline FORCE_INLINE float32x4_t neon_elliott_f32(float32x4_t x)
{
	float32x4_t den = vaddq_f32(vabsq_f32(x), vdupq_n_f32(1.0f));
	return vmulq_f32(x, neon_rcp24_f32(den));
}

inline FORCE_INLINE float32x4_t neon_softmax_exp_f32(float32x4_t x)
{
	const uint32x4_t abs_mask = vdupq_n_u32(UINT32_MAX >> 1);

	uint32x4_t xbits = vreinterpretq_u32_f32(x);
	float32x4_t xabs = vreinterpretq_f32_u32(vandq_u32(xbits, abs_mask));
	uint32x4_t xsign = vbicq_u32(xbits, abs_mask);

	x = vminq_f32(xabs, vdupq_n_f32(80.0f));
	x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), xsign));
	return neon_expf_f32(x);
}


inline FORCE_INLINE void gather_pixels_neon(const float * const *src, ptrdiff_t offset_x, ptrdiff_t xdim, ptrdiff_t ydim,
                                            float *buf, unsigned stride, unsigned pixel, double *sum_out, double *sumsq_out)
{
	float64x2_t sum = vdupq_n_f64(0.0);
	float64x2_t sumsq = vdupq_n_f64(0.0);

	for (ptrdiff_t i = 0; i < ydim; ++i) {
		const float *srcp = src[i] + offset_x;

		for (ptrdiff_t j = 0; j < xdim; j += 4) {
			float32x4_t val = vld1q_f32(srcp + j);
			float *dst = buf + (i * xdim + j) * stride + pixel;

			// Inputs are stored interleaved, such that the k-th input of all
			// gathered pixels is contiguous.
			vst1q_lane_f32(dst + 0 * stride, val, 0);
			vst1q_lane_f32(dst + 1 * stride, val, 1);
			vst1q_lane_f32(dst + 2 * stride, val, 2);
			vst1q_lane_f32(dst + 3 * stride, val, 3);

			float64x2_t lo = vcvt_f64_f32(vget_low_f32(val));
			float64x2_t hi = vcvt_high_f64_f32(val);

			sum = vaddq_f64(sum, vaddq_f64(lo, hi));
			sumsq = vfmaq_f64(sumsq, lo, lo);
			sumsq = vfmaq_f64(sumsq, hi, hi);
		}
	}

	*sum_out = vaddvq_f64(sum);
	*sumsq_out = vaddvq_f64(sumsq);
}

inline FORCE_INLINE void input_stddev_neon(const double *sums, const double *sumsqs, float *mstd, unsigned num, unsigned batch, double inv_size)
{
	for (unsigned p = 0; p < num; ++p) {
		float mean = static_cast<float>(sums[p] * inv_size);
		double tmp = sumsqs[p] * inv_size - static_cast<double>(mean) * mean;

		mstd[0 * batch + p] = mean;
		mstd[3 * batch + p] = 0.0f;

		if (tmp < FLT_EPSILON) {
			mstd[1 * batch + p] = 0.0f;
			mstd[2 * batch + p] = 0.0f;
		} else {
			float stddev = static_cast<float>(std::sqrt(tmp));
			mstd[1 * batch + p] = stddev;
			mstd[2 * batch + p] = 1.0f / stddev;
		}
	}
}

} // namespace znedi3

#endif // ZNEDI3_ARM_KERNEL_NEON_COMMON_H_

#endif // ZNEDI3_ARM