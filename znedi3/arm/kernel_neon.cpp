#ifdef ZNEDI3_ARM

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <arm_neon.h>
#include "alloc.h"
#include "ccdep.h"
#include "kernel.h"
#include "kernel_arm.h"
#include "kernel_neon_common.h"

namespace znedi3 {
namespace {

class PrescreenerOldNEON final : public Prescreener {
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
	PrescreenerOldNEON(const PrescreenerOldCoefficients &data, double half) :
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

	// Layers 1-2 for one pixel, given the summed layer-0 accumulator chain
	// (before bias). Identical operation sequence to the original scalar-loop
	// body, so results are bit-identical regardless of pixel batching.
	static FORCE_INLINE unsigned char layers12(const InterleavedPrescreenerOldCoefficients &data, float32x4_t accum0)
	{
		// Neurons 1-3 pass through the elliott function, neuron 0 stays linear.
		alignas(16) static constexpr uint32_t l0_mask[4] = { 0, UINT32_MAX, UINT32_MAX, UINT32_MAX };
		float32x4_t l0 = vaddq_f32(accum0, vld1q_f32(data.bias_l0));
		l0 = vbslq_f32(vld1q_u32(l0_mask), neon_elliott_f32(l0), l0);

		// Layer 1.
		float32x4_t accum1;
		float32x4_t accum2;
		float32x4_t accum3;

		accum0 = vmulq_laneq_f32(vld1q_f32(data.kernel_l1[0]), l0, 0);
		accum1 = vmulq_laneq_f32(vld1q_f32(data.kernel_l1[1]), l0, 1);
		accum2 = vmulq_laneq_f32(vld1q_f32(data.kernel_l1[2]), l0, 2);
		accum3 = vmulq_laneq_f32(vld1q_f32(data.kernel_l1[3]), l0, 3);

		accum0 = vaddq_f32(accum0, accum1);
		accum2 = vaddq_f32(accum2, accum3);
		accum0 = vaddq_f32(accum0, accum2);

		float32x4_t l1 = neon_elliott_f32(vaddq_f32(accum0, vld1q_f32(data.bias_l1)));

		// Layer 2.
		accum0 = vmulq_laneq_f32(vld1q_f32(data.kernel_l2[0]), l0, 0);
		accum1 = vmulq_laneq_f32(vld1q_f32(data.kernel_l2[1]), l0, 1);
		accum2 = vmulq_laneq_f32(vld1q_f32(data.kernel_l2[2]), l0, 2);
		accum3 = vmulq_laneq_f32(vld1q_f32(data.kernel_l2[3]), l0, 3);

		accum0 = vfmaq_laneq_f32(accum0, vld1q_f32(data.kernel_l2[4]), l1, 0);
		accum1 = vfmaq_laneq_f32(accum1, vld1q_f32(data.kernel_l2[5]), l1, 1);
		accum2 = vfmaq_laneq_f32(accum2, vld1q_f32(data.kernel_l2[6]), l1, 2);
		accum3 = vfmaq_laneq_f32(accum3, vld1q_f32(data.kernel_l2[7]), l1, 3);

		accum0 = vaddq_f32(accum0, accum1);
		accum2 = vaddq_f32(accum2, accum3);
		accum0 = vaddq_f32(accum0, accum2);

		float32x4_t l2 = vaddq_f32(accum0, vld1q_f32(data.bias_l2));

		float lhs = std::max(vgetq_lane_f32(l2, 2), vgetq_lane_f32(l2, 3));
		float rhs = std::max(vgetq_lane_f32(l2, 0), vgetq_lane_f32(l2, 1));
		return lhs <= rhs ? UCHAR_MAX : 0;
	}
public:
	size_t get_tmp_size() const noexcept override { return 0; }

	void process(const float * const src[4], unsigned char *prescreen, void *, unsigned n) const noexcept override
	{
		const InterleavedPrescreenerOldCoefficients &data = m_data.front();
		ptrdiff_t window_offset = 5;
		ptrdiff_t j = 0;

		// Main loop: 4 pixels per iteration. Each layer-0 coefficient vector
		// is loaded once per group of 4 pixels instead of once per pixel.
		// Every pixel keeps its own 4-accumulator chain in the original
		// (ki, kj) tap order, so the output is bit-identical to the
		// single-pixel loop below. Accumulators are individually named --
		// gcc keeps an accumulator array in memory (load/store per tap
		// group), which forfeits the whole win.
		for (; j + 4 <= static_cast<ptrdiff_t>(n); j += 4) {
			float32x4_t a00 = vdupq_n_f32(0.0f), a01 = vdupq_n_f32(0.0f), a02 = vdupq_n_f32(0.0f), a03 = vdupq_n_f32(0.0f);
			float32x4_t a10 = vdupq_n_f32(0.0f), a11 = vdupq_n_f32(0.0f), a12 = vdupq_n_f32(0.0f), a13 = vdupq_n_f32(0.0f);
			float32x4_t a20 = vdupq_n_f32(0.0f), a21 = vdupq_n_f32(0.0f), a22 = vdupq_n_f32(0.0f), a23 = vdupq_n_f32(0.0f);
			float32x4_t a30 = vdupq_n_f32(0.0f), a31 = vdupq_n_f32(0.0f), a32 = vdupq_n_f32(0.0f), a33 = vdupq_n_f32(0.0f);

			// Layer 0.
			for (ptrdiff_t ki = 0; ki < 4; ++ki) {
				const float *srcp = src[ki];

				for (ptrdiff_t kj = 0; kj < 12; kj += 4) {
					float32x4_t c0 = vld1q_f32(data.kernel_l0[ki * 12 + kj + 0]);
					float32x4_t c1 = vld1q_f32(data.kernel_l0[ki * 12 + kj + 1]);
					float32x4_t c2 = vld1q_f32(data.kernel_l0[ki * 12 + kj + 2]);
					float32x4_t c3 = vld1q_f32(data.kernel_l0[ki * 12 + kj + 3]);

					float32x4_t x0 = vld1q_f32(srcp - window_offset + j + 0 + kj);
					a00 = vfmaq_laneq_f32(a00, c0, x0, 0);
					a01 = vfmaq_laneq_f32(a01, c1, x0, 1);
					a02 = vfmaq_laneq_f32(a02, c2, x0, 2);
					a03 = vfmaq_laneq_f32(a03, c3, x0, 3);

					float32x4_t x1 = vld1q_f32(srcp - window_offset + j + 1 + kj);
					a10 = vfmaq_laneq_f32(a10, c0, x1, 0);
					a11 = vfmaq_laneq_f32(a11, c1, x1, 1);
					a12 = vfmaq_laneq_f32(a12, c2, x1, 2);
					a13 = vfmaq_laneq_f32(a13, c3, x1, 3);

					float32x4_t x2 = vld1q_f32(srcp - window_offset + j + 2 + kj);
					a20 = vfmaq_laneq_f32(a20, c0, x2, 0);
					a21 = vfmaq_laneq_f32(a21, c1, x2, 1);
					a22 = vfmaq_laneq_f32(a22, c2, x2, 2);
					a23 = vfmaq_laneq_f32(a23, c3, x2, 3);

					float32x4_t x3 = vld1q_f32(srcp - window_offset + j + 3 + kj);
					a30 = vfmaq_laneq_f32(a30, c0, x3, 0);
					a31 = vfmaq_laneq_f32(a31, c1, x3, 1);
					a32 = vfmaq_laneq_f32(a32, c2, x3, 2);
					a33 = vfmaq_laneq_f32(a33, c3, x3, 3);
				}
			}

			prescreen[j + 0] = layers12(data, vaddq_f32(vaddq_f32(a00, a01), vaddq_f32(a02, a03)));
			prescreen[j + 1] = layers12(data, vaddq_f32(vaddq_f32(a10, a11), vaddq_f32(a12, a13)));
			prescreen[j + 2] = layers12(data, vaddq_f32(vaddq_f32(a20, a21), vaddq_f32(a22, a23)));
			prescreen[j + 3] = layers12(data, vaddq_f32(vaddq_f32(a30, a31), vaddq_f32(a32, a33)));
		}

		// Tail: original single-pixel loop.
		for (; j < static_cast<ptrdiff_t>(n); ++j) {
			float32x4_t accum0 = vdupq_n_f32(0.0f);
			float32x4_t accum1 = vdupq_n_f32(0.0f);
			float32x4_t accum2 = vdupq_n_f32(0.0f);
			float32x4_t accum3 = vdupq_n_f32(0.0f);

			// Layer 0.
			for (ptrdiff_t ki = 0; ki < 4; ++ki) {
				const float *srcp = src[ki];

				for (ptrdiff_t kj = 0; kj < 12; kj += 4) {
					float32x4_t x = vld1q_f32(srcp - window_offset + j + kj);

					accum0 = vfmaq_laneq_f32(accum0, vld1q_f32(data.kernel_l0[ki * 12 + kj + 0]), x, 0);
					accum1 = vfmaq_laneq_f32(accum1, vld1q_f32(data.kernel_l0[ki * 12 + kj + 1]), x, 1);
					accum2 = vfmaq_laneq_f32(accum2, vld1q_f32(data.kernel_l0[ki * 12 + kj + 2]), x, 2);
					accum3 = vfmaq_laneq_f32(accum3, vld1q_f32(data.kernel_l0[ki * 12 + kj + 3]), x, 3);
				}
			}

			accum0 = vaddq_f32(accum0, accum1);
			accum2 = vaddq_f32(accum2, accum3);
			accum0 = vaddq_f32(accum0, accum2);

			prescreen[j] = layers12(data, accum0);
		}
	}
};


class PrescreenerNewNEON final : public Prescreener {
	struct InterleavedPrescreenerNewCoefficients {
		float kernel_l0[64][4];
		float bias_l0[4];

		float kernel_l1[4][4];
		float bias_l1[4];
	};

	AlignedVector<InterleavedPrescreenerNewCoefficients> m_data;
public:
	PrescreenerNewNEON(const PrescreenerNewCoefficients &data, double half) :
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

	// Layer 1 + decision store for one group of 4 pixels, given the summed
	// layer-0 accumulator chain (before bias). Identical operation sequence
	// to the original loop body, so results are bit-identical regardless of
	// group batching.
	static FORCE_INLINE void layer1_store(const InterleavedPrescreenerNewCoefficients &data, float32x4_t accum0, unsigned char *prescreen)
	{
		float32x4_t l0 = neon_elliott_f32(vaddq_f32(accum0, vld1q_f32(data.bias_l0)));

		// Layer 1.
		float32x4_t accum1;
		float32x4_t accum2;
		float32x4_t accum3;

		accum0 = vmulq_laneq_f32(vld1q_f32(data.kernel_l1[0]), l0, 0);
		accum1 = vmulq_laneq_f32(vld1q_f32(data.kernel_l1[1]), l0, 1);
		accum2 = vmulq_laneq_f32(vld1q_f32(data.kernel_l1[2]), l0, 2);
		accum3 = vmulq_laneq_f32(vld1q_f32(data.kernel_l1[3]), l0, 3);

		accum0 = vaddq_f32(accum0, accum1);
		accum2 = vaddq_f32(accum2, accum3);
		accum0 = vaddq_f32(accum0, accum2);

		float32x4_t l1 = neon_elliott_f32(vaddq_f32(accum0, vld1q_f32(data.bias_l1)));

		uint32x4_t result = vcgtq_f32(l1, vdupq_n_f32(0.0f));
		uint16x4_t result_w = vmovn_u32(result);
		uint8x8_t result_b = vmovn_u16(vcombine_u16(result_w, result_w));
		vst1_lane_u32(reinterpret_cast<uint32_t *>(prescreen), vreinterpret_u32_u8(result_b), 0);
	}
public:
	size_t get_tmp_size() const noexcept override { return 0; }

	void process(const float * const src[4], unsigned char *prescreen, void *, unsigned n) const noexcept override
	{
		const InterleavedPrescreenerNewCoefficients &data = m_data.front();
		ptrdiff_t window_offset = 6;
		ptrdiff_t j = 0;

		// Main loop: 4 groups x 4 pixels per iteration. Each layer-0
		// coefficient vector is loaded once per 16 pixels instead of once
		// per 4. Every group keeps its own 4-accumulator chain in the
		// original (ki, kj) tap order, so the output is bit-identical to
		// the single-group loop below. Accumulators are individually named
		// -- gcc keeps an accumulator array in memory (load/store per tap
		// group), which forfeits the whole win.
		for (; j + 16 <= static_cast<ptrdiff_t>(n); j += 16) {
			float32x4_t a00 = vdupq_n_f32(0.0f), a01 = vdupq_n_f32(0.0f), a02 = vdupq_n_f32(0.0f), a03 = vdupq_n_f32(0.0f);
			float32x4_t a10 = vdupq_n_f32(0.0f), a11 = vdupq_n_f32(0.0f), a12 = vdupq_n_f32(0.0f), a13 = vdupq_n_f32(0.0f);
			float32x4_t a20 = vdupq_n_f32(0.0f), a21 = vdupq_n_f32(0.0f), a22 = vdupq_n_f32(0.0f), a23 = vdupq_n_f32(0.0f);
			float32x4_t a30 = vdupq_n_f32(0.0f), a31 = vdupq_n_f32(0.0f), a32 = vdupq_n_f32(0.0f), a33 = vdupq_n_f32(0.0f);

			// Layer 0.
			for (ptrdiff_t ki = 0; ki < 4; ++ki) {
				const float *srcp = src[ki];

				for (ptrdiff_t kj = 0; kj < 16; kj += 4) {
					float32x4_t c0 = vld1q_f32(data.kernel_l0[ki * 16 + kj + 0]);
					float32x4_t c1 = vld1q_f32(data.kernel_l0[ki * 16 + kj + 1]);
					float32x4_t c2 = vld1q_f32(data.kernel_l0[ki * 16 + kj + 2]);
					float32x4_t c3 = vld1q_f32(data.kernel_l0[ki * 16 + kj + 3]);

					float32x4_t x0 = vld1q_f32(srcp - window_offset + j + 0 + kj);
					a00 = vfmaq_laneq_f32(a00, c0, x0, 0);
					a01 = vfmaq_laneq_f32(a01, c1, x0, 1);
					a02 = vfmaq_laneq_f32(a02, c2, x0, 2);
					a03 = vfmaq_laneq_f32(a03, c3, x0, 3);

					float32x4_t x1 = vld1q_f32(srcp - window_offset + j + 4 + kj);
					a10 = vfmaq_laneq_f32(a10, c0, x1, 0);
					a11 = vfmaq_laneq_f32(a11, c1, x1, 1);
					a12 = vfmaq_laneq_f32(a12, c2, x1, 2);
					a13 = vfmaq_laneq_f32(a13, c3, x1, 3);

					float32x4_t x2 = vld1q_f32(srcp - window_offset + j + 8 + kj);
					a20 = vfmaq_laneq_f32(a20, c0, x2, 0);
					a21 = vfmaq_laneq_f32(a21, c1, x2, 1);
					a22 = vfmaq_laneq_f32(a22, c2, x2, 2);
					a23 = vfmaq_laneq_f32(a23, c3, x2, 3);

					float32x4_t x3 = vld1q_f32(srcp - window_offset + j + 12 + kj);
					a30 = vfmaq_laneq_f32(a30, c0, x3, 0);
					a31 = vfmaq_laneq_f32(a31, c1, x3, 1);
					a32 = vfmaq_laneq_f32(a32, c2, x3, 2);
					a33 = vfmaq_laneq_f32(a33, c3, x3, 3);
				}
			}

			layer1_store(data, vaddq_f32(vaddq_f32(a00, a01), vaddq_f32(a02, a03)), prescreen + j + 0);
			layer1_store(data, vaddq_f32(vaddq_f32(a10, a11), vaddq_f32(a12, a13)), prescreen + j + 4);
			layer1_store(data, vaddq_f32(vaddq_f32(a20, a21), vaddq_f32(a22, a23)), prescreen + j + 8);
			layer1_store(data, vaddq_f32(vaddq_f32(a30, a31), vaddq_f32(a32, a33)), prescreen + j + 12);
		}

		// Tail: original 4-pixel loop.
		for (; j < static_cast<ptrdiff_t>(n); j += 4) {
			float32x4_t accum0 = vdupq_n_f32(0.0f);
			float32x4_t accum1 = vdupq_n_f32(0.0f);
			float32x4_t accum2 = vdupq_n_f32(0.0f);
			float32x4_t accum3 = vdupq_n_f32(0.0f);

			// Layer 0.
			for (ptrdiff_t ki = 0; ki < 4; ++ki) {
				const float *srcp = src[ki];

				for (ptrdiff_t kj = 0; kj < 16; kj += 4) {
					float32x4_t x = vld1q_f32(srcp - window_offset + j + kj);

					accum0 = vfmaq_laneq_f32(accum0, vld1q_f32(data.kernel_l0[ki * 16 + kj + 0]), x, 0);
					accum1 = vfmaq_laneq_f32(accum1, vld1q_f32(data.kernel_l0[ki * 16 + kj + 1]), x, 1);
					accum2 = vfmaq_laneq_f32(accum2, vld1q_f32(data.kernel_l0[ki * 16 + kj + 2]), x, 2);
					accum3 = vfmaq_laneq_f32(accum3, vld1q_f32(data.kernel_l0[ki * 16 + kj + 3]), x, 3);
				}
			}

			accum0 = vaddq_f32(accum0, accum1);
			accum2 = vaddq_f32(accum2, accum3);
			accum0 = vaddq_f32(accum0, accum2);

			layer1_store(data, accum0, prescreen + j);
		}
	}
};


inline FORCE_INLINE void softmax_exp_neon(float *ptr, unsigned n)
{
	for (unsigned i = 0; i < n; i += 4) {
		vst1q_f32(ptr + i, neon_softmax_exp_f32(vld1q_f32(ptr + i)));
	}
}

// 8-pixel batched sgemv: 8-row strips x 8 pixels. Streams the interleaved
// weight matrix once per 8 gathered pixels, and each column iteration is
// 2 column-vector loads + 2 input-vector loads + 16 FMAs (a 4-pixel batch
// needs 4 + 1 + 16 for the same FMA count). Per-output-row accumulation
// order over j matches the scalar kernel, so results are bit-identical to
// the pre-batching implementation.
inline FORCE_INLINE void sgemv_x8_neon(const float *matrix, const float *inputT, const float *bias, unsigned matrix_rows, unsigned matrix_cols,
                                       float *activation_softmax, float *activation_elliott, unsigned nns, const float *scale)
{
	for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(matrix_rows); i += 8) {
		float32x4_t accum00 = vdupq_n_f32(0.0f), accum01 = vdupq_n_f32(0.0f);
		float32x4_t accum10 = vdupq_n_f32(0.0f), accum11 = vdupq_n_f32(0.0f);
		float32x4_t accum20 = vdupq_n_f32(0.0f), accum21 = vdupq_n_f32(0.0f);
		float32x4_t accum30 = vdupq_n_f32(0.0f), accum31 = vdupq_n_f32(0.0f);
		float32x4_t accum40 = vdupq_n_f32(0.0f), accum41 = vdupq_n_f32(0.0f);
		float32x4_t accum50 = vdupq_n_f32(0.0f), accum51 = vdupq_n_f32(0.0f);
		float32x4_t accum60 = vdupq_n_f32(0.0f), accum61 = vdupq_n_f32(0.0f);
		float32x4_t accum70 = vdupq_n_f32(0.0f), accum71 = vdupq_n_f32(0.0f);

		for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(matrix_cols); ++j) {
			const float *col = matrix + j * static_cast<ptrdiff_t>(matrix_rows) + i;

			float32x4_t xa = vld1q_f32(inputT + j * 8 + 0);
			float32x4_t xb = vld1q_f32(inputT + j * 8 + 4);
			float32x4_t c0 = vld1q_f32(col + 0);
			float32x4_t c1 = vld1q_f32(col + 4);

			accum00 = vfmaq_laneq_f32(accum00, c0, xa, 0);
			accum01 = vfmaq_laneq_f32(accum01, c1, xa, 0);

			accum10 = vfmaq_laneq_f32(accum10, c0, xa, 1);
			accum11 = vfmaq_laneq_f32(accum11, c1, xa, 1);

			accum20 = vfmaq_laneq_f32(accum20, c0, xa, 2);
			accum21 = vfmaq_laneq_f32(accum21, c1, xa, 2);

			accum30 = vfmaq_laneq_f32(accum30, c0, xa, 3);
			accum31 = vfmaq_laneq_f32(accum31, c1, xa, 3);

			accum40 = vfmaq_laneq_f32(accum40, c0, xb, 0);
			accum41 = vfmaq_laneq_f32(accum41, c1, xb, 0);

			accum50 = vfmaq_laneq_f32(accum50, c0, xb, 1);
			accum51 = vfmaq_laneq_f32(accum51, c1, xb, 1);

			accum60 = vfmaq_laneq_f32(accum60, c0, xb, 2);
			accum61 = vfmaq_laneq_f32(accum61, c1, xb, 2);

			accum70 = vfmaq_laneq_f32(accum70, c0, xb, 3);
			accum71 = vfmaq_laneq_f32(accum71, c1, xb, 3);
		}

		float32x4_t bias0 = vld1q_f32(bias + i + 0);
		float32x4_t bias1 = vld1q_f32(bias + i + 4);

		// nns is a multiple of 16, so the strip never straddles the softmax
		// and elliott halves of the matrix.
		float *dst = i >= static_cast<ptrdiff_t>(nns) ? activation_elliott + (i - nns) : activation_softmax + i;

		vst1q_f32(dst + 0 * static_cast<ptrdiff_t>(nns) + 0, vfmaq_n_f32(bias0, accum00, scale[0]));
		vst1q_f32(dst + 0 * static_cast<ptrdiff_t>(nns) + 4, vfmaq_n_f32(bias1, accum01, scale[0]));

		vst1q_f32(dst + 1 * static_cast<ptrdiff_t>(nns) + 0, vfmaq_n_f32(bias0, accum10, scale[1]));
		vst1q_f32(dst + 1 * static_cast<ptrdiff_t>(nns) + 4, vfmaq_n_f32(bias1, accum11, scale[1]));

		vst1q_f32(dst + 2 * static_cast<ptrdiff_t>(nns) + 0, vfmaq_n_f32(bias0, accum20, scale[2]));
		vst1q_f32(dst + 2 * static_cast<ptrdiff_t>(nns) + 4, vfmaq_n_f32(bias1, accum21, scale[2]));

		vst1q_f32(dst + 3 * static_cast<ptrdiff_t>(nns) + 0, vfmaq_n_f32(bias0, accum30, scale[3]));
		vst1q_f32(dst + 3 * static_cast<ptrdiff_t>(nns) + 4, vfmaq_n_f32(bias1, accum31, scale[3]));

		vst1q_f32(dst + 4 * static_cast<ptrdiff_t>(nns) + 0, vfmaq_n_f32(bias0, accum40, scale[4]));
		vst1q_f32(dst + 4 * static_cast<ptrdiff_t>(nns) + 4, vfmaq_n_f32(bias1, accum41, scale[4]));

		vst1q_f32(dst + 5 * static_cast<ptrdiff_t>(nns) + 0, vfmaq_n_f32(bias0, accum50, scale[5]));
		vst1q_f32(dst + 5 * static_cast<ptrdiff_t>(nns) + 4, vfmaq_n_f32(bias1, accum51, scale[5]));

		vst1q_f32(dst + 6 * static_cast<ptrdiff_t>(nns) + 0, vfmaq_n_f32(bias0, accum60, scale[6]));
		vst1q_f32(dst + 6 * static_cast<ptrdiff_t>(nns) + 4, vfmaq_n_f32(bias1, accum61, scale[6]));

		vst1q_f32(dst + 7 * static_cast<ptrdiff_t>(nns) + 0, vfmaq_n_f32(bias0, accum70, scale[7]));
		vst1q_f32(dst + 7 * static_cast<ptrdiff_t>(nns) + 4, vfmaq_n_f32(bias1, accum71, scale[7]));
	}
}

// wae5 for a batch of 8 pixels: two independent 4-pixel reductions, with the
// mstd rows laid out at stride 8.
inline FORCE_INLINE void wae5_x8_neon(const float *softmax, const float *elliott, unsigned n, float *mstd)
{
	for (ptrdiff_t p = 0; p < 8; p += 4) {
		const float *sm = softmax + p * static_cast<ptrdiff_t>(n);
		const float *el = elliott + p * static_cast<ptrdiff_t>(n);

		float32x4_t vsum0 = vdupq_n_f32(0.0f), vsum1 = vdupq_n_f32(0.0f), vsum2 = vdupq_n_f32(0.0f), vsum3 = vdupq_n_f32(0.0f);
		float32x4_t wsum0 = vdupq_n_f32(0.0f), wsum1 = vdupq_n_f32(0.0f), wsum2 = vdupq_n_f32(0.0f), wsum3 = vdupq_n_f32(0.0f);

		for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); i += 4) {
			float32x4_t s0 = vld1q_f32(sm + 0 * static_cast<ptrdiff_t>(n) + i);
			float32x4_t s1 = vld1q_f32(sm + 1 * static_cast<ptrdiff_t>(n) + i);
			float32x4_t s2 = vld1q_f32(sm + 2 * static_cast<ptrdiff_t>(n) + i);
			float32x4_t s3 = vld1q_f32(sm + 3 * static_cast<ptrdiff_t>(n) + i);

			float32x4_t e0 = neon_elliott_f32(vld1q_f32(el + 0 * static_cast<ptrdiff_t>(n) + i));
			float32x4_t e1 = neon_elliott_f32(vld1q_f32(el + 1 * static_cast<ptrdiff_t>(n) + i));
			float32x4_t e2 = neon_elliott_f32(vld1q_f32(el + 2 * static_cast<ptrdiff_t>(n) + i));
			float32x4_t e3 = neon_elliott_f32(vld1q_f32(el + 3 * static_cast<ptrdiff_t>(n) + i));

			vsum0 = vfmaq_f32(vsum0, s0, e0);
			vsum1 = vfmaq_f32(vsum1, s1, e1);
			vsum2 = vfmaq_f32(vsum2, s2, e2);
			vsum3 = vfmaq_f32(vsum3, s3, e3);

			wsum0 = vaddq_f32(wsum0, s0);
			wsum1 = vaddq_f32(wsum1, s1);
			wsum2 = vaddq_f32(wsum2, s2);
			wsum3 = vaddq_f32(wsum3, s3);
		}

		// Reduce to one lane per pixel.
		float32x4_t vsum = vpaddq_f32(vpaddq_f32(vsum0, vsum1), vpaddq_f32(vsum2, vsum3));
		float32x4_t wsum = vpaddq_f32(vpaddq_f32(wsum0, wsum1), vpaddq_f32(wsum2, wsum3));

		uint32x4_t mask = vcgtq_f32(wsum, vdupq_n_f32(1e-10f));

		float32x4_t mstd0 = vld1q_f32(mstd + 0 * 8 + p);
		float32x4_t mstd1 = vld1q_f32(mstd + 1 * 8 + p);
		float32x4_t mstd3 = vld1q_f32(mstd + 3 * 8 + p);

		float32x4_t result = vmulq_n_f32(vsum, 5.0f);
		result = vdivq_f32(result, wsum);
		result = vfmaq_f32(mstd0, result, mstd1);
		result = vbslq_f32(mask, result, mstd0);

		vst1q_f32(mstd + 3 * 8 + p, vaddq_f32(mstd3, result));
	}
}


// The weight matrix is streamed once per 8 gathered pixels; the batch is
// flushed through the model whenever 8 non-prescreened pixels accumulate.
class PredictorNEONBatch8 final : public Predictor {
	static constexpr unsigned BATCH = 8;

	InterleavedPredictorModel m_model;
	double m_inv_filter_size;
	bool m_use_q2;

	void apply_model(const float *input, float *activation, float *mstd, const double *sums, const double *sumsqs) const
	{
		unsigned filter_size = m_model.xdim * m_model.ydim;
		unsigned nns = m_model.nns;

		float *activation_softmax = activation;
		float *activation_elliott = activation + BATCH * static_cast<ptrdiff_t>(nns);

		input_stddev_neon(sums, sumsqs, mstd, BATCH, BATCH, m_inv_filter_size);

		for (unsigned q = 0; q < (m_use_q2 ? 2U : 1U); ++q) {
			const float *neurons = q ? m_model.neurons_q2 : m_model.neurons_q1;
			const float *bias = q ? m_model.bias_q2 : m_model.bias_q1;

			sgemv_x8_neon(neurons, input, bias, nns * 2, filter_size, activation_softmax, activation_elliott, nns, mstd + 2 * BATCH);
			softmax_exp_neon(activation_softmax, BATCH * nns);
			wae5_x8_neon(activation_softmax, activation_elliott, nns, mstd);
		}
	}
public:
	PredictorNEONBatch8(const PredictorModel &model, bool use_q2) :
		m_model(create_interleaved_predictor_model(model)),
		m_inv_filter_size{ 1.0 / (m_model.xdim * m_model.ydim) },
		m_use_q2{ use_q2 }
	{
		assert(model.first.xdim * model.first.ydim <= 48 * 6);
	}

	size_t get_tmp_size() const noexcept override
	{
		FakeAllocator alloc;

		alloc.allocate_n<float>(48 * 6 * BATCH);
		alloc.allocate_n<float>(256 * 2 * BATCH);
		alloc.allocate_n<float>(4 * BATCH);

		return alloc.count();
	}

	void process(const float * const src[6], float *dst, const unsigned char *prescreen, void *tmp, unsigned n) const noexcept override
	{
		LinearAllocator alloc{ tmp };

		ptrdiff_t window_offset_y = 3 - static_cast<ptrdiff_t>(m_model.ydim / 2);
		ptrdiff_t window_offset_x = static_cast<ptrdiff_t>(m_model.xdim) / 2 - 1;

		float *input = alloc.allocate_n<float>(48 * 6 * BATCH);
		float *activation = alloc.allocate_n<float>(256 * 2 * BATCH);
		float *mstd = alloc.allocate_n<float>(4 * BATCH);

		double sums[BATCH] = {};
		double sumsqs[BATCH] = {};
		unsigned gathered_idx[BATCH];
		size_t num_gathered = 0;

		for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
			if (prescreen[i])
				continue;

			gather_pixels_neon(src + window_offset_y, i - window_offset_x, m_model.xdim, m_model.ydim,
			                   input, BATCH, static_cast<unsigned>(num_gathered), &sums[num_gathered], &sumsqs[num_gathered]);
			gathered_idx[num_gathered] = static_cast<unsigned>(i);
			++num_gathered;

			if (num_gathered == BATCH) {
				apply_model(input, activation, mstd, sums, sumsqs);

				for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(BATCH); ++idx) {
					dst[gathered_idx[idx]] = mstd[3 * BATCH + idx] * (m_use_q2 ? 0.5f : 1.0f);
				}

				num_gathered = 0;
			}
		}
		if (num_gathered) {
			apply_model(input, activation, mstd, sums, sumsqs);

			for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(num_gathered); ++idx) {
				dst[gathered_idx[idx]] = mstd[3 * BATCH + idx] * (m_use_q2 ? 0.5f : 1.0f);
			}
		}
	}
};

} // namespace


void byte_to_float_neon(const void *src, void *dst, size_t n)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		uint16x8_t x = vmovl_u8(vld1_u8(src_p + i));
		vst1q_f32(dst_p + i + 0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(x))));
		vst1q_f32(dst_p + i + 4, vcvtq_f32_u32(vmovl_high_u16(x)));
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		dst_p[i] = src_p[i];
	}
}

void word_to_float_neon(const void *src, void *dst, size_t n)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		uint16x8_t x = vld1q_u16(src_p + i);
		vst1q_f32(dst_p + i + 0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(x))));
		vst1q_f32(dst_p + i + 4, vcvtq_f32_u32(vmovl_high_u16(x)));
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		dst_p[i] = src_p[i];
	}
}

void half_to_float_neon(const void *src, void *dst, size_t n)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		float16x8_t x = vreinterpretq_f16_u16(vld1q_u16(src_p + i));
		vst1q_f32(dst_p + i + 0, vcvt_f32_f16(vget_low_f16(x)));
		vst1q_f32(dst_p + i + 4, vcvt_high_f32_f16(x));
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		__fp16 x;
		std::copy_n(reinterpret_cast<const unsigned char *>(src_p + i), sizeof(x), reinterpret_cast<unsigned char *>(&x));
		dst_p[i] = x;
	}
}

void float_to_byte_neon(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		uint32x4_t lo = vcvtnq_u32_f32(vmaxq_f32(vld1q_f32(src_p + i + 0), vdupq_n_f32(0.0f)));
		uint32x4_t hi = vcvtnq_u32_f32(vmaxq_f32(vld1q_f32(src_p + i + 4), vdupq_n_f32(0.0f)));
		uint16x8_t x = vcombine_u16(vqmovn_u32(lo), vqmovn_u32(hi));
		vst1_u8(dst_p + i, vqmovn_u16(x));
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		float x = std::min(std::max(src_p[i], 0.0f), 255.0f);
		dst_p[i] = static_cast<uint8_t>(std::lrint(x));
	}
}

void float_to_word_neon(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		uint32x4_t lo = vcvtnq_u32_f32(vmaxq_f32(vld1q_f32(src_p + i + 0), vdupq_n_f32(0.0f)));
		uint32x4_t hi = vcvtnq_u32_f32(vmaxq_f32(vld1q_f32(src_p + i + 4), vdupq_n_f32(0.0f)));
		vst1q_u16(dst_p + i, vcombine_u16(vqmovn_u32(lo), vqmovn_u32(hi)));
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		float x = std::min(std::max(src_p[i], 0.0f), 65535.0f);
		dst_p[i] = static_cast<uint16_t>(std::lrint(x));
	}
}

void float_to_half_neon(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	for (size_t i = 0; i < n - n % 8; i += 8) {
		float16x4_t lo = vcvt_f16_f32(vld1q_f32(src_p + i + 0));
		float16x8_t x = vcvt_high_f16_f32(lo, vld1q_f32(src_p + i + 4));
		vst1q_u16(dst_p + i, vreinterpretq_u16_f16(x));
	}
	for (size_t i = n - n % 8; i < n; ++i) {
		__fp16 x = src_p[i];
		std::copy_n(reinterpret_cast<const unsigned char *>(&x), sizeof(x), reinterpret_cast<unsigned char *>(dst_p + i));
	}
}

void cubic_interpolation_neon(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n)
{
	const float *srcp0 = src[0];
	const float *srcp1 = src[1];
	const float *srcp2 = src[2];
	const float *srcp3 = src[3];

	const float32x4_t k0 = vdupq_n_f32(-3.0f / 32.0f);
	const float32x4_t k1 = vdupq_n_f32(19.0f / 32.0f);

	for (unsigned i = 0; i < n; i += 8) {
		uint16x8_t pscrn_w = vmovl_u8(vld1_u8(prescreen + i));
		uint32x4_t mask_lo = vtstq_u32(vmovl_u16(vget_low_u16(pscrn_w)), vdupq_n_u32(UINT32_MAX));
		uint32x4_t mask_hi = vtstq_u32(vmovl_high_u16(pscrn_w), vdupq_n_u32(UINT32_MAX));

		float32x4_t accum_lo = vmulq_f32(k0, vld1q_f32(srcp0 + i + 0));
		float32x4_t accum_hi = vmulq_f32(k0, vld1q_f32(srcp0 + i + 4));
		accum_lo = vfmaq_f32(accum_lo, k1, vld1q_f32(srcp1 + i + 0));
		accum_hi = vfmaq_f32(accum_hi, k1, vld1q_f32(srcp1 + i + 4));
		accum_lo = vfmaq_f32(accum_lo, k1, vld1q_f32(srcp2 + i + 0));
		accum_hi = vfmaq_f32(accum_hi, k1, vld1q_f32(srcp2 + i + 4));
		accum_lo = vfmaq_f32(accum_lo, k0, vld1q_f32(srcp3 + i + 0));
		accum_hi = vfmaq_f32(accum_hi, k0, vld1q_f32(srcp3 + i + 4));

		accum_lo = vbslq_f32(mask_lo, accum_lo, vld1q_f32(dst + i + 0));
		accum_hi = vbslq_f32(mask_hi, accum_hi, vld1q_f32(dst + i + 4));

		vst1q_f32(dst + i + 0, accum_lo);
		vst1q_f32(dst + i + 4, accum_hi);
	}
}


std::unique_ptr<Prescreener> create_prescreener_old_neon(const PrescreenerOldCoefficients &coeffs, double pixel_half)
{
	return std::make_unique<PrescreenerOldNEON>(coeffs, pixel_half);
}

std::unique_ptr<Prescreener> create_prescreener_new_neon(const PrescreenerNewCoefficients &coeffs, double pixel_half)
{
	return std::make_unique<PrescreenerNewNEON>(coeffs, pixel_half);
}

std::unique_ptr<Predictor> create_predictor_neon(const PredictorModel &model, bool use_q2)
{
	return std::make_unique<PredictorNEONBatch8>(model, use_q2);
}

} // namespace znedi3

#endif // ZNEDI3_ARM
