// Non-streaming driver for the SME predictor: pixel gathering, statistics,
// and NEON post-processing. Compiled without SME/SVE target features; only
// the matmul runs in streaming mode, called through the entry points in
// kernel_sme.cpp.

#if defined(ZNEDI3_ARM) && defined(ZNEDI3_ARM_SME)

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

// Maximum architectural streaming vector length is 2048 bits (64 words).
constexpr unsigned SVL_WORDS_MAX = 64;


// Bias, scaling, activation functions, and weighted average, applied to the
// activations produced by the streaming-mode matmul.
void postprocess_neon(float *activation, const float *bias, const float *mstd, float *mstd3, unsigned rows, unsigned nns, unsigned num, unsigned batch)
{
	for (unsigned p = 0; p < num; ++p) {
		float *act = activation + static_cast<size_t>(p) * rows;
		float scale = mstd[2 * batch + p];

		float32x4_t vsum = vdupq_n_f32(0.0f);
		float32x4_t wsum = vdupq_n_f32(0.0f);

		for (unsigned r = 0; r < nns; r += 4) {
			float32x4_t s = vfmaq_n_f32(vld1q_f32(bias + r), vld1q_f32(act + r), scale);
			s = neon_softmax_exp_f32(s);

			float32x4_t e = vfmaq_n_f32(vld1q_f32(bias + nns + r), vld1q_f32(act + nns + r), scale);
			e = neon_elliott_f32(e);

			vsum = vfmaq_f32(vsum, s, e);
			wsum = vaddq_f32(wsum, s);
		}

		float vsum_reduced = vaddvq_f32(vsum);
		float wsum_reduced = vaddvq_f32(wsum);

		if (wsum_reduced > 1e-10f)
			mstd3[p] += (5.0f * vsum_reduced) / wsum_reduced * mstd[1 * batch + p] + mstd[0 * batch + p];
		else
			mstd3[p] += mstd[0 * batch + p];
	}
}

class PredictorSME final : public Predictor {
	InterleavedPredictorModel m_model;
	double m_inv_filter_size;
	bool m_use_q2;
	unsigned m_batch;

	void apply_model(const float *inputT, float *activation, float *mstd, const double *sums, const double *sumsqs, unsigned num) const
	{
		unsigned filter_size = m_model.xdim * m_model.ydim;
		unsigned nns = m_model.nns;
		unsigned rows = nns * 2;

		input_stddev_neon(sums, sumsqs, mstd, num, m_batch, m_inv_filter_size);

		for (unsigned q = 0; q < (m_use_q2 ? 2U : 1U); ++q) {
			const float *neurons = q ? m_model.neurons_q2 : m_model.neurons_q1;
			const float *bias = q ? m_model.bias_q2 : m_model.bias_q1;

			sme_matmul(neurons, inputT, activation, rows, filter_size, m_batch, num);
			postprocess_neon(activation, bias, mstd, mstd + 3 * m_batch, rows, nns, num, m_batch);
		}
	}
public:
	PredictorSME(const PredictorModel &model, bool use_q2) :
		m_model(create_interleaved_predictor_model(model)),
		m_inv_filter_size{ 1.0 / (m_model.xdim * m_model.ydim) },
		m_use_q2{ use_q2 },
		m_batch{ sme_streaming_vector_length_words() }
	{
		assert(model.first.xdim * model.first.ydim <= 48 * 6);
		assert(m_batch <= SVL_WORDS_MAX);
	}

	size_t get_tmp_size() const noexcept override
	{
		FakeAllocator alloc;

		alloc.allocate_n<float>(48 * 6 * m_batch);
		alloc.allocate_n<float>(256 * 2 * m_batch);
		alloc.allocate_n<float>(4 * m_batch);

		return alloc.count();
	}

	void process(const float * const src[6], float *dst, const unsigned char *prescreen, void *tmp, unsigned n) const noexcept override
	{
		LinearAllocator alloc{ tmp };

		ptrdiff_t window_offset_y = 3 - static_cast<ptrdiff_t>(m_model.ydim / 2);
		ptrdiff_t window_offset_x = static_cast<ptrdiff_t>(m_model.xdim) / 2 - 1;

		float *input = alloc.allocate_n<float>(48 * 6 * m_batch);
		float *activation = alloc.allocate_n<float>(256 * 2 * m_batch);
		float *mstd = alloc.allocate_n<float>(4 * m_batch);

		double sums[SVL_WORDS_MAX] = {};
		double sumsqs[SVL_WORDS_MAX] = {};
		unsigned gathered_idx[SVL_WORDS_MAX];
		unsigned num_gathered = 0;

		for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
			if (prescreen[i])
				continue;

			gather_pixels_neon(src + window_offset_y, i - window_offset_x, m_model.xdim, m_model.ydim,
			                   input, m_batch, num_gathered, &sums[num_gathered], &sumsqs[num_gathered]);
			gathered_idx[num_gathered] = static_cast<unsigned>(i);
			++num_gathered;

			if (num_gathered == m_batch) {
				apply_model(input, activation, mstd, sums, sumsqs, num_gathered);

				for (unsigned idx = 0; idx < num_gathered; ++idx) {
					dst[gathered_idx[idx]] = mstd[3 * m_batch + idx] * (m_use_q2 ? 0.5f : 1.0f);
				}

				num_gathered = 0;
			}
		}
		if (num_gathered) {
			apply_model(input, activation, mstd, sums, sumsqs, num_gathered);

			for (unsigned idx = 0; idx < num_gathered; ++idx) {
				dst[gathered_idx[idx]] = mstd[3 * m_batch + idx] * (m_use_q2 ? 0.5f : 1.0f);
			}
		}
	}
};

} // namespace


std::unique_ptr<Predictor> create_predictor_sme(const PredictorModel &model, bool use_q2)
{
	return std::make_unique<PredictorSME>(model, use_q2);
}

} // namespace znedi3

#endif // ZNEDI3_ARM && ZNEDI3_ARM_SME
