#pragma once

#ifdef ZNEDI3_X86

#ifndef ZNEDI3_X86_KERNEL_SSE_COMMON_H_
#define ZNEDI3_X86_KERNEL_SSE_COMMON_H_

#ifdef KERNEL_IMPL_INCLUDED
  #error Must not include multiple impl headers
#endif

#define KERNEL_IMPL_INCLUDED

#include <cassert>
#include <cstddef>
#include "ccdep.h"
#include "kernel.h"
#include "kernel_x86.h"

namespace znedi3 {
namespace {

inline FORCE_INLINE __m128 mm_rsqrt24_ss(__m128 x)
{
	__m128 tmp0 = _mm_rsqrt_ss(x);
	__m128 tmp1 = _mm_mul_ss(x, tmp0);
	__m128 tmp2 = _mm_mul_ss(_mm_set_ps1(0.5f), tmp0);
	__m128 tmp3 = _mm_sub_ss(_mm_set_ps1(3.0f), _mm_mul_ss(tmp1, tmp0));
	return _mm_mul_ss(tmp2, tmp3);
}

inline FORCE_INLINE __m128 mm_rcp24_ps(__m128 x)
{
	__m128 tmp0 = _mm_rcp_ps(x);
	__m128 tmp1 = _mm_sub_ps(_mm_set_ps1(1.0f), _mm_mul_ps(x, tmp0));
	__m128 tmp2 = _mm_sub_ps(tmp0, _mm_mul_ps(tmp0, tmp1));
	return tmp2;
}

inline FORCE_INLINE void sgemv_sse(const float *matrix, const float *vector, const float *bias, unsigned matrix_rows, unsigned matrix_cols, float scale, float *result)
{
	for (ptrdiff_t i = 0; i < matrix_rows; i += 16) {
		__m128 accum0 = _mm_setzero_ps();
		__m128 accum1 = _mm_setzero_ps();
		__m128 accum2 = _mm_setzero_ps();
		__m128 accum3 = _mm_setzero_ps();

		for (ptrdiff_t j = 0; j < matrix_cols; ++j) {
			__m128 x = _mm_set_ps1(vector[j]);
			__m128 coeffs;

			coeffs = _mm_load_ps(matrix + j * matrix_rows + i + 0);
			accum0 = _mm_add_ps(accum0, _mm_mul_ps(coeffs, x));

			coeffs = _mm_load_ps(matrix + j * matrix_rows + i + 4);
			accum1 = _mm_add_ps(accum1, _mm_mul_ps(coeffs, x));

			coeffs = _mm_load_ps(matrix + j * matrix_rows + i + 8);
			accum2 = _mm_add_ps(accum2, _mm_mul_ps(coeffs, x));

			coeffs = _mm_load_ps(matrix + j * matrix_rows + i + 12);
			accum3 = _mm_add_ps(accum3, _mm_mul_ps(coeffs, x));
		}

		__m128 scale_ps = _mm_set_ps1(scale);
		accum0 = _mm_mul_ps(accum0, scale_ps);
		accum1 = _mm_mul_ps(accum1, scale_ps);
		accum2 = _mm_mul_ps(accum2, scale_ps);
		accum3 = _mm_mul_ps(accum3, scale_ps);

		accum0 = _mm_add_ps(accum0, _mm_load_ps(bias + i + 0));
		accum1 = _mm_add_ps(accum1, _mm_load_ps(bias + i + 4));
		accum2 = _mm_add_ps(accum2, _mm_load_ps(bias + i + 8));
		accum3 = _mm_add_ps(accum3, _mm_load_ps(bias + i + 12));

		_mm_store_ps(result + i + 0, accum0);
		_mm_store_ps(result + i + 4, accum1);
		_mm_store_ps(result + i + 8, accum2);
		_mm_store_ps(result + i + 12, accum3);
	}
}


template <class Traits>
class PredictorSSEBase : public Predictor {
	InterleavedPredictorModel m_model;
	double m_inv_filter_size;
	bool m_use_q2;

	void apply_model(const float *input, float *activation, float *mstd) const
	{
		unsigned filter_size = m_model.xdim * m_model.ydim;
		unsigned nns = m_model.nns;

		float *activation_softmax = activation;
		float *activation_elliott = activation + nns;

		for (unsigned q = 0; q < (m_use_q2 ? 2U : 1U); ++q) {
			const float *neurons = q ? m_model.neurons_q2 : m_model.neurons_q1;
			const float *bias = q ? m_model.bias_q2 : m_model.bias_q1;

			sgemv_sse(neurons, input, bias, nns * 2, filter_size, mstd[2], activation);
			Traits::softmax_exp(activation_softmax, nns);
			Traits::wae5(activation_softmax, activation_elliott, nns, mstd);
		}
	}
public:
	PredictorSSEBase(const PredictorModel &model, bool use_q2) :
		m_model(create_interleaved_predictor_model(model)),
		m_inv_filter_size{ 1.0 / (m_model.xdim * m_model.ydim) },
		m_use_q2{ use_q2 }
	{
		assert(model.first.xdim * model.first.ydim <= 48 * 6);
	}

	size_t get_tmp_size() const override { return 0; }

	void process(const void *src, ptrdiff_t src_stride, void *dst, const unsigned char *prescreen, void *, unsigned n) const override
	{
		const float *src_p = static_cast<const float *>(src);
		float *dst_p = static_cast<float *>(dst);
		ptrdiff_t src_stride_f = src_stride / sizeof(float);

		// Adjust source pointer to point to top-left of filter window.
		const float *window = src_p - static_cast<ptrdiff_t>(m_model.ydim / 2) * src_stride_f - (m_model.xdim / 2 - 1);
		unsigned filter_size = m_model.xdim * m_model.ydim;
		unsigned nns = m_model.nns;

		for (unsigned i = 0; i < n; ++i) {
			if (prescreen[i])
				continue;

			alignas(16) float input[48 * 6];
			alignas(16) float activation[256 * 2];
			alignas(16) float mstd[4];

			Traits::gather_input(window + i, src_stride, m_model.xdim, m_model.ydim, input, mstd, m_inv_filter_size);
			apply_model(input, activation, mstd);

			dst_p[i] = mstd[3] * (m_use_q2 ? 0.5f : 1.0f);
		}
	}
};

} // namespace
} // namespace znedi3

#endif // ZNEDI3_X86_KERNEL_SSE_COMMON_H_

#endif // ZNEDI3_X86
