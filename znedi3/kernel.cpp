#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include "kernel.h"
#include "weights.h"
#include "znedi3_impl.h"

#ifdef ZNEDI3_X86
  #include "x86/kernel_x86.h"
#endif

namespace znedi3 {
namespace {

template <class T>
void integer_to_float(const void *src, void *dst, size_t n)
{
	const T *src_p = static_cast<const T *>(src);
	float *dst_p = static_cast<float *>(dst);

	std::transform(src_p, src_p + n, dst_p, [](T x) { return static_cast<float>(x); });
}

template <class T>
void float_to_integer(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	T *dst_p = static_cast<T *>(dst);

	std::transform(src_p, src_p + n, dst_p, [](float x)
	{
		x = std::min(std::max(x, 0.0f), static_cast<float>(std::numeric_limits<T>::max()));
		return static_cast<T>(std::lrint(x));
	});
}

float dot_product(const float *kernel, const float *input, unsigned n, float scale, float bias)
{
	float accum = 0.0f;

	for (unsigned i = 0; i < n; ++i) {
		accum += kernel[i] * input[i];
	}
	return accum * scale + bias;
}

float elliott(float x)
{
	return x / (1.0f + std::fabs(x));
}

float softmax_exp(float x)
{
	return std::exp(std::min(std::max(x, -80.0f), 80.0f));
}

void cubic_interpolation_c(const void *src, ptrdiff_t src_stride, void *dst, const unsigned char *prescreen, unsigned n)
{
	const float *src_p = static_cast<const float *>(src);
	float *dst_p = static_cast<float *>(dst);
	ptrdiff_t src_stride_f = src_stride / sizeof(float);

	const float *window = src_p - 2 * src_stride_f;

	for (unsigned i = 0; i < n; ++i) {
		if (!prescreen[i])
			continue;

		float accum = 0.0f;
		accum += (-3.0f / 32.0f) * window[0 * src_stride_f + i];
		accum += (19.0f / 32.0f) * window[1 * src_stride_f + i];
		accum += (19.0f / 32.0f) * window[2 * src_stride_f + i];
		accum += (-3.0f / 32.0f) * window[3 * src_stride_f + i];

		dst_p[i] = accum;
	}
}


class PrescreenerOldC final : public Prescreener {
	PrescreenerOldCoefficients m_data;
public:
	PrescreenerOldC(const PrescreenerOldCoefficients &data, double half) :
		m_data(data)
	{
		subtract_mean(m_data, half);
	}

	size_t get_tmp_size() const override { return 0; }

	void process(const void *src, ptrdiff_t src_stride, unsigned char *prescreen, void *, unsigned n) const override
	{
		const float *src_p = static_cast<const float *>(src);
		ptrdiff_t src_stride_f = src_stride / sizeof(float);

		// Adjust source pointer to point to top-left of filter window.
		const float *window = src_p - 2 * src_stride_f - 5;

		for (unsigned j = 0; j < n; ++j) {
			float input[48];
			float state[12];

			for (unsigned i = 0; i < 4; ++i) {
				std::copy_n(window + i * src_stride_f + j, 12, input + i * 12);
			}

			// Layer 0.
			for (unsigned n = 0; n < 4; ++n) {
				state[n] = dot_product(m_data.kernel_l0[n], input, 48, 1.0f, m_data.bias_l0[n]);
			}
			std::transform(state + 1, state + 4, state + 1, elliott);

			// Layer 1.
			for (unsigned n = 0; n < 4; ++n) {
				state[n + 4] = dot_product(m_data.kernel_l1[n], state, 4, 1.0f, m_data.bias_l1[n]);
			}
			std::transform(state + 4, state + 8, state + 4, elliott);

			// Layer 2.
			for (unsigned n = 0; n < 4; ++n) {
				state[n + 8] = dot_product(m_data.kernel_l2[n], state, 8, 1.0f, m_data.bias_l2[n]);
			}

			prescreen[j] = std::max(state[10], state[11]) <= std::max(state[8], state[9]) ? UCHAR_MAX : 0;
		}
	}
};


class PrescreenerNewC final : public Prescreener {
	PrescreenerNewCoefficients m_data;

	void gather_input(const float *src, ptrdiff_t src_stride, float buf[48]) const
	{
		ptrdiff_t src_stride_f = src_stride / 4;

		for (unsigned i = 0; i < 4; ++i) {
			std::copy_n(src + i * src_stride_f, 16, buf + i * 16);
		}
	}
public:
	PrescreenerNewC(const PrescreenerNewCoefficients &data, double half) :
		m_data(data)
	{
		subtract_mean(m_data, half);
	}

	size_t get_tmp_size() const override { return 0; }

	void process(const void *src, ptrdiff_t src_stride, unsigned char *prescreen, void *, unsigned n) const override
	{
		const float *src_p = static_cast<const float *>(src);
		ptrdiff_t src_stride_f = src_stride / sizeof(float);

		// Adjust source pointer to point to top-left of filter window.
		const float *window = src_p - 2 * src_stride_f - 6;

		for (unsigned j = 0; j < n; j += 4) {
			float input[64];
			float state[8];

			for (unsigned i = 0; i < 4; ++i) {
				std::copy_n(window + i * src_stride_f + j, 16, input + i * 16);
			}

			for (unsigned n = 0; n < 4; ++n) {
				state[n] = dot_product(m_data.kernel_l0[n], input, 64, 1.0f, m_data.bias_l0[n]);
			}
			std::transform(state, state + 4, state, elliott);

			for (unsigned n = 0; n < 4; ++n) {
				state[n + 4] = dot_product(m_data.kernel_l1[n], state, 4, 1.0f, m_data.bias_l1[n]);
			}

			for (unsigned n = 0; n < 4; ++n) {
				prescreen[j + n] = state[n + 4] > 0.0f;
			}
		}
	}
};


class PredictorC final : public Predictor {
	PredictorModel m_model;
	bool m_use_q2;

	size_t filter_offset(unsigned nn) const { return static_cast<size_t>(nn) * m_model.first.xdim * m_model.first.ydim; }

	const float *softmax_q1_filter(unsigned nn) const { return m_model.second.softmax_q1 + filter_offset(nn); }
	const float *elliott_q1_filter(unsigned nn) const { return m_model.second.elliott_q1 + filter_offset(nn); }
	const float *softmax_q2_filter(unsigned nn) const { return m_model.second.softmax_q2 + filter_offset(nn); }
	const float *elliott_q2_filter(unsigned nn) const { return m_model.second.elliott_q2 + filter_offset(nn); }

	void gather_input(const float *src, ptrdiff_t src_stride, float *buf, float mstd[4]) const
	{
		ptrdiff_t src_stride_f = src_stride / sizeof(float);

		double sum = 0;
		double sum_sq = 0;

		for (unsigned i = 0; i < m_model.first.ydim; ++i) {
			for (unsigned j = 0; j < m_model.first.xdim; ++j) {
				float val = src[i * src_stride_f + j];

				buf[i * m_model.first.xdim + j] = val;
				sum += val;
				sum_sq += static_cast<double>(val) * val;
			}
		}

		mstd[0] = static_cast<float>(sum / (m_model.first.xdim * m_model.first.ydim));
		mstd[3] = 0.0f;

		double tmp = sum_sq / (m_model.first.xdim * m_model.first.ydim) - static_cast<double>(mstd[0]) * mstd[0];
		if (tmp < FLT_EPSILON) {
			mstd[1] = 0.0f;
			mstd[2] = 0.0f;
		} else {
			mstd[1] = static_cast<float>(std::sqrt(tmp));
			mstd[2] = 1.0f / mstd[1];
		}
	}

	void wae5(const float *softmax, const float *elliott, unsigned n, float mstd[4]) const
	{
		float vsum = 0.0f;
		float wsum = 0.0f;

		for (unsigned i = 0; i < n; ++i) {
			vsum += softmax[i] * znedi3::elliott(elliott[i]);
			wsum += softmax[i];
		}

		if (wsum > 1e-10f)
			mstd[3] += (5.0f * vsum) / wsum * mstd[1] + mstd[0];
		else
			mstd[3] += mstd[0];
	}
public:
	PredictorC(const PredictorModel &model, bool use_q2) :
		m_model{ copy_model(model) },
		m_use_q2{ use_q2 }
	{
		assert(model.first.xdim * model.first.ydim <= 48 * 6);
		subtract_mean(m_model);
	}

	size_t get_tmp_size() const override { return 0; }

	void process(const void *src, ptrdiff_t src_stride, void *dst, const unsigned char *prescreen, void *, unsigned n) const override
	{
		const float *src_p = static_cast<const float *>(src);
		float *dst_p = static_cast<float *>(dst);
		ptrdiff_t src_stride_f = src_stride / sizeof(float);

		// Adjust source pointer to point to top-left of filter window.
		const float *window = src_p - static_cast<ptrdiff_t>(m_model.first.ydim / 2) * src_stride_f - (m_model.first.xdim / 2 - 1);
		unsigned filter_size = m_model.first.xdim * m_model.first.ydim;
		unsigned nns = m_model.first.nns;

		for (unsigned i = 0; i < n; ++i) {
			if (prescreen[i])
				continue;

			float input[48 * 6];
			float activation[256 * 2];
			float mstd[4];

			gather_input(window + i, src_stride, input, mstd);
			float scale = mstd[2];

			for (unsigned nn = 0; nn < nns; ++nn) {
				activation[nn] = dot_product(softmax_q1_filter(nn), input, filter_size, scale, m_model.second.softmax_bias_q1[nn]);
			}
			for (unsigned nn = 0; nn < nns; ++nn) {
				activation[m_model.first.nns + nn] = dot_product(elliott_q1_filter(nn), input, filter_size, scale, m_model.second.elliott_bias_q1[nn]);
			}

			std::transform(activation, activation + nns, activation, softmax_exp);
			wae5(activation, activation + nns, nns, mstd);

			if (m_use_q2) {
				for (unsigned nn = 0; nn < nns; ++nn) {
					activation[nn] = dot_product(softmax_q2_filter(nn), input, filter_size, scale, m_model.second.softmax_bias_q2[nn]);
				}
				for (unsigned nn = 0; nn < nns; ++nn) {
					activation[nns + nn] = dot_product(elliott_q2_filter(nn), input, filter_size, scale, m_model.second.elliott_bias_q2[nn]);
				}

				std::transform(activation, activation + nns, activation, softmax_exp);
				wae5(activation, activation + nns, nns, mstd);
			}

			dst_p[i] = mstd[3] / (m_use_q2 ? 2 : 1);
		}
	}
};

} // namespace


pixel_io_func select_pixel_io_func(PixelType in, PixelType out, CPUClass cpu)
{
	pixel_io_func ret = nullptr;

#ifdef ZNEDI3_X86
	ret = select_pixel_io_func_x86(in, out, cpu);
#endif

	if (!ret && in == PixelType::BYTE && out == PixelType::FLOAT)
		ret = integer_to_float<uint8_t>;
	if (!ret && in == PixelType::WORD && out == PixelType::FLOAT)
		ret = integer_to_float<uint16_t>;
	if (!ret && in == PixelType::FLOAT && out == PixelType::BYTE)
		ret = float_to_integer<uint8_t>;
	if (!ret && in == PixelType::FLOAT && out == PixelType::WORD)
		ret = float_to_integer<uint16_t>;

	return ret;
}

interpolate_func select_interpolate_func(CPUClass cpu)
{
	interpolate_func ret = nullptr;

#ifdef ZNEDI3_X86
	ret = select_interpolate_func_x86(cpu);
#endif

	if (!ret)
		ret = cubic_interpolation_c;

	assert(ret);
	return ret;
}

std::unique_ptr<Prescreener> create_prescreener_old(const PrescreenerOldCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	std::unique_ptr<Prescreener> ret;

#ifdef ZNEDI3_X86
	ret = create_prescreener_old_x86(coeffs, pixel_half, cpu);
#endif

	if (!ret)
		ret = std::make_unique<PrescreenerOldC>(coeffs, pixel_half);

	assert(ret);
	return ret;
}

std::unique_ptr<Prescreener> create_prescreener_new(const PrescreenerNewCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	std::unique_ptr<Prescreener> ret;

#ifdef ZNEDI3_X86
	ret = create_prescreener_new_x86(coeffs, pixel_half, cpu);
#endif

	if (!ret)
		ret = std::make_unique<PrescreenerNewC>(coeffs, pixel_half);

	assert(ret);
	return ret;
}

std::unique_ptr<Predictor> create_predictor(const std::pair<const PredictorTraits, PredictorCoefficients> &model, bool use_q2, CPUClass cpu)
{
	std::unique_ptr<Predictor> ret;

#ifdef ZNEDI3_X86
	ret = create_predictor_x86(model, use_q2, cpu);
#endif

	if (!ret)
		ret = std::make_unique<PredictorC>(model, use_q2);

	assert(ret);
	return ret;
}

} // namespace znedi3
