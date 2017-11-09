#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <vector>
#include "weights.h"

namespace znedi3 {
namespace {

double mean(const float *buf, size_t n)
{
	return std::accumulate(buf, buf + n, 0.0) / n;
}

} // namespace


PredictorModel allocate_model(const PredictorTraits &traits)
{
	size_t filter_size = static_cast<size_t>(traits.nns) * traits.xdim * traits.ydim;
	size_t bias_size = traits.nns;

	PredictorCoefficients coeffs;
	coeffs.data.reset(new float[(filter_size + bias_size) * 4]);

	float *ptr = coeffs.data.get();
	auto allocate = [&](size_t size) { float *ret = ptr; ptr += size; return ret; };

	coeffs.softmax_q1 = allocate(filter_size);
	coeffs.elliott_q1 = allocate(filter_size);
	coeffs.softmax_bias_q1 = allocate(bias_size);
	coeffs.elliott_bias_q1 = allocate(bias_size);

	coeffs.softmax_q2 = allocate(filter_size);
	coeffs.elliott_q2 = allocate(filter_size);
	coeffs.softmax_bias_q2 = allocate(bias_size);
	coeffs.elliott_bias_q2 = allocate(bias_size);

	return{ traits, std::move(coeffs) };
}

PredictorModel copy_model(const PredictorModel &model)
{
	size_t filter_size = model.first.nns * model.first.xdim * model.first.ydim;
	size_t bias_size = model.first.nns;

	PredictorModel m = allocate_model(model.first);

	std::copy_n(model.second.softmax_q1, filter_size, m.second.softmax_q1);
	std::copy_n(model.second.elliott_q1, filter_size, m.second.elliott_q1);
	std::copy_n(model.second.softmax_bias_q1, bias_size, m.second.softmax_bias_q1);
	std::copy_n(model.second.elliott_bias_q1, bias_size, m.second.elliott_bias_q1);

	std::copy_n(model.second.softmax_q2, filter_size, m.second.softmax_q2);
	std::copy_n(model.second.elliott_q2, filter_size, m.second.elliott_q2);
	std::copy_n(model.second.softmax_bias_q2, bias_size, m.second.softmax_bias_q2);
	std::copy_n(model.second.elliott_bias_q2, bias_size, m.second.elliott_bias_q2);

	return m;
}

void subtract_mean(PrescreenerOldCoefficients &coeffs, double pixel_half)
{
	for (unsigned n = 0; n < 4; ++n) {
		double m = mean(coeffs.kernel_l0[n], 48);

		std::transform(&coeffs.kernel_l0[n][0], &coeffs.kernel_l0[n][48], &coeffs.kernel_l0[n][0], [=](float x)
		{
			return static_cast<float>((x - m) / pixel_half);
		});
	}
}

void subtract_mean(PrescreenerNewCoefficients &coeffs, double pixel_half)
{
	for (unsigned n = 0; n < 4; ++n) {
		double m = mean(coeffs.kernel_l0[n], 64);

		std::transform(&coeffs.kernel_l0[n][0], &coeffs.kernel_l0[n][64], &coeffs.kernel_l0[n][0], [=](float x)
		{
			return static_cast<float>((x - m) / pixel_half);
		});
	}
}

void subtract_mean(PredictorModel &model)
{
	size_t filter_size = model.first.xdim * model.first.ydim;
	unsigned nns = model.first.nns;

	std::vector<double> softmax_means(256); // Average of individual softmax filters.
	std::vector<double> elliott_means(256); // Average of individual elliott filters.
	std::vector<double> mean_filter(48 * 6); // Pointwise average of all softmax filters.
	double mean_bias;

	// Quality 1.
	for (unsigned nn = 0; nn < nns; ++nn) {
		softmax_means[nn] = mean(model.second.softmax_q1 + nn * filter_size, filter_size);
		elliott_means[nn] = mean(model.second.elliott_q1 + nn * filter_size, filter_size);

		for (unsigned k = 0; k < filter_size; ++k) {
			mean_filter[k] += model.second.softmax_q1[nn * filter_size + k] - softmax_means[nn];
		}
	}
	for (unsigned k = 0; k < filter_size; ++k) {
		mean_filter[k] /= nns;
	}
	mean_bias = mean(model.second.softmax_bias_q1, nns);

	for (unsigned nn = 0; nn < nns; ++nn) {
		for (unsigned k = 0; k < filter_size; ++k) {
			model.second.softmax_q1[nn * filter_size + k] -= static_cast<float>(softmax_means[nn] + mean_filter[k]);
			model.second.elliott_q1[nn * filter_size + k] -= static_cast<float>(elliott_means[nn]);
		}
		model.second.softmax_bias_q1[nn] -= static_cast<float>(mean_bias);
	}

	// Quality 2.
	mean_filter.assign(48 * 6, 0.0);

	for (unsigned nn = 0; nn < nns; ++nn) {
		softmax_means[nn] = mean(model.second.softmax_q2 + nn * filter_size, filter_size);
		elliott_means[nn] = mean(model.second.elliott_q2 + nn * filter_size, filter_size);

		for (unsigned k = 0; k < filter_size; ++k) {
			mean_filter[k] += model.second.softmax_q2[nn * filter_size + k] - softmax_means[nn];
		}
	}
	for (unsigned k = 0; k < filter_size; ++k) {
		mean_filter[k] /= nns;
	}
	mean_bias = mean(model.second.softmax_bias_q2, nns);

	for (unsigned nn = 0; nn < nns; ++nn) {
		for (unsigned k = 0; k < filter_size; ++k) {
			model.second.softmax_q2[nn * filter_size + k] -= static_cast<float>(softmax_means[nn] + mean_filter[k]);
			model.second.elliott_q2[nn * filter_size + k] -= static_cast<float>(elliott_means[nn]);
		}
		model.second.softmax_bias_q2[nn] -= static_cast<float>(mean_bias);
	}
}

std::unique_ptr<NNEDI3Weights> read_nnedi3_weights(const float *data)
{
	std::unique_ptr<NNEDI3Weights> nnedi3 = std::make_unique<NNEDI3Weights>();
	size_t size = NNEDI3_WEIGHTS_SIZE / sizeof(float);

	auto read = [&](float *dst, size_t n) { assert(n <= size); std::copy_n(data, n, dst); data += n; size -= n; };

	// Old prescreener data.
	{
		read(&nnedi3->prescreener_old().kernel_l0[0][0], 4 * 48);
		read(nnedi3->prescreener_old().bias_l0, 4);

		read(&nnedi3->prescreener_old().kernel_l1[0][0], 4 * 4);
		read(nnedi3->prescreener_old().bias_l1, 4);

		read(&nnedi3->prescreener_old().kernel_l2[0][0], 4 * 8);
		read(nnedi3->prescreener_old().bias_l2, 4);
	}

	// New prescreener data.
	for (unsigned i = 0; i < 3; ++i) {
		auto &data = nnedi3->prescreener_new(i);

		float kernel_l0_shuffled[4 * 64];
		float kernel_l1_shuffled[4 * 4];

		read(kernel_l0_shuffled, 4 * 64);
		read(nnedi3->prescreener_new(i).bias_l0, 4);

		read(kernel_l1_shuffled, 4 * 4);
		read(nnedi3->prescreener_new(i).bias_l1, 4);

		// Convert kernels back to row-major order.
		for (unsigned n = 0; n < 4; ++n) {
			for (unsigned k = 0; k < 64; ++k) {
				data.kernel_l0[n][k] = kernel_l0_shuffled[(k / 8) * 32 + n * 8 + k % 8];
			}
			for (unsigned k = 0; k < 4; ++k) {
				data.kernel_l1[n][k] = kernel_l1_shuffled[k * 4 + n];
			}
		}
	}

	// ABS model + MSE model.
	for (unsigned m = 0; m < 2; ++m) {
		auto &model_set = m ? nnedi3->mse_models() : nnedi3->abs_models();

		// Grouping by neuron count.
		for (unsigned i = 0; i < 5; ++i) {
			unsigned nns = NNEDI3_NNS[i];

			// Grouping by window size.
			for (unsigned j = 0; j < 7; ++j) {
				unsigned xdim = NNEDI3_XDIM[j];
				unsigned ydim = NNEDI3_YDIM[j];
				size_t filter_size = xdim * ydim;

				auto model = allocate_model({ xdim, ydim, nns });

				// Quality 1 model. NNS[i] * (XDIM[j] * YDIM[j]) * 2 coefficients.
				read(model.second.softmax_q1, nns * filter_size);
				read(model.second.elliott_q1, nns * filter_size);

				// Quality 1 model bias. NNS[i] * 2 coefficients.
				read(model.second.softmax_bias_q1, nns);
				read(model.second.elliott_bias_q1, nns);

				// Quality 2 model. NNS[i] * (XDIM[j] * YDIM[j]) * 2 coefficients.
				read(model.second.softmax_q2, nns * filter_size);
				read(model.second.elliott_q2, nns * filter_size);

				// Quality 2 model bias. NNS[i] * 2 coefficients.
				read(model.second.softmax_bias_q2, nns);
				read(model.second.elliott_bias_q2, nns);

				model_set.emplace(std::move(model));
			}
		}
	}
	assert(size == 0);

	return nnedi3;
}

} // namespace znedi3
