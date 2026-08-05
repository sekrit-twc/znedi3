#include <algorithm>
#include <cassert>
#include "alloc.h"
#include "kernel_interleaved.h"
#include "weights.h"

namespace znedi3 {

InterleavedPredictorModel create_interleaved_predictor_model(const PredictorModel &model)
{
	assert(model.first.nns % 16 == 0);

	unsigned filter_size = model.first.xdim * model.first.ydim;
	unsigned nns = model.first.nns;

	PredictorModel m = copy_model(model);
	subtract_mean(m);

	InterleavedPredictorModel interleaved{};
	interleaved.data.resize(nns * filter_size * 4 + nns * 4);

	interleaved.xdim = m.first.xdim;
	interleaved.ydim = m.first.ydim;
	interleaved.nns = m.first.nns;

	LinearAllocator alloc{ interleaved.data.data() };
	interleaved.neurons_q1 = alloc.allocate_n<float>(nns * filter_size * 2);
	interleaved.bias_q1 = alloc.allocate_n<float>(nns * 2);
	interleaved.neurons_q2 = alloc.allocate_n<float>(nns * filter_size * 2);
	interleaved.bias_q2 = alloc.allocate_n<float>(nns * 2);
	assert(alloc.count() / sizeof(float) == interleaved.data.size());

	for (unsigned k = 0; k < filter_size; ++k) {
		for (unsigned nn = 0; nn < nns; ++nn) {
			interleaved.neurons_q1[k * nns * 2 + nn] = m.second.softmax_q1[nn * filter_size + k];
			interleaved.neurons_q1[k * nns * 2 + nn + nns] = m.second.elliott_q1[nn * filter_size + k];
		}
		for (unsigned nn = 0; nn < nns; ++nn) {
			interleaved.neurons_q2[k * nns * 2 + nn] = m.second.softmax_q2[nn * filter_size + k];
			interleaved.neurons_q2[k * nns * 2 + nn + nns] = m.second.elliott_q2[nn * filter_size + k];
		}
	}
	std::copy_n(m.second.softmax_bias_q1, nns, interleaved.bias_q1);
	std::copy_n(m.second.elliott_bias_q1, nns, interleaved.bias_q1 + nns);
	std::copy_n(m.second.softmax_bias_q2, nns, interleaved.bias_q2);
	std::copy_n(m.second.elliott_bias_q2, nns, interleaved.bias_q2 + nns);

	return interleaved;
}

} // namespace znedi3
