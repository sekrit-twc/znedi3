#ifdef ZNEDI3_ARM

#include <algorithm>
#include <cassert>
#include "alloc.h"
#include "kernel.h"
#include "kernel_arm.h"
#include "znedi3_impl.h"

namespace znedi3 {
namespace {

pixel_io_func select_pixel_io_func_neon(PixelType in, PixelType out)
{
	if (in == PixelType::BYTE && out == PixelType::FLOAT)
		return byte_to_float_sse2;
	else if (in == PixelType::WORD && out == PixelType::FLOAT)
		return word_to_float_sse2;
	else if (in == PixelType::FLOAT && out == PixelType::BYTE)
		return float_to_byte_sse2;
	else if (in == PixelType::FLOAT && out == PixelType::WORD)
		return float_to_word_sse2;
	else
		return nullptr;
}


} // namespace

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

pixel_io_func select_pixel_io_func_arm(PixelType in, PixelType out, CPUClass cpu)
{
	return select_pixel_io_func_neon(in, out);
}

interpolate_func select_interpolate_func_arm(CPUClass cpu)
{
	return cubic_interpolation_sse2;
}

std::unique_ptr<Prescreener> create_prescreener_old_arm(const PrescreenerOldCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	return create_prescreener_old_sse(coeffs, pixel_half);
}

std::unique_ptr<Prescreener> create_prescreener_new_arm(const PrescreenerNewCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	return create_prescreener_new_sse(coeffs, pixel_half);
}

std::unique_ptr<Predictor> create_predictor_arm(const PredictorModel &model, bool use_q2, CPUClass cpu)
{
	return create_predictor_sse2(model, use_q2);
}

} // namespace znedi3

#endif // ZNEDI3_ARM
