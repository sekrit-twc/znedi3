#ifdef ZNEDI3_X86

#include <algorithm>
#include <cassert>
#include "alloc.h"
#include "cpuinfo.h"
#include "cpuinfo_x86.h"
#include "kernel.h"
#include "kernel_x86.h"
#include "znedi3_impl.h"

namespace znedi3 {
namespace {

pixel_io_func select_pixel_io_func_sse2(PixelType in, PixelType out)
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

pixel_io_func select_pixel_io_func_f16c(PixelType in, PixelType out)
{
	if (in == PixelType::HALF && out == PixelType::FLOAT)
		return half_to_float_f16c;
	else if (in == PixelType::FLOAT && out == PixelType::HALF)
		return float_to_half_f16c;
	else
		return nullptr;
}

pixel_io_func select_pixel_io_func_avx2(PixelType in, PixelType out)
{
	if (in == PixelType::BYTE && out == PixelType::FLOAT)
		return byte_to_float_avx2;
	else if (in == PixelType::WORD && out == PixelType::FLOAT)
		return word_to_float_avx2;
	else if (in == PixelType::FLOAT && out == PixelType::BYTE)
		return float_to_byte_avx2;
	else if (in == PixelType::FLOAT && out == PixelType::WORD)
		return float_to_word_avx2;
	else
		return nullptr;
}

#ifdef ZNEDI3_X86_AVX512
pixel_io_func select_pixel_io_func_avx512f(PixelType in, PixelType out)
{
	if (in == PixelType::BYTE && out == PixelType::FLOAT)
		return byte_to_float_avx512f;
	else if (in == PixelType::WORD && out == PixelType::FLOAT)
		return word_to_float_avx512f;
	else if (in == PixelType::HALF && out == PixelType::FLOAT)
		return half_to_float_avx512f;
	else if (in == PixelType::FLOAT && out == PixelType::BYTE)
		return float_to_byte_avx512f;
	else if (in == PixelType::FLOAT && out == PixelType::WORD)
		return float_to_word_avx512f;
	else if (in == PixelType::FLOAT && out == PixelType::HALF)
		return float_to_half_avx512f;
	else
		return nullptr;
}
#endif // ZNEDI3_X86_AVX512

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

pixel_io_func select_pixel_io_func_x86(PixelType in, PixelType out, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	pixel_io_func ret = nullptr;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f)
			ret = select_pixel_io_func_avx512f(in, out);
		if (!ret && caps.avx2)
			ret = select_pixel_io_func_avx2(in, out);
		if (!ret && caps.avx && caps.f16c)
			ret = select_pixel_io_func_f16c(in, out);
		if (!ret && caps.sse2)
			ret = select_pixel_io_func_sse2(in, out);
#endif
	} else {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = select_pixel_io_func_avx512f(in, out);
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = select_pixel_io_func_avx2(in, out);
		if (!ret && cpu >= CPUClass::X86_F16C)
			ret = select_pixel_io_func_f16c(in, out);
		if (!ret && cpu >= CPUClass::X86_SSE2)
			ret = select_pixel_io_func_sse2(in, out);
#endif
	}

	return ret;
}

interpolate_func select_interpolate_func_x86(CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	interpolate_func ret = nullptr;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f)
			ret = cubic_interpolation_avx512f;
#endif
		if (!ret && caps.avx2 && caps.fma)
			ret = cubic_interpolation_avx2;
		if (!ret && caps.avx)
			ret = cubic_interpolation_avx;
		if (!ret && caps.sse2)
			ret = cubic_interpolation_sse2;
	} else {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = cubic_interpolation_avx512f;
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = cubic_interpolation_avx2;
		if (!ret && cpu >= CPUClass::X86_AVX)
			ret = cubic_interpolation_avx;
		if (!ret && cpu >= CPUClass::X86_SSE2)
			ret = cubic_interpolation_sse2;
#endif
	}

	return ret;
}

std::unique_ptr<Prescreener> create_prescreener_old_x86(const PrescreenerOldCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Prescreener> ret;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f)
			ret = create_prescreener_old_avx512f(coeffs, pixel_half);
#endif
		if (!ret && caps.avx2 && caps.fma)
			ret = create_prescreener_old_avx2(coeffs, pixel_half);
		if (!ret && caps.avx)
			ret = create_prescreener_old_avx(coeffs, pixel_half);
		if (!ret && caps.sse)
			ret = create_prescreener_old_sse(coeffs, pixel_half);
	} else {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = create_prescreener_old_avx512f(coeffs, pixel_half);
#endif
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_prescreener_old_avx2(coeffs, pixel_half);
		if (!ret && cpu >= CPUClass::X86_AVX)
			ret = create_prescreener_old_avx(coeffs, pixel_half);
		if (!ret && cpu >= CPUClass::X86_SSE)
			ret = create_prescreener_old_sse(coeffs, pixel_half);
	}

	return ret;
}

std::unique_ptr<Prescreener> create_prescreener_new_x86(const PrescreenerNewCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Prescreener> ret;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f)
			ret = create_prescreener_new_avx512f(coeffs, pixel_half);
#endif
		if (!ret && caps.avx2 && caps.fma)
			ret = create_prescreener_new_avx2(coeffs, pixel_half);
		if (!ret && caps.avx)
			ret = create_prescreener_new_avx(coeffs, pixel_half);
		if (!ret && caps.sse)
			ret = create_prescreener_new_sse(coeffs, pixel_half);
	} else {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = create_prescreener_new_avx512f(coeffs, pixel_half);
#endif
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_prescreener_new_avx2(coeffs, pixel_half);
		if (!ret && cpu >= CPUClass::X86_AVX)
			ret = create_prescreener_new_avx(coeffs, pixel_half);
		if (!ret && cpu >= CPUClass::X86_SSE)
			ret = create_prescreener_new_sse(coeffs, pixel_half);
	}

	return ret;
}

std::unique_ptr<Predictor> create_predictor_x86(const PredictorModel &model, bool use_q2, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Predictor> ret;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f)
			ret = create_predictor_avx512f(model, use_q2);
#endif
		if (!ret && caps.avx && caps.fma)
			ret = create_predictor_avx2(model, use_q2);
		if (!ret && caps.avx)
			ret = create_predictor_avx(model, use_q2);
		if (!ret && caps.sse2)
			ret = create_predictor_sse2(model, use_q2);
		if (!ret && caps.sse)
			ret = create_predictor_sse(model, use_q2);
	} else {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = create_predictor_avx512f(model, use_q2);
#endif
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_predictor_avx2(model, use_q2);
		if (!ret && cpu >= CPUClass::X86_AVX)
			ret = create_predictor_avx(model, use_q2);
		if (!ret && cpu >= CPUClass::X86_SSE2)
			ret = create_predictor_sse2(model, use_q2);
		if (!ret && cpu >= CPUClass::X86_SSE)
			ret = create_predictor_sse(model, use_q2);
	}

	return ret;
}

} // namespace znedi3

#endif // ZNEDI3_X86
