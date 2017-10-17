#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include "align.h"
#include "alloc.h"
#include "cpuinfo.h"
#include "weights.h"
#include "znedi3_impl.h"

namespace znedi3 {
namespace {

CPUClass translate_cpu_type(znedi3_cpu_type_e e)
{
	switch (e) {
	case ZNEDI3_CPU_NONE:
		return CPUClass::NONE;
	case ZNEDI3_CPU_AUTO:
		return CPUClass::AUTO;
	case ZNEDI3_CPU_AUTO_64B:
		return CPUClass::AUTO_64B;
#if defined(__i386) || defined(_M_IX86) || defined(_M_X64) || defined(__x86_64__)
	case ZNEDI3_CPU_X86_MMX:
		return CPUClass::NONE;
	case ZNEDI3_CPU_X86_SSE:
		return CPUClass::X86_SSE;
	case ZNEDI3_CPU_X86_SSE2:
	case ZNEDI3_CPU_X86_SSE3:
	case ZNEDI3_CPU_X86_SSSE3:
	case ZNEDI3_CPU_X86_SSE41:
	case ZNEDI3_CPU_X86_SSE42:
		return CPUClass::X86_SSE2;
	case ZNEDI3_CPU_X86_AVX:
		return CPUClass::X86_AVX;
	case ZNEDI3_CPU_X86_F16C:
		return CPUClass::X86_F16C;
	case ZNEDI3_CPU_X86_AVX2:
	case ZNEDI3_CPU_X86_AVX512F:
		return CPUClass::X86_AVX2;
	case ZNEDI3_CPU_X86_AVX512_SKL:
		return CPUClass::X86_AVX512;
#endif
	default:
		throw std::domain_error{ "bad cpu" };
	}
}

PixelType translate_pixel_type(znedi3_pixel_type_e e)
{
	switch (e) {
	case ZNEDI_PIXEL_BYTE:
		return PixelType::BYTE;
	case ZNEDI_PIXEL_WORD:
		return PixelType::WORD;
	case ZNEDI_PIXEL_HALF:
		return PixelType::HALF;
	case ZNEDI_PIXEL_FLOAT:
		return PixelType::FLOAT;
	default:
		throw std::domain_error{ "bad pixel_type" };
	}
}

void binary_prescreen_mask(const void *, ptrdiff_t, void *dst, const unsigned char *prescreen, unsigned n)
{
	float *dst_p = static_cast<float *>(dst);

	for (unsigned i = 0; i < n; ++i) {
		dst_p[i] = prescreen[i] ? 0.0f : 65535.0f;
	}
}

void show_prescreen_mask(const void *src, ptrdiff_t src_stride, void *dst, const unsigned char *prescreen, unsigned n)
{
	select_interpolate_func(CPUClass::NONE)(src, src_stride, dst, prescreen, n);

	float *dst_p = static_cast<float *>(dst);

	for (unsigned i = 0; i < n; ++i) {
		if (!prescreen[i])
			dst_p[i] = 65535.0f;
	}
}

} // namespace


znedi3_filter::znedi3_filter(const NNEDI3Weights &weights, const znedi3_filter_params &params) :
	m_pixel_load_func{},
	m_pixel_store_func{},
	m_interpolate_func{},
	m_type{ translate_pixel_type(params.pixel_type) },
	m_cpu{ translate_cpu_type(params.cpu) }
{
	unsigned bit_depth;

	switch (m_type) {
	case PixelType::BYTE:
		if ((bit_depth = params.bit_depth ? params.bit_depth : 8) > 8)
			throw std::domain_error{ "bad bit_depth value" };
		break;
	case PixelType::WORD:
		if ((bit_depth = params.bit_depth ? params.bit_depth : 16) > 16)
			throw std::domain_error{ "bad bit_depth value" };
		break;
	default:
		bit_depth = 0;
	}

	if (params.nns < 0 || params.nns > ZNEDI3_NNS_256)
		throw std::domain_error{ "bad nns value" };
	if (params.nsize < 0 || params.nsize > ZNEDI3_NSIZE_32x4)
		throw std::domain_error{ "bad nsize value" };
	if (params.qual != ZNEDI3_QUAL_1 && params.qual != ZNEDI3_QUAL_2)
		throw std::domain_error{ "bad qual value" };
	if (params.etype != ZNEDI3_ETYPE_ABS && params.etype != ZNEDI3_ETYPE_MSE)
		throw std::domain_error{ "bad etype value" };
	if (params.prescreen < 0 || params.prescreen > ZNEDI3_PRESCREEN_NEW_L2)
		throw std::domain_error{ "bad prescreen value" };

	PredictorTraits traits{ NNEDI3_XDIM[params.nsize], NNEDI3_YDIM[params.nsize], NNEDI3_NNS[params.nns] };
	const PredictorModelSet &model_set = (params.etype == ZNEDI3_ETYPE_MSE) ? weights.mse_models() : weights.abs_models();
	assert(model_set.find(traits) != model_set.end());
	const PredictorModel &model = *model_set.find(traits);

	double pixel_half_val = static_cast<double>((1UL << bit_depth) - 1) / 2.0;

	switch (params.prescreen) {
	case ZNEDI3_PRESCREEN_NONE:
		break;
	case ZNEDI3_PRESCREEN_OLD:
		m_prescreener = create_prescreener_old(weights.prescreener_old(), pixel_half_val, m_cpu);
		break;
	case ZNEDI3_PRESCREEN_NEW_L0:
	case ZNEDI3_PRESCREEN_NEW_L1:
	case ZNEDI3_PRESCREEN_NEW_L2:
		m_prescreener = create_prescreener_new(weights.prescreener_new(params.prescreen - ZNEDI3_PRESCREEN_NEW_L0), pixel_half_val, m_cpu);
		break;
	default:
		assert(false);
	}

	if (params.show_mask >= 2) {
		m_interpolate_func = binary_prescreen_mask;
	} else if (params.show_mask) {
		m_interpolate_func = show_prescreen_mask;
	} else {
		m_predictor = create_predictor(model, params.qual >= ZNEDI3_QUAL_2, m_cpu);
		m_interpolate_func = select_interpolate_func(m_cpu);
	}

	if (!(m_pixel_load_func = select_pixel_io_func(m_type, PixelType::FLOAT, m_cpu)))
		throw std::runtime_error{ "not implemented" };
	if (!(m_pixel_store_func = select_pixel_io_func(PixelType::FLOAT, m_type, m_cpu)))
		throw std::runtime_error{ "not implemented" };
}

size_t znedi3_filter::get_tmp_size(unsigned width, unsigned height) const
{
	return 0;
}

void znedi3_filter::process(unsigned width, unsigned height, const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride, void *tmp, unsigned parity) const
{
	assert(static_cast<uintmax_t>(width) < static_cast<uintmax_t>(PTRDIFF_MAX));
	assert(static_cast<uintmax_t>(height) < static_cast<uintmax_t>(PTRDIFF_MAX));

	ptrdiff_t width_d = width;
	ptrdiff_t height_d = height;

	assert(width_d < PTRDIFF_MAX - 48);
	assert(height_d < PTRDIFF_MAX - 6);

	// Create padded image.
	AlignedVector<float> pad;

	ptrdiff_t pad_stride = ceil_n((width + 64) * sizeof(float), ALIGNMENT);
	ptrdiff_t pad_stride_f = pad_stride / sizeof(float);
	pad.resize(pad_stride_f * (height + 6));

	float *pad_p = pad.data() + 3 * pad_stride_f + 32;

	for (ptrdiff_t i = 0; i < height_d; ++i) {
		m_pixel_load_func(static_cast<const unsigned char *>(src) + i * src_stride, pad_p + i * pad_stride_f, width);

		for (ptrdiff_t j = -32; j < 0; ++j) {
			pad_p[i * pad_stride_f + j] = pad_p[i * pad_stride_f];
		}
		for (ptrdiff_t j = width_d; j < width_d + 32; ++j) {
			pad_p[i * pad_stride_f + j] = pad_p[i * pad_stride_f + (width_d - 1)];
		}
	}
	for (ptrdiff_t i = -3; i < 0; ++i) {
		std::copy_n(pad_p - 32, width_d + 64, pad_p + i * pad_stride_f - 32);
	}
	for (ptrdiff_t i = height_d; i < height_d + 3; ++i) {
		std::copy_n(pad_p + (height_d - 1) * pad_stride_f - 32, width_d + 64, pad_p + i * pad_stride_f - 32);
	}

	// Create temporary destination image.
	AlignedVector<float> dst_tmp;
	ptrdiff_t dst_tmp_stride_f = ceil_n(width_d, ALIGNMENT / sizeof(float));

	dst_tmp.resize(dst_tmp_stride_f * height_d);
	float *dst_tmp_p = dst_tmp.data();

	// Initialize pointers. Set the source pointer to the row below the current output row.
	const float *src_p = pad_p + (parity ? pad_stride_f : 0);
	float *dst_p = dst_tmp_p;
	ptrdiff_t src_p_stride_f = pad_stride_f;
	ptrdiff_t dst_p_stride_f = dst_tmp_stride_f;

	// Main loop.
	AlignedVector<unsigned char> prescreen(ceil_n(width_d + 4, ALIGNMENT));

	for (ptrdiff_t i = 0; i < height_d; ++i) {
		if (m_prescreener)
			m_prescreener->process(src_p, src_p_stride_f * sizeof(float), prescreen.data(), width);
		if (m_predictor)
			m_predictor->process(src_p, src_p_stride_f * sizeof(float), dst_p, prescreen.data(), width);
		if (m_prescreener)
			m_interpolate_func(src_p, src_p_stride_f * sizeof(float), dst_p, prescreen.data(), width);

		src_p += src_p_stride_f;
		dst_p += dst_p_stride_f;
	}

	// Copy temporary image to output.
	for (ptrdiff_t i = 0; i < height_d; ++i) {
		m_pixel_store_func(dst_tmp_p + i * dst_tmp_stride_f, static_cast<unsigned char *>(dst) + i * dst_stride, width);
	}
}

} // namespace znedi3
