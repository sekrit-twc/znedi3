#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <utility>
#include "graphengine/filter.h"
#include "graphengine/graph.h"
#include "graphengine/types.h"
#include "align.h"
#include "alloc.h"
#include "cpuinfo.h"
#include "weights.h"
#include "znedi3_impl.h"

namespace znedi3 {
namespace {

using graphengine::node_id;

CPUClass translate_cpu_type(znedi3_cpu_type_e e)
{
	switch (e) {
	case ZNEDI3_CPU_NONE:
		return CPUClass::NONE;
	case ZNEDI3_CPU_AUTO:
		return CPUClass::AUTO;
	case ZNEDI3_CPU_AUTO_64B:
		return CPUClass::AUTO_64B;
#ifdef ZNEDI3_X86
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
		return CPUClass::X86_AVX2;
	case ZNEDI3_CPU_X86_AVX512F:
	case ZNEDI3_CPU_X86_AVX512_SKX:
	case ZNEDI3_CPU_X86_AVX512_CLX:
	case ZNEDI3_CPU_X86_AVX512_PMC:
	case ZNEDI3_CPU_X86_AVX512_SNC:
	case ZNEDI3_CPU_X86_AVX512_WLC:
	case ZNEDI3_CPU_X86_AVX512_GLC:
		return CPUClass::X86_AVX512;
#endif
	default:
		throw std::domain_error{ "bad cpu" };
	}
}

PixelType translate_pixel_type(znedi3_pixel_type_e e)
{
	switch (e) {
	case ZNEDI3_PIXEL_BYTE:
		return PixelType::BYTE;
	case ZNEDI3_PIXEL_WORD:
		return PixelType::WORD;
	case ZNEDI3_PIXEL_HALF:
		return PixelType::HALF;
	case ZNEDI3_PIXEL_FLOAT:
		return PixelType::FLOAT;
	default:
		throw std::domain_error{ "bad pixel_type" };
	}
}

unsigned pixel_size(PixelType type)
{
	switch (type) {
	case PixelType::BYTE:
		return sizeof(uint8_t);
	case PixelType::WORD:
	case PixelType::HALF:
		return sizeof(uint16_t);
	case PixelType::FLOAT:
		return sizeof(float);
	default:
		return 0;
	}
}

void binary_prescreen_mask(const float * const *, float *dst, const unsigned char *prescreen, unsigned n)
{
	for (unsigned i = 0; i < n; ++i) {
		dst[i] = prescreen[i] ? 0.0f : 65535.0f;
	}
}

void show_prescreen_mask(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n)
{
	select_interpolate_func(CPUClass::NONE)(src, dst, prescreen, n);

	for (unsigned i = 0; i < n; ++i) {
		if (!prescreen[i])
			dst[i] = 65535.0f;
	}
}


constexpr unsigned PADDING_H = 32;
constexpr unsigned PADDING_V = 3;

unsigned padded_width(unsigned width) { return width + PADDING_H * 2; }
unsigned padded_height(unsigned height) { return height + PADDING_V * 2; }


class PadFilter final : public graphengine::Filter {
	graphengine::FilterDescriptor m_desc{};
	pixel_io_func m_func;
	unsigned m_orig_width;
	unsigned m_orig_height;
	PixelType m_src_type;
	unsigned m_parity;
public:
	PadFilter(unsigned width, unsigned height, PixelType src_type, bool parity, CPUClass cpu) :
		m_func{ select_pixel_io_func(src_type, PixelType::FLOAT, cpu) },
		m_orig_width{ width },
		m_orig_height{ height },
		m_src_type{ src_type },
		m_parity{ !!parity }
	{
		if (!m_func)
			throw std::runtime_error{ "not implemented pixel type" };

		m_desc.format = { padded_width(width), padded_height(height), sizeof(float) };
		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.step = 1;
		m_desc.alignment_mask = 63;
	}

	int version() const noexcept override { return VERSION; }

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	pair_unsigned get_row_deps(unsigned i) const noexcept override
	{
		i = std::max(i, PADDING_V - m_parity) - (PADDING_V - m_parity);
		i = std::min(i, m_orig_height - 1);
		return{ i, i + 1 };
	}

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override
	{
		left = std::max(left, PADDING_H) - PADDING_H;
		right = std::min(std::max(right, PADDING_H) - PADDING_H, m_orig_width);
		return{ left, right };
	}

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out, unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		unsigned src_row = get_row_deps(i).first;
		const uint8_t *srcp = in->get_line<uint8_t>(src_row);
		float *dstp = out->get_line<float>(i);

		// Padding is 32 and alignment is 64.
		unsigned left_span = std::max(left, PADDING_H);
		unsigned right_span = std::min(right, m_orig_width + PADDING_H);
		m_func(srcp + static_cast<size_t>(left_span - PADDING_H) * pixel_size(m_src_type), dstp + left_span, right_span - left_span);

		if (left < PADDING_H) {
			float val = dstp[PADDING_H];
			std::fill_n(dstp, PADDING_H, val);
		}
		if (right >= m_orig_width + PADDING_H) {
			float val = dstp[PADDING_H];
			std::fill_n(dstp + m_orig_width + PADDING_H, PADDING_H, val);
		}
	}
};

class StoreFilter final : public graphengine::Filter {
	graphengine::FilterDescriptor m_desc{};
	pixel_io_func m_func;
public:
	StoreFilter(unsigned width, unsigned height, PixelType dst_type, CPUClass cpu) :
		m_func{ select_pixel_io_func(PixelType::FLOAT, dst_type, cpu) }
	{
		if (!m_func)
			throw std::runtime_error{ "not implemented pixel type" };

		m_desc.format = { width, height, pixel_size(dst_type) };
		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.step = 1;
		m_desc.alignment_mask = 15;
	}

	int version() const noexcept override { return VERSION; }

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	pair_unsigned get_row_deps(unsigned i) const noexcept override
	{
		return{ i + PADDING_V, i + PADDING_V + 1 };
	}

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override
	{
		return{ left + PADDING_H, right + PADDING_H };
	}

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out, unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		unsigned input_row = get_row_deps(i).first;
		size_t input_left = get_col_deps(left, right).first;

		const float *srcp = in[0].get_line<float>(input_row);
		uint8_t *dstp = out->get_line<uint8_t>(i);

		m_func(srcp + input_left, dstp + static_cast<size_t>(left) * m_desc.format.bytes_per_sample, right - left);
	}
};

class PrescreenFilter final : public graphengine::Filter {
	graphengine::FilterDescriptor m_desc{};
	std::unique_ptr<Prescreener> m_prescreener;
public:
	PrescreenFilter(unsigned width, unsigned height, std::unique_ptr<Prescreener> prescreener) :
		m_prescreener{ std::move(prescreener) }
	{
		m_desc.format = { padded_width(width), padded_height(height), sizeof(uint8_t) };
		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.step = 1;
		m_desc.alignment_mask = 15;

		m_desc.scratchpad_size = m_prescreener->get_tmp_size();
	}

	int version() const noexcept override { return VERSION; }

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	pair_unsigned get_row_deps(unsigned i) const noexcept override
	{
		i = std::max(i, PADDING_V);

		// 2 rows above, 2 rows below.
		return{ i - 2, std::min(i + 2, m_desc.format.height) };
	}

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override
	{
		left = std::max(left, PADDING_H);
		right = std::min(right, m_desc.format.width - PADDING_H);

		// Old: 5 before, 6 after. Window=12.
		// New: 6 before, 9 after. Window=16.
		return{ left - 6, right + 10 };
	}

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out, unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		if (i < PADDING_V || i >= m_desc.format.height - PADDING_V)
			return;

		left = std::max(left, PADDING_H);
		right = std::min(std::max(right, left), m_desc.format.width - PADDING_H);

		const float *srcp[4] = {
			in->get_line<float>(i - 2) + left,
			in->get_line<float>(i - 1) + left,
			in->get_line<float>(i + 0) + left,
			in->get_line<float>(i + 1) + left,
		};
		uint8_t *dstp = out->get_line<uint8_t>(i) + left;
		m_prescreener->process(srcp, dstp, tmp, right - left);
	}
};

class PredictFilter final : public graphengine::Filter {
	graphengine::FilterDescriptor m_desc{};
	std::unique_ptr<Predictor> m_predictor;
	bool m_prescreen;
public:
	PredictFilter(unsigned width, unsigned height, std::unique_ptr<Predictor> predictor, bool prescreener) :
		m_predictor{ std::move(predictor) },
		m_prescreen{ prescreener }
	{
		m_desc.format = { padded_width(width), padded_height(height), sizeof(float) };
		m_desc.num_deps = prescreener ? 2 : 1;
		m_desc.num_planes = 1;
		m_desc.step = 1;

		m_desc.context_size = prescreener ? 0 : width;
		m_desc.scratchpad_size = m_predictor->get_tmp_size();
	}

	int version() const noexcept override { return VERSION; }

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	pair_unsigned get_row_deps(unsigned i) const noexcept override
	{
		i = std::max(i, PADDING_V);

		// 3 rows above, 3 rows below.
		return{ i - 3, std::min(i + 3, m_desc.format.height) };
	}

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override
	{
		left = std::max(left, PADDING_H);
		right = std::min(right, m_desc.format.width - PADDING_H);

		// Max model: 23 before, 24 after. Window=48.
		return{ left - 23, right + 25 };
	}

	void init_context(void *context) const noexcept override
	{
		if (!m_prescreen)
			std::memset(static_cast<uint8_t *>(context) + PADDING_H, 1, m_desc.format.width - PADDING_H * 2 );
	}

	void process(const graphengine::BufferDescriptor in[], const graphengine::BufferDescriptor *out, unsigned i, unsigned left, unsigned right, void *context, void *tmp) const noexcept override
	{
		if (i < PADDING_V || i >= m_desc.format.height - PADDING_V)
			return;

		left = std::max(left, PADDING_H);
		right = std::min(std::max(right, left), m_desc.format.width - PADDING_H);

		const float *srcp[6] = {
			in->get_line<float>(i - 3) + left,
			in->get_line<float>(i - 2) + left,
			in->get_line<float>(i - 1) + left,
			in->get_line<float>(i + 0) + left,
			in->get_line<float>(i + 1) + left,
			in->get_line<float>(i + 2) + left,
		};

		const uint8_t *pscrn = (m_prescreen ? in[1].get_line<uint8_t>(i) : static_cast<uint8_t *>(context)) + left;
		float *dstp = out->get_line<float>(i) + left;

		// Kernel.
		m_predictor->process(srcp, dstp, pscrn, tmp, right - left);
	}
};

class InterpolationFilter final : public graphengine::Filter {
	graphengine::FilterDescriptor m_desc{};
	interpolate_func m_func;
public:
	InterpolationFilter(unsigned width, unsigned height, interpolate_func func) : m_func{ func }
	{
		m_desc.format = { padded_width(width), padded_height(height), sizeof(float) };
		m_desc.num_deps = 3;
		m_desc.num_planes = 1;
		m_desc.step = 1;
		m_desc.alignment_mask = 15;

		m_desc.inplace_hint.enabled = 1;
		m_desc.inplace_hint.preferred_index = 2;
		m_desc.inplace_hint.disallow_mask = 0x1;

		m_desc.flags.in_place = true;
	}

	int version() const noexcept override { return VERSION; }

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	pair_unsigned get_row_deps(unsigned i) const noexcept override
	{
		i = std::max(i, PADDING_V);

		// 2 rows above, 2 rows below.
		return{ i - 2, std::min(i + 2, m_desc.format.height) };
	}

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override { return{ left, right }; }

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor in[], const graphengine::BufferDescriptor *out, unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		left = std::max(left, PADDING_H);
		right = std::min(std::max(right, left), m_desc.format.width - PADDING_H);

		const float *srcp[4] = {
			in[0].get_line<float>(i - 2) + left,
			in[0].get_line<float>(i - 1) + left,
			in[0].get_line<float>(i + 0) + left,
			in[0].get_line<float>(i + 1) + left,
		};

		const uint8_t *pscrn = in[1].get_line<uint8_t>(i) + left;
		const float *predicted = in[2].get_line<float>(i) + left;
		float *dstp = out->get_line<float>(i) + left;

		if (dstp != predicted)
			std::memcpy(dstp, predicted, (right - left) * sizeof(float));

		m_func(srcp, dstp, pscrn, right - left);
	}
};

} // namespace


struct znedi3_filter::filters {
	std::unique_ptr<graphengine::Filter> pad_load_t;
	std::unique_ptr<graphengine::Filter> pad_load_b;
	std::unique_ptr<graphengine::Filter> prescreener;
	std::unique_ptr<graphengine::Filter> predictor;
	std::unique_ptr<graphengine::Filter> interpolator;
	std::unique_ptr<graphengine::Filter> store;
};

struct znedi3_filter::data {
	graphengine::GraphImpl graph;
	node_id src_node = graphengine::null_node;
	node_id dst_node = graphengine::null_node;
};

znedi3_filter::znedi3_filter(const NNEDI3Weights &weights, const znedi3_filter_params &params, unsigned width, unsigned height) try
{
	m_filters = setup_filters(weights, params, width, height);
	m_data_t = setup_graph(weights, params, width, height, false);
	m_data_b = setup_graph(weights, params, width, height, true);
} catch (const graphengine::Exception &e) {
	throw std::runtime_error{ e.msg };
}

znedi3_filter::~znedi3_filter() = default;

std::unique_ptr<znedi3_filter::filters> znedi3_filter::setup_filters(const NNEDI3Weights &weights, const znedi3_filter_params &params, unsigned width, unsigned height)
{
	PixelType type = translate_pixel_type(params.pixel_type);
	CPUClass cpu = translate_cpu_type(params.cpu);
	unsigned bit_depth;

	switch (type) {
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


	std::unique_ptr<Prescreener> prescreener;
	double pixel_half_val = (type == PixelType::HALF || type == PixelType::FLOAT) ? 0.5 : static_cast<double>((1UL << bit_depth) - 1) / 2.0;

	switch (params.prescreen) {
	case ZNEDI3_PRESCREEN_NONE:
		break;
	case ZNEDI3_PRESCREEN_OLD:
		prescreener = create_prescreener_old(weights.prescreener_old(), pixel_half_val, cpu);
		break;
	case ZNEDI3_PRESCREEN_NEW_L0:
	case ZNEDI3_PRESCREEN_NEW_L1:
	case ZNEDI3_PRESCREEN_NEW_L2:
		prescreener = create_prescreener_new(weights.prescreener_new(params.prescreen - ZNEDI3_PRESCREEN_NEW_L0), pixel_half_val, cpu);
		break;
	default:
		assert(false);
	}

	PredictorTraits traits{ NNEDI3_XDIM[params.nsize], NNEDI3_YDIM[params.nsize], NNEDI3_NNS[params.nns] };
	const PredictorModelSet &model_set = (params.etype == ZNEDI3_ETYPE_MSE) ? weights.mse_models() : weights.abs_models();
	assert(model_set.find(traits) != model_set.end());
	std::unique_ptr<Predictor> predictor = create_predictor(*model_set.find(traits), params.qual >= ZNEDI3_QUAL_2, cpu);

	interpolate_func interpolate = nullptr;
	if (params.show_mask >= 2)
		interpolate = binary_prescreen_mask;
	else if (params.show_mask)
		interpolate = show_prescreen_mask;
	else if (prescreener)
		interpolate = select_interpolate_func(cpu);


	std::unique_ptr<filters> f = std::make_unique<filters>();

	f->pad_load_t = std::make_unique<PadFilter>(width, height, type, false, cpu);
	f->pad_load_b = std::make_unique<PadFilter>(width, height, type, true, cpu);

	if (prescreener)
		f->prescreener = std::make_unique<PrescreenFilter>(width, height, std::move(prescreener));

	f->predictor = std::make_unique<PredictFilter>(width, height, std::move(predictor), !!f->prescreener);

	if (interpolate)
		f->interpolator = std::make_unique<InterpolationFilter>(width, height, interpolate);

	f->store = std::make_unique<StoreFilter>(width, height, type, cpu);

	return f;
}

std::unique_ptr<znedi3_filter::data> znedi3_filter::setup_graph(const NNEDI3Weights &weights, const znedi3_filter_params &params, unsigned width, unsigned height, bool parity)
{
	using graphengine::node_dep_desc;

	std::unique_ptr<data> d = std::make_unique<data>();

	graphengine::PlaneDescriptor src_desc{ width, height, pixel_size(translate_pixel_type(params.pixel_type)) };
	d->src_node = d->graph.add_source(1, &src_desc);

	node_dep_desc pad_dep{ d->src_node, 0 };
	node_id padded = d->graph.add_transform(parity ? m_filters->pad_load_b.get() : m_filters->pad_load_t.get(), &pad_dep);

	node_id merged = graphengine::null_node;

	if (m_filters->prescreener) {
		node_dep_desc prescreen_dep{ padded, 0 };
		node_id prescreened = d->graph.add_transform(m_filters->prescreener.get(), &prescreen_dep);

		node_dep_desc predict_dep[] = { { padded, 0 }, { prescreened, 0 } };
		node_id predicted = d->graph.add_transform(m_filters->predictor.get(), predict_dep);

		node_dep_desc interpolate_deps[] = { { padded, 0 }, { prescreened, 0 }, { predicted, 0 } };
		node_id interpolated = d->graph.add_transform(m_filters->interpolator.get(), interpolate_deps);

		merged = interpolated;
	} else {
		node_dep_desc predict_dep{ padded, 0 };
		node_id predicted = d->graph.add_transform(m_filters->predictor.get(), &predict_dep);

		merged = predicted;
	}

	node_dep_desc store_dep = { merged, 0 };
	node_id stored = d->graph.add_transform(m_filters->store.get(), &store_dep);

	node_dep_desc sink_dep = { stored, 0 };
	d->dst_node = d->graph.add_sink(1, &sink_dep);
	return d;
}

size_t znedi3_filter::get_tmp_size() const
{
	return std::max(m_data_t->graph.get_tmp_size(), m_data_b->graph.get_tmp_size());
}

void znedi3_filter::process(const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride, void *tmp, bool parity) const
{
	data *data = parity ? m_data_b.get() : m_data_t.get();

	graphengine::Graph::EndpointConfiguration endpoints{};
	endpoints[0].id = data->src_node;
	endpoints[0].buffer[0] = { const_cast<void *>(src), src_stride, graphengine::BUFFER_MAX };
	endpoints[1].id = data->dst_node;
	endpoints[1].buffer[0] = { dst, dst_stride, graphengine::BUFFER_MAX };

	data->graph.run(endpoints, tmp);
}

} // namespace znedi3
