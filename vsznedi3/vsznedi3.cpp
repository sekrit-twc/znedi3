#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include <znedi3.h>
#include "vsxx_pluginmain.h"

using namespace vsxx;

namespace {

constexpr char PLUGIN_ID[] = "xxx.abc.znedi3";

struct ZNEDI3WeightsFree {
	void operator()(znedi3_weights *ptr) { znedi3_weights_free(ptr); }
};

struct ZNEDI3FilterFree {
	void operator()(znedi3_filter *ptr) { znedi3_filter_free(ptr); }
};

enum class FieldOperation {
	KEEP_BOTTOM = 0,
	KEEP_TOP = 1,
	BOB_BOTTOM_FIRST = 2,
	BOB_TOP_FIRST = 3,
};

void override_cpu_type(znedi3_cpu_type_e &dst, const std::string &str)
{
#if defined(__i386) || defined(_M_IX86) || defined(_M_X64) || defined(__x86_64__)
	if (str == "mmx")
		dst = ZNEDI3_CPU_X86_MMX;
	else if (str == "sse")
		dst = ZNEDI3_CPU_X86_SSE;
	else if (str == "sse2")
		dst = ZNEDI3_CPU_X86_SSE2;
	else if (str == "sse3")
		dst = ZNEDI3_CPU_X86_SSE3;
	else if (str == "ssse3")
		dst = ZNEDI3_CPU_X86_SSSE3;
	else if (str == "sse41")
		dst = ZNEDI3_CPU_X86_SSE41;
	else if (str == "sse42")
		dst = ZNEDI3_CPU_X86_SSE42;
	else if (str == "avx")
		dst = ZNEDI3_CPU_X86_AVX;
	else if (str == "f16c")
		dst = ZNEDI3_CPU_X86_F16C;
	else if (str == "avx2")
		dst = ZNEDI3_CPU_X86_AVX2;
	else if (str == "avx512f")
		dst = ZNEDI3_CPU_X86_AVX512F;
	else if (str == "avx512_skl")
		dst = ZNEDI3_CPU_X86_AVX512_SKL;
#endif
}

} // namespace


class VSZNEDI3 : public vsxx::FilterBase {
	std::unique_ptr<znedi3_filter, ZNEDI3FilterFree> m_nnedi3;
	std::vector<std::pair<size_t, VideoFrame>> m_cache;
	std::mutex m_cache_mutex;
	FilterNode m_clip;
	VSVideoInfo m_vi;
	FieldOperation m_mode;
	bool m_dh;
	bool m_planes[3];

	int get_src_frameno(int n) const
	{
		if ((m_mode == FieldOperation::BOB_BOTTOM_FIRST || m_mode == FieldOperation::BOB_TOP_FIRST) && !m_dh)
			return n / 2;
		else
			return n;
	}

	unsigned get_src_parity(const ConstVideoFrame &src, int n) const
	{
		const ConstPropertyMap &props = src.frame_props_ro();
		unsigned default_parity = (m_mode == FieldOperation::KEEP_BOTTOM || m_mode == FieldOperation::BOB_BOTTOM_FIRST) ? 1 : 0;

		if (m_dh) {
			unsigned parity = props.get_prop<unsigned>("_Field", map::default_val(default_parity));
			return parity;
		} else if (m_mode == FieldOperation::BOB_BOTTOM_FIRST || m_mode == FieldOperation::BOB_TOP_FIRST) {
			unsigned field_based = props.get_prop<unsigned>("_FieldBased", map::default_val(0U));
			unsigned parity = field_based ? !(field_based - 1) : default_parity;

			return n % 2 ? !parity : parity;
		} else {
			return m_mode == FieldOperation::KEEP_BOTTOM ? 1 : 0;
		}
	}
public:
	explicit VSZNEDI3(void *) :
		m_vi{},
		m_mode{},
		m_dh{},
		m_planes{ true, true, true }
	{}

	const char *get_name(int) noexcept override { return "znedi3"; }

	std::pair<::VSFilterMode, int> init(const ConstPropertyMap &in, const PropertyMap &out, const VapourCore &core) override
	{
		Plugin this_plugin{ get_vsapi()->getPluginById(PLUGIN_ID, core.get()) };
		std::string plugin_path = this_plugin.path();

		std::string weights_path;
#ifdef NNEDI3_WEIGHTS_PATH
		weights_path = NNEDI3_WEIGHTS_PATH;
#else
		weights_path = plugin_path.substr(0, plugin_path.find_last_of('/')) + "/nnedi3_weights.bin";
#endif
		if (in.contains("x_nnedi3_weights_bin"))
			weights_path = in.get_prop<std::string>("x_nnedi3_weights_bin");

		std::unique_ptr<znedi3_weights, ZNEDI3WeightsFree> nnedi3_weights{ znedi3_weights_from_file(weights_path.c_str()) };
		if (!nnedi3_weights)
			throw std::runtime_error{ "error reading weights" };

		m_clip = in.get_prop<FilterNode>("clip");
		m_vi = m_clip.video_info();

		if (!m_vi.format)
			throw std::runtime_error{ "clip must have a constant format" };

		m_mode = static_cast<FieldOperation>(in.get_prop<int>("field"));
		if (m_mode < FieldOperation::KEEP_BOTTOM || m_mode > FieldOperation::BOB_TOP_FIRST)
			throw std::runtime_error{ "bad field operation" };

		m_dh = in.get_prop<bool>("dh", map::default_val(false));
		if (m_dh)
			m_vi.height *= 2;

		if ((m_mode == FieldOperation::BOB_BOTTOM_FIRST || m_mode == FieldOperation::BOB_TOP_FIRST) && !m_dh) {
			m_vi.numFrames = int64ToIntS(m_vi.numFrames * 2LL);
			m_vi.fpsNum *= 2;
			muldivRational(&m_vi.fpsNum, &m_vi.fpsDen, 1, 1);
		}

		if (in.contains("planes")) {
			size_t n = in.num_elements("planes");
			std::fill_n(m_planes, 3, false);

			for (size_t i = 0; i < n; ++i) {
				unsigned p = in.get_prop<unsigned>("planes", static_cast<int>(i));
				if (p < 3)
					m_planes[p] = true;
			}
		}

		auto propagate_if_set = [&](auto &dst, int val)
		{
			if (val >= 0)
				dst = static_cast<std::remove_reference_t<decltype(dst)>>(val);
		};

		znedi3_filter_params nnedi3_params;
		znedi3_filter_params_default(&nnedi3_params);
		nnedi3_params.cpu = ZNEDI3_CPU_AUTO_64B;

		if (m_vi.format->sampleType == stInteger && m_vi.format->bytesPerSample == 1)
			nnedi3_params.pixel_type = ZNEDI3_PIXEL_BYTE;
		else if (m_vi.format->sampleType == stInteger && m_vi.format->bytesPerSample == 2)
			nnedi3_params.pixel_type = ZNEDI3_PIXEL_WORD;
		else if (m_vi.format->sampleType == stFloat && m_vi.format->bytesPerSample == 2)
			nnedi3_params.pixel_type = ZNEDI3_PIXEL_HALF;
		else if (m_vi.format->sampleType == stFloat && m_vi.format->bytesPerSample == 4)
			nnedi3_params.pixel_type = ZNEDI3_PIXEL_FLOAT;
		else
			throw std::runtime_error{ "unsupported format" };

		nnedi3_params.bit_depth = m_vi.format->bitsPerSample;

		propagate_if_set(nnedi3_params.nsize, in.get_prop<int>("nsize", map::default_val(-1)));
		propagate_if_set(nnedi3_params.nns, in.get_prop<int>("nns", map::default_val(-1)));
		propagate_if_set(nnedi3_params.qual, in.get_prop<int>("qual", map::default_val(-1)));
		propagate_if_set(nnedi3_params.etype, in.get_prop<int>("etype", map::default_val(-1)));
		propagate_if_set(nnedi3_params.prescreen, in.get_prop<int>("pscrn", map::default_val(-1)));

		if (in.contains("opt") && !in.get_prop<bool>("opt"))
			nnedi3_params.cpu = ZNEDI3_CPU_NONE;
		if (in.contains("int16_prescreener"))
			nnedi3_params.int16_prescreen = in.get_prop<bool>("int16_prescreener");
		if (in.contains("int16_predictor"))
			nnedi3_params.int16_predict = in.get_prop<bool>("int16_predictor");

		if (in.contains("x_cpu"))
			override_cpu_type(nnedi3_params.cpu, in.get_prop<std::string>("x_cpu"));

		propagate_if_set(nnedi3_params.slow_exp, in.get_prop<int>("exp", map::default_val(-1)));
		propagate_if_set(nnedi3_params.show_mask, in.get_prop<int>("show_mask", map::default_val(-1)));

		m_nnedi3.reset(znedi3_filter_create(nnedi3_weights.get(), &nnedi3_params));
		if (!m_nnedi3)
			throw std::runtime_error{ "failed to create nnedi3" };

		return{ fmParallel, 0 };
	}

	std::pair<const ::VSVideoInfo *, size_t> get_video_info() noexcept override
	{
		return{ &m_vi, 1 };
	}

	ConstVideoFrame get_frame_initial(int n, const VapourCore &, ::VSFrameContext *frame_ctx) override
	{
		m_clip.request_frame_filter(get_src_frameno(n), frame_ctx);
		return ConstVideoFrame{};
	}

	ConstVideoFrame get_frame(int n, const VapourCore &core, ::VSFrameContext *frame_ctx) override
	{
		ConstVideoFrame src_frame = m_clip.get_frame_filter(get_src_frameno(n), frame_ctx);
		VideoFrame dst_frame = core.new_video_frame(src_frame.format(), src_frame.width(0), src_frame.height(0) * (m_dh ? 2 : 1), src_frame);

		unsigned src_parity = get_src_parity(src_frame, n);

		std::pair<size_t, VideoFrame> tmp_buffer{};
		void *tmp = nullptr;

		for (int p = 0; p < src_frame.format().numPlanes; ++p) {
			if (!m_planes[p])
				continue;

			unsigned width = src_frame.width(p);
			unsigned height = src_frame.height(p);
			size_t rowsize = static_cast<size_t>(width) * src_frame.format().bytesPerSample;

			if (!m_dh)
				height /= 2;

			const uint8_t *src_p = src_frame.read_ptr(p);
			int src_stride = src_frame.stride(p);

			uint8_t *dst_p = dst_frame.write_ptr(p);
			int dst_stride = dst_frame.stride(p);

			const uint8_t *src_field_p = src_p + (m_dh ? 0 : static_cast<int>(src_parity) * src_stride);
			int src_field_stride = src_stride * (m_dh ? 1 : 2);

			uint8_t *dst_field_p = dst_p + static_cast<int>(!src_parity) * dst_stride;
			int dst_field_stride = dst_stride * 2;

			size_t tmp_size = znedi3_filter_get_tmp_size(m_nnedi3.get(), width, height);
			if (tmp_size > tmp_buffer.first) {
				std::pair<size_t, VideoFrame> buffer{};

				{
					std::lock_guard<std::mutex> lock{ m_cache_mutex };

					if (!m_cache.empty()) {
						buffer = std::move(m_cache.back());
						m_cache.pop_back();
					}
				}

				if (buffer.first < tmp_size) {
					buffer.first = tmp_size;
					buffer.second = core.new_video_frame(*core.format_preset(pfGray8), static_cast<int>(tmp_size), 1);
				}

				tmp_buffer = std::move(buffer);
				tmp = tmp_buffer.second.write_ptr(0);
			}

			znedi3_filter_process(m_nnedi3.get(), width, height, src_field_p, src_field_stride, dst_field_p, dst_field_stride, tmp, !src_parity);

			uint8_t *dst_other_field_p = dst_p + static_cast<int>(src_parity) * dst_stride;
			vs_bitblt(dst_other_field_p, dst_field_stride, src_field_p, src_field_stride, rowsize, height);
		}

		if (tmp_buffer.first) {
			std::lock_guard<std::mutex> lock{ m_cache_mutex };
			m_cache.emplace_back(std::move(tmp_buffer));
		}

		PropertyMapRef dst_props = dst_frame.frame_props();
		dst_props.set_prop("_FieldBased", 0, paReplace);
		dst_props.erase("_Field");

		return dst_frame;
	}
};

const PluginInfo g_plugin_info{
	PLUGIN_ID, "znedi3", "Neural network edge directed interpolation (3rd gen.)",
	{
		{ vsxx::FilterBase::filter_create<VSZNEDI3>, "nnedi3",
			"clip:clip;"
			"field:int;"
			"dh:int:opt;"
			"planes:int[]:opt;"
			"nsize:int:opt;"
			"nns:int:opt;"
			"qual:int:opt;"
			"etype:int:opt;"
			"pscrn:int:opt;"
			"opt:int:opt;"
			"int16_prescreener:int:opt;"
			"int16_predictor:int:opt;"
			"exp:int:opt;"
			"show_mask:int:opt;"
			"x_nnedi3_weights_bin:data:opt;"
			"x_cpu:data:opt;"
		}
	}
};
