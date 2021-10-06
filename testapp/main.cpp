#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <znedi3.h>
#include "aligned_malloc.h"
#include "argparse.h"
#include "timer.h"
#include "win32_bitmap.h"

namespace {

template <class T>
struct AlignedAllocator {
	typedef T value_type;

	AlignedAllocator() = default;

	template <class U>
	AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

	T *allocate(size_t n) const
	{
		T *ptr = static_cast<T *>(aligned_malloc(n * sizeof(T), 64));

		if (!ptr)
			throw std::bad_alloc{};

		return ptr;
	}

	void deallocate(void *ptr, size_t) const noexcept
	{
		aligned_free(ptr);
	}

	bool operator==(const AlignedAllocator &) const noexcept { return true; }
	bool operator!=(const AlignedAllocator &) const noexcept { return false; }
};

struct FreeWeights {
	void operator()(znedi3_weights *ptr) const { znedi3_weights_free(ptr); }
};

struct FreeFilter {
	void operator()(znedi3_filter *ptr) const { znedi3_filter_free(ptr); }
};

struct PlanarImage {
	std::vector<unsigned char, AlignedAllocator<unsigned char>> data[3];
	ptrdiff_t stride[3];
	unsigned width[3];
	unsigned height[3];
};

void bitmap_to_planar(const WindowsBitmap &bmp, PlanarImage &planar)
{
	unsigned width = bmp.width();
	unsigned height = bmp.height();

	ptrdiff_t stride = width % 64 ? width - width % 64 + 64 : width;

	for (unsigned p = 0; p < 3; ++p) {
		planar.data[p].resize(stride * height);
		planar.stride[p] = stride;
		planar.width[p] = width;
		planar.height[p] = height;
	}

	for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(height); ++i) {
		const unsigned char *src_p = bmp.read_ptr() + i * bmp.stride();
		unsigned step = bmp.bit_count() / 8;

		for (unsigned j = 0; j < width; ++j) {
			planar.data[0][i * planar.stride[0] + j] = src_p[j * step + 2];
			planar.data[1][i * planar.stride[1] + j] = src_p[j * step + 1];
			planar.data[2][i * planar.stride[2] + j] = src_p[j * step + 0];
		}
	}
}

void planar_to_bitmap(const PlanarImage &planar, WindowsBitmap &bmp)
{
	assert(static_cast<unsigned>(bmp.width()) == planar.width[0]);
	assert(static_cast<unsigned>(bmp.height()) == planar.height[0]);
	assert(planar.width[0] == planar.width[1] && planar.width[0] == planar.width[2]);
	assert(planar.height[0] == planar.height[1] && planar.height[0] == planar.height[2]);

	for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(planar.height[0]); ++i) {
		unsigned char *dst_p = bmp.write_ptr() + i * bmp.stride();
		unsigned step = bmp.bit_count() / 8;

		for (unsigned j = 0; j < planar.width[0]; ++j) {
			dst_p[j * step + 0] = planar.data[2][i * planar.stride[2] + j];
			dst_p[j * step + 1] = planar.data[1][i * planar.stride[1] + j];
			dst_p[j * step + 2] = planar.data[0][i * planar.stride[0] + j];
		}
	}
}


int arg_decode_cpu(const ArgparseOption *, void *out, const char *param, int)
{
	int *cpu = static_cast<int *>(out);

#define ELSE_IF(str, e) else if (!std::strcmp(param, str)) do { *cpu = e; } while (0)

	if (!std::strcmp(param, "none"))
		*cpu = ZNEDI3_CPU_NONE;
	else if (!std::strcmp(param, "auto"))
		*cpu = ZNEDI3_CPU_AUTO_64B;
#if defined(__i386) || defined(_M_IX86) || defined(_M_X64) || defined(__x86_64__)
	ELSE_IF("sse",    ZNEDI3_CPU_X86_SSE);
	ELSE_IF("sse2",   ZNEDI3_CPU_X86_SSE2);
	ELSE_IF("sse3",   ZNEDI3_CPU_X86_SSE3);
	ELSE_IF("ssse3",  ZNEDI3_CPU_X86_SSSE3);
	ELSE_IF("sse41",  ZNEDI3_CPU_X86_SSE41);
	ELSE_IF("sse42",  ZNEDI3_CPU_X86_SSE42);
	ELSE_IF("avx",    ZNEDI3_CPU_X86_AVX);
	ELSE_IF("f16c",   ZNEDI3_CPU_X86_F16C);
	ELSE_IF("avx",    ZNEDI3_CPU_X86_AVX);
	ELSE_IF("avx2",   ZNEDI3_CPU_X86_AVX2);
	ELSE_IF("avx512", ZNEDI3_CPU_X86_AVX512F);
#endif
	else
		std::cerr << "unrecognized CPU type: " << param;

	return 0;
}

struct Arguments {
	const char *weights;
	const char *input;
	const char *output;
	int nsize = -1;
	int nns = -1;
	int qual = -1;
	int etype = -1;
	int prescreen = -1;
	int cpu = -1;
	int show_mask = -1;
	char top = 1;
	char bottom = 1;
	int times = 1;
};

constexpr ArgparseOption program_switches[] = {
	{ OPTION_INT,   "s",     "nsize",     offsetof(Arguments, nsize),     nullptr, "window size (0-6)" },
	{ OPTION_INT,   "n",     "nns",       offsetof(Arguments, nns),       nullptr, "neuron size (0-4)" },
	{ OPTION_INT,   "q",     "qual",      offsetof(Arguments, qual),      nullptr, "quality level (1-2)" },
	{ OPTION_INT,   "e",     "etype",     offsetof(Arguments, etype),     nullptr, "error type (0-1)" },
	{ OPTION_INT,   "p",     "prescreen", offsetof(Arguments, prescreen), nullptr, "prescreener (0-4)" },
	{ OPTION_USER1, nullptr, "cpu",       offsetof(Arguments, cpu),       arg_decode_cpu, "cpu type" },
	{ OPTION_INT,   "m",     "show-mask", offsetof(Arguments, show_mask), nullptr, "show mask" },
	{ OPTION_FLAG,  "t",     "top",       offsetof(Arguments, top),       nullptr, "interpolate top field" },
	{ OPTION_FLAG,  "b",     "bottom",    offsetof(Arguments, bottom),    nullptr, "interpolate bottom field" },
	{ OPTION_INT,   nullptr, "times",     offsetof(Arguments, times),     nullptr, "number of iterations" },
	{ OPTION_NULL },
};

constexpr ArgparseOption program_positional[] = {
	{ OPTION_STRING, nullptr, "weights", offsetof(Arguments, weights), nullptr, "path to nnedi3_weights.bin" },
	{ OPTION_STRING, nullptr, "input",   offsetof(Arguments, input),   nullptr, "input BMP file" },
	{ OPTION_STRING, nullptr, "output",  offsetof(Arguments, output),  nullptr, "output BMP file" },
	{ OPTION_NULL },
};

constexpr ArgparseCommandLine program_cmd = {
	program_switches,
	program_positional,
	"testapp",
	"process images with nnedi3",
	nullptr
};


void execute(const Arguments &args, const znedi3_filter *filter, const PlanarImage &in, PlanarImage &out)
{
	std::shared_ptr<void> tmp(aligned_malloc(znedi3_filter_get_tmp_size(filter, in.width[0], in.height[0] / 2), 64), aligned_free);

	std::pair<double, double> results = measure_benchmark(args.times, [&]()
	{
		for (unsigned p = 0; p < 3; ++p) {
			// Interpolate top field.
			if (args.top) {
				znedi3_filter_process(filter, out.width[p], out.height[p] / 2,
				                      in.data[p].data() + in.stride[p], in.stride[p] * 2, out.data[p].data(), out.stride[p] * 2,
				                      tmp.get(), 0);
			}

			// Interpolate bottom field.
			if (args.bottom) {
				znedi3_filter_process(filter, out.width[p], out.height[p] / 2,
				                      in.data[p].data(), in.stride[p] * 2, out.data[p].data() + out.stride[p], out.stride[p] * 2,
				                      tmp.get(), 1);
			}
		}
	});

	double pels_per_field = static_cast<double>(out.width[0]) * (out.height[0] / 2) * 3;
	double pels_filtered = pels_per_field * (!!args.top + !!args.bottom);

	std::cout << "average: " << results.first << '\n';
	std::cout << "min: " << results.second << '\n';
	std::cout << "Mpx/s: " << (pels_filtered / results.first) / 1e6 << '\n';
}

} // namespace


int main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	if ((ret = argparse_parse(&program_cmd, &args, argc, argv)) < 0)
		return ret == ARGPARSE_HELP_MESSAGE ? 0 : ret;

	try {
		std::unique_ptr<znedi3_weights, FreeWeights> weights{ znedi3_weights_from_file(args.weights) };
		if (!weights)
			throw std::runtime_error{ "error loading weights file" };

		znedi3_filter_params params;
		znedi3_filter_params_default(&params);

		auto propagate_if_set = [](auto &dst, int src)
		{
			if (src >= 0)
				dst = static_cast<std::remove_reference_t<decltype(dst)>>(src);
		};

		params.pixel_type = ZNEDI3_PIXEL_BYTE;
		propagate_if_set(params.nsize, args.nsize);
		propagate_if_set(params.nns, args.nns);
		propagate_if_set(params.qual, args.qual);
		propagate_if_set(params.etype, args.etype);
		propagate_if_set(params.prescreen, args.prescreen);
		propagate_if_set(params.cpu, args.cpu);
		propagate_if_set(params.show_mask, args.show_mask);

		std::unique_ptr<znedi3_filter, FreeFilter> filter{ znedi3_filter_create(weights.get(), &params) };
		if (!filter)
			throw std::runtime_error{ "error creating filter" };

		WindowsBitmap in{ args.input, WindowsBitmap::READ_TAG };

		PlanarImage in_planar{};
		bitmap_to_planar(in, in_planar);
		PlanarImage out_planar = in_planar;

		execute(args, filter.get(), in_planar, out_planar);

		WindowsBitmap out{ args.output, in.width(), in.height(), 24 };
		planar_to_bitmap(out_planar, out);
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 2;
	}

	return 0;
}
