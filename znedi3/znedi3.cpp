#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "weights.h"
#include "znedi3.h"
#include "znedi3_impl.h"

#ifdef _WIN32
  #include <Windows.h>
#endif

static_assert(std::numeric_limits<float>::is_iec559, "IEEE-754 required");
static_assert(sizeof(float) == 4, "8-bit byte required");

namespace {

struct FileCloser {
	void operator()(std::FILE *file) const { std::fclose(file); }
};

#ifdef _WIN32
std::wstring utf8_to_utf16(const std::string &s)
{
	if (s.size() > static_cast<size_t>(INT_MAX))
		throw std::length_error{ "" };

	std::wstring us(s.size(), L'\0');

	int len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), &us[0], static_cast<int>(us.size()));
	if (!len)
		throw std::runtime_error{ "bad UTF-8/UTF-16 conversion" };

	us.resize(len);
	return us;
}
#endif

} // namespace


znedi3_weights *znedi3_weights_read(const void *data, size_t size) try
{
	if (size != znedi3::NNEDI3_WEIGHTS_SIZE)
		return nullptr;

	return znedi3::read_nnedi3_weights(static_cast<const float *>(data)).release();
} catch (...) {
	return nullptr;
}

znedi3_weights *znedi3_weights_from_file(const char *path) try
{
#ifdef _WIN32
	std::wstring path_utf16 = utf8_to_utf16(path);
	std::unique_ptr<std::FILE, FileCloser> file_uptr{ _wfopen(path_utf16.c_str(), L"rb") };
#else
	std::unique_ptr<std::FILE, FileCloser> file_uptr{ std::fopen(path, "rb") };
#endif
	std::FILE *file = file_uptr.get();
	if (!file)
		return nullptr;

	if (std::fseek(file, 0, SEEK_END))
		return nullptr;

	long size = std::ftell(file);
	if (size != znedi3::NNEDI3_WEIGHTS_SIZE)
		return nullptr;

	std::rewind(file);

	std::unique_ptr<char[]> buf{ new char[size] };
	char *p = buf.get();
	size_t remain = znedi3::NNEDI3_WEIGHTS_SIZE;

	while (remain) {
		size_t n = std::fread(p, 1, remain, file);
		if (std::ferror(file) || std::feof(file))
			return nullptr;

		p += n;
		remain -= n;
	}

	return znedi3_weights_read(buf.get(), znedi3::NNEDI3_WEIGHTS_SIZE);
} catch (...) {
	return nullptr;
}

void znedi3_weights_free(znedi3_weights *ptr)
{
	delete static_cast<znedi3::NNEDI3Weights *>(ptr);
}

void znedi3_filter_params_default(znedi3_filter_params *params)
{
	params->pixel_type = static_cast<znedi3_pixel_type_e>(-1);
	params->bit_depth = 0;
	params->nsize = ZNEDI3_NSIZE_32x4;
	params->nns = ZNEDI3_NNS_32;
	params->qual = ZNEDI3_QUAL_1;
	params->etype = ZNEDI3_ETYPE_ABS;
	params->prescreen = ZNEDI3_PRESCREEN_NEW_L0;
	params->cpu = ZNEDI3_CPU_AUTO;
	params->int16_predict = 0;
	params->int16_prescreen = 0;
	params->slow_exp = 0;
	params->show_mask = 0;
}

znedi3_filter *znedi3_filter_create(const znedi3_weights *weights, const znedi3_filter_params *params) try
{
	const znedi3::NNEDI3Weights *nnedi3 = static_cast<const znedi3::NNEDI3Weights *>(weights);
	std::unique_ptr<znedi3::znedi3_filter> filter = std::make_unique<znedi3::znedi3_filter>(*nnedi3, *params);
	return filter.release();
} catch (...) {
	return nullptr;
}

void znedi3_filter_free(znedi3_filter *ptr)
{
	delete static_cast<znedi3::znedi3_filter *>(ptr);
}

size_t znedi3_filter_get_tmp_size(const znedi3_filter *ptr, unsigned width, unsigned height)
{
	return static_cast<const znedi3::znedi3_filter *>(ptr)->get_tmp_size(width, height);
}

void znedi3_filter_process(const znedi3_filter *ptr, unsigned width, unsigned height,
                           const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride, void *tmp, unsigned parity)
{
	static_cast<const znedi3::znedi3_filter *>(ptr)->process(width, height, src, src_stride, dst, dst_stride, tmp, parity);
}
