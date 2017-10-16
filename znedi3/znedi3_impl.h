#pragma once

#ifndef ZNEDI3_IMPL_H_
#define ZNEDI3_IMPL_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
#include "kernel.h"
#include "znedi3.h"

struct znedi3_filter {
protected:
	~znedi3_filter() = default;
};

namespace znedi3 {

enum class CPUClass {
	NONE,
	AUTO,
	AUTO_64B,
};

enum class PixelType {
	BYTE,
	WORD,
	HALF,
	FLOAT,
};

class NNEDI3Weights;

class znedi3_pixel_io {
public:
	virtual ~znedi3_pixel_io() = default;

	virtual void convert(const void *src, void *dst, size_t n) const = 0;
};

class znedi3_prescreener {
public:
	virtual ~znedi3_prescreener() = default;

	virtual void prescreen(const void *src, ptrdiff_t src_stride, unsigned char *prescreen, unsigned n) const = 0;
};

class znedi3_predictor {
public:
	virtual ~znedi3_predictor() = default;

	virtual void predict(const void *src, ptrdiff_t src_stride, void *dst, const unsigned char *prescreen, unsigned n) const = 0;
};

class znedi3_filter : public ::znedi3_filter {
	std::unique_ptr<Prescreener> m_prescreener;
	std::unique_ptr<Predictor> m_predictor;
	interpolate_func m_interpolate_func;
	pixel_io_func m_pixel_load_func;
	pixel_io_func m_pixel_store_func;

	PixelType m_type;
	CPUClass m_cpu;
public:
	explicit znedi3_filter(const NNEDI3Weights &weights, const znedi3_filter_params &params);

	size_t get_tmp_size(unsigned width, unsigned height) const;

	void process(unsigned width, unsigned height, const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride, void *tmp, unsigned parity) const;
};

} // namespace znedi3

#endif // ZNEDI3_IMPL_H_
