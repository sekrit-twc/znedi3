#pragma once

#ifndef ZNEDI3_ZNEDI3_IMPL_H_
#define ZNEDI3_ZNEDI3_IMPL_H_

#include <cstddef>
#include <memory>
#include "kernel.h"
#include "znedi3.h"

struct znedi3_filter {
protected:
	~znedi3_filter() = default;
};

namespace znedi3 {

enum class PixelType {
	BYTE,
	WORD,
	HALF,
	FLOAT,
};

class NNEDI3Weights;

class znedi3_filter : public ::znedi3_filter {
	struct filters;
	struct data;

	std::unique_ptr<filters> m_filters;
	std::unique_ptr<data> m_data_t;
	std::unique_ptr<data> m_data_b;

	std::unique_ptr<filters> setup_filters(const NNEDI3Weights &weights, const znedi3_filter_params &params, unsigned width, unsigned height);

	std::unique_ptr<data> setup_graph(const NNEDI3Weights &weights, const znedi3_filter_params &params, unsigned width, unsigned height, bool parity);
public:
	znedi3_filter(const NNEDI3Weights &weights, const znedi3_filter_params &params, unsigned width, unsigned height);

	~znedi3_filter();

	size_t get_tmp_size() const;

	void process(const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride, void *tmp, bool parity) const;
};

} // namespace znedi3

#endif // ZNEDI3_ZNEDI3_IMPL_H_
