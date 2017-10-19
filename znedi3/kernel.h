#pragma once

#ifndef ZNEDI3_KERNEL_H_
#define ZNEDI3_KERNEL_H_

#include <cstddef>
#include <memory>
#include <utility>
#include "weights.h"

namespace znedi3 {

enum class CPUClass;
enum class PixelType;

typedef void (*pixel_io_func)(const void *src, void *dst, size_t n);
typedef void (*interpolate_func)(const void *src, ptrdiff_t src_stride, void *dst, const unsigned char *prescreen, unsigned n);

class Prescreener {
public:
	virtual ~Prescreener() = default;

	virtual size_t get_tmp_size() const = 0;

	virtual void process(const void *src, ptrdiff_t src_stride, unsigned char *prescreen, void *tmp, unsigned n) const = 0;
};

class Predictor {
public:
	virtual ~Predictor() = default;

	virtual size_t get_tmp_size() const = 0;

	virtual void process(const void *src, ptrdiff_t src_stride, void *dst, const unsigned char *prescreen, void *tmp, unsigned n) const = 0;
};

pixel_io_func select_pixel_io_func(PixelType in, PixelType out, CPUClass cpu);
interpolate_func select_interpolate_func(CPUClass cpu);

std::unique_ptr<Prescreener> create_prescreener_old(const PrescreenerOldCoefficients &coeffs, double pixel_half, CPUClass cpu);
std::unique_ptr<Prescreener> create_prescreener_new(const PrescreenerNewCoefficients &coeffs, double pixel_half, CPUClass cpu);
std::unique_ptr<Predictor> create_predictor(const std::pair<const PredictorTraits, PredictorCoefficients> &model, bool use_q2, CPUClass cpu);

} // namespace znedi3

#endif // ZNEDI3_KERNEL_H_
