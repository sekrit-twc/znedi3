#pragma once

#ifdef ZNEDI3_X86

#ifndef X86_KERNEL_X86_H_
#define X86_KERNEL_X86_H_

#include <memory>
#include <utility>
#include "alloc.h"
#include "weights.h"

namespace znedi3 {

enum class CPUClass;

class Predictor;

// Polynomial coefficients for exp2f(x - 1) on the domain [1.0, 2.0].
// Coefficients are stored low-order to high-order.
constexpr float EXP2F_X_PLUS1_REMEZ[] = {
	0.509871020343597804469416f,
	0.312146713032169896138863f,
	0.166617139319965966118107f,
	-2.19061993049215080032874e-3f,
	1.3555747234758484073940937e-2f
};

// Coefficients such that converting (EXPF_LN2_INV_SCALED * x + EXPF_ONE_SCALED)
// to integer and reinterpreting the result as a float produces the integer
// component of (x / ln(2)) in the exponent and the fractional component in the
// mantissa.
constexpr float EXPF_LN2_INV_SCALED = 12102203.1615614f; // (1.0 / (127 * ln(2))) * EXPF_ONE_SCALED.
constexpr float EXPF_ONE_SCALED = 1065353216.f; // Integer representation of 1.0f.


struct InterleavedPredictorModel {
	AlignedVector<float> data;
	unsigned xdim;
	unsigned ydim;
	unsigned nns;

	// Filter coefficients are stored interleaved, such that all the softmax
	// coeffcients for kernel position n are stored continguously, followed by
	// all the elliott coefficients for kernel position n.
	//
	// f[nn=0][k=0] f[nn=1][k=0] f[nn=2][k=0] ... f[nn=nns*2][k=0]
	// f[nn=0][k=1] f[nn=1][k=1] f[nn=2][k=1] ... f[nn=nns*2][k=1]
	// ...
	//
	// Likewise, the softmax and elliott biases are stored contiguously.
	float *neurons_q1;
	float *bias_q1;
	float *neurons_q2;
	float *bias_q2;
};

InterleavedPredictorModel create_interleaved_predictor_model(const std::pair<const PredictorTraits, PredictorCoefficients> &model);


void byte_to_float_avx512f(const void *src, void *dst, size_t n);
void word_to_float_avx512f(const void *src, void *dst, size_t n);
void half_to_float_avx512f(const void *src, void *dst, size_t n);
void float_to_byte_avx512f(const void *src, void *dst, size_t n);
void float_to_word_avx512f(const void *src, void *dst, size_t n);
void float_to_half_avx512f(const void *src, void *dst, size_t n);

std::unique_ptr<Predictor> create_predictor_avx512f(const std::pair<const PredictorTraits, PredictorCoefficients> &model, bool use_q2);


pixel_io_func select_pixel_io_func_x86(PixelType in, PixelType out, CPUClass cpu);

std::unique_ptr<Predictor> create_predictor_x86(const std::pair<const PredictorTraits, PredictorCoefficients> &model, bool use_q2, CPUClass cpu);

} // namespace znedi3

#endif // X86_KERNEL_X86_H_

#endif // ZNEDI3_X86
