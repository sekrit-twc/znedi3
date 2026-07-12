#pragma once

#ifdef ZNEDI3_ARM

#ifndef ZNEDI3_ARM_KERNEL_ARM_H_
#define ZNEDI3_ARM_KERNEL_ARM_H_

#include <memory>
#include <utility>
#include "alloc.h"
#include "kernel.h"
#include "kernel_interleaved.h"
#include "weights.h"

namespace znedi3 {

enum class CPUClass;
enum class PixelType;

// NEON. Compiled for baseline armv8-a, so this path runs on every AArch64
// CPU (Graviton1 onward, all Apple Silicon, Raspberry Pi class included).
void byte_to_float_neon(const void *src, void *dst, size_t n);
void word_to_float_neon(const void *src, void *dst, size_t n);
void half_to_float_neon(const void *src, void *dst, size_t n);
void float_to_byte_neon(const void *src, void *dst, size_t n);
void float_to_word_neon(const void *src, void *dst, size_t n);
void float_to_half_neon(const void *src, void *dst, size_t n);

void cubic_interpolation_neon(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n);

std::unique_ptr<Prescreener> create_prescreener_old_neon(const PrescreenerOldCoefficients &coeffs, double pixel_half);
std::unique_ptr<Prescreener> create_prescreener_new_neon(const PrescreenerNewCoefficients &coeffs, double pixel_half);
std::unique_ptr<Predictor> create_predictor_neon(const PredictorModel &model, bool use_q2);

#ifdef ZNEDI3_ARM_SVE
// Non-streaming SVE (e.g. Graviton3/4, Neoverse V1/V2). Predictor only; the
// remaining kernels stay NEON. Returns nullptr if the vector length is
// unsupported.
std::unique_ptr<Predictor> create_predictor_sve(const PredictorModel &model, bool use_q2);
#endif

#ifdef ZNEDI3_ARM_SME
// SME. The matrix multiplication runs in streaming mode as outer products on
// the ZA tiles; everything else stays NEON.
//
// The streaming-mode entry points are implemented in a separate translation
// unit compiled with the SME target enabled. Everything else must be built
// without SVE/SME codegen, as Apple Silicon has no non-streaming SVE.
unsigned sme_streaming_vector_length_words() noexcept;

void sme_matmul(const float *matrix, const float *inputT, float *activation,
                unsigned rows, unsigned cols, unsigned batch, unsigned num) noexcept;

std::unique_ptr<Predictor> create_predictor_sme(const PredictorModel &model, bool use_q2);
#endif


pixel_io_func select_pixel_io_func_arm(PixelType in, PixelType out, CPUClass cpu);
interpolate_func select_interpolate_func_arm(CPUClass cpu);

std::unique_ptr<Prescreener> create_prescreener_old_arm(const PrescreenerOldCoefficients &coeffs, double pixel_half, CPUClass cpu);
std::unique_ptr<Prescreener> create_prescreener_new_arm(const PrescreenerNewCoefficients &coeffs, double pixel_half, CPUClass cpu);
std::unique_ptr<Predictor> create_predictor_arm(const PredictorModel &model, bool use_q2, CPUClass cpu);

} // namespace znedi3

#endif // ZNEDI3_ARM_KERNEL_ARM_H_

#endif // ZNEDI3_ARM
