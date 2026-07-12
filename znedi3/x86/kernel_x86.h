#pragma once

#ifdef ZNEDI3_X86

#ifndef X86_KERNEL_X86_H_
#define X86_KERNEL_X86_H_

#include <memory>
#include <utility>
#include "alloc.h"
#include "kernel.h"
#include "kernel_interleaved.h"
#include "weights.h"

namespace znedi3 {

enum class CPUClass;
enum class PixelType;

// SSE
std::unique_ptr<Prescreener> create_prescreener_old_sse(const PrescreenerOldCoefficients &coeffs, double pixel_half);
std::unique_ptr<Prescreener> create_prescreener_new_sse(const PrescreenerNewCoefficients &coeffs, double pixel_half);
std::unique_ptr<Predictor> create_predictor_sse(const PredictorModel &model, bool use_q2);

// SSE2
void byte_to_float_sse2(const void *src, void *dst, size_t n);
void word_to_float_sse2(const void *src, void *dst, size_t n);
void float_to_byte_sse2(const void *src, void *dst, size_t n);
void float_to_word_sse2(const void *src, void *dst, size_t n);

void cubic_interpolation_sse2(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n);

std::unique_ptr<Predictor> create_predictor_sse2(const PredictorModel &model, bool use_q2);

// AVX
void cubic_interpolation_avx(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n);

std::unique_ptr<Prescreener> create_prescreener_old_avx(const PrescreenerOldCoefficients &coeffs, double pixel_half);
std::unique_ptr<Prescreener> create_prescreener_new_avx(const PrescreenerNewCoefficients &coeffs, double pixel_half);
std::unique_ptr<Predictor> create_predictor_avx(const PredictorModel &model, bool use_q2);

// F16C
void half_to_float_f16c(const void *src, void *dst, size_t n);
void float_to_half_f16c(const void *src, void *dst, size_t n);

// AVX2
void byte_to_float_avx2(const void *src, void *dst, size_t n);
void word_to_float_avx2(const void *src, void *dst, size_t n);
void float_to_byte_avx2(const void *src, void *dst, size_t n);
void float_to_word_avx2(const void *src, void *dst, size_t n);

void cubic_interpolation_avx2(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n);

std::unique_ptr<Prescreener> create_prescreener_old_avx2(const PrescreenerOldCoefficients &coeffs, double pixel_half);
std::unique_ptr<Prescreener> create_prescreener_new_avx2(const PrescreenerNewCoefficients &coeffs, double pixel_half);
std::unique_ptr<Predictor> create_predictor_avx2(const PredictorModel &model, bool use_q2);

// AVX-512F
void byte_to_float_avx512f(const void *src, void *dst, size_t n);
void word_to_float_avx512f(const void *src, void *dst, size_t n);
void half_to_float_avx512f(const void *src, void *dst, size_t n);
void float_to_byte_avx512f(const void *src, void *dst, size_t n);
void float_to_word_avx512f(const void *src, void *dst, size_t n);
void float_to_half_avx512f(const void *src, void *dst, size_t n);

void cubic_interpolation_avx512f(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n);

std::unique_ptr<Prescreener> create_prescreener_old_avx512f(const PrescreenerOldCoefficients &coeffs, double pixel_half);
std::unique_ptr<Prescreener> create_prescreener_new_avx512f(const PrescreenerNewCoefficients &coeffs, double pixel_half);
std::unique_ptr<Predictor> create_predictor_avx512f(const PredictorModel &model, bool use_q2);


pixel_io_func select_pixel_io_func_x86(PixelType in, PixelType out, CPUClass cpu);
interpolate_func select_interpolate_func_x86(CPUClass cpu);

std::unique_ptr<Prescreener> create_prescreener_old_x86(const PrescreenerOldCoefficients &coeffs, double pixel_half, CPUClass cpu);
std::unique_ptr<Prescreener> create_prescreener_new_x86(const PrescreenerNewCoefficients &coeffs, double pixel_half, CPUClass cpu);
std::unique_ptr<Predictor> create_predictor_x86(const PredictorModel &model, bool use_q2, CPUClass cpu);

} // namespace znedi3

#endif // X86_KERNEL_X86_H_

#endif // ZNEDI3_X86
