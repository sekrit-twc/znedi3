#ifdef ZNEDI3_X86

#include <immintrin.h>
#include "kernel.h"
#include "kernel_x86.h"

#define USE_FMA 0
#include "kernel_avx_common.h"

namespace znedi3 {

void cubic_interpolation_avx(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n)
{
	cubic_interpolation_avx_impl(src, dst, prescreen, n);
}


std::unique_ptr<Prescreener> create_prescreener_old_avx(const PrescreenerOldCoefficients &coeffs, double pixel_half)
{
	return std::make_unique<PrescreenerOldAVX>(coeffs, pixel_half);
}

std::unique_ptr<Prescreener> create_prescreener_new_avx(const PrescreenerNewCoefficients &coeffs, double pixel_half)
{
	return std::make_unique<PrescreenerNewAVX>(coeffs, pixel_half);
}

std::unique_ptr<Predictor> create_predictor_avx(const PredictorModel &model, bool use_q2)
{
	return std::make_unique<PredictorAVX>(model, use_q2);
}

} // namespace znedi3

#endif // ZNEDI3_X86
