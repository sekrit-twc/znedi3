#ifdef ZNEDI3_X86

#include <immintrin.h>
#include "kernel.h"
#include "kernel_x86.h"

#define USE_FMA 0
#include "kernel_avx_common.h"

namespace znedi3 {

std::unique_ptr<Prescreener> create_prescreener_old_avx(const PrescreenerOldCoefficients &coeffs, double pixel_half)
{
	return std::make_unique<PrescreenerOldAVX>(coeffs, pixel_half);
}

std::unique_ptr<Predictor> create_predictor_avx(const PredictorModel &model, bool use_q2)
{
	return std::make_unique<PredictorAVX>(model, use_q2);
}

} // namespace znedi3

#endif // ZNEDI3_X86
