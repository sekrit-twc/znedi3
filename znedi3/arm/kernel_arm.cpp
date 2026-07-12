#ifdef ZNEDI3_ARM

#include <algorithm>
#include <cassert>
#include "alloc.h"
#include "cpuinfo.h"
#include "cpuinfo_arm.h"
#include "kernel.h"
#include "kernel_arm.h"
#include "znedi3_impl.h"

namespace znedi3 {
namespace {

pixel_io_func select_pixel_io_func_neon(PixelType in, PixelType out)
{
	if (in == PixelType::BYTE && out == PixelType::FLOAT)
		return byte_to_float_neon;
	else if (in == PixelType::WORD && out == PixelType::FLOAT)
		return word_to_float_neon;
	else if (in == PixelType::HALF && out == PixelType::FLOAT)
		return half_to_float_neon;
	else if (in == PixelType::FLOAT && out == PixelType::BYTE)
		return float_to_byte_neon;
	else if (in == PixelType::FLOAT && out == PixelType::WORD)
		return float_to_word_neon;
	else if (in == PixelType::FLOAT && out == PixelType::HALF)
		return float_to_half_neon;
	else
		return nullptr;
}

} // namespace


pixel_io_func select_pixel_io_func_arm(PixelType in, PixelType out, CPUClass cpu)
{
	ArmCapabilities caps = query_arm_capabilities();
	pixel_io_func ret = nullptr;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.neon)
			ret = select_pixel_io_func_neon(in, out);
	} else {
		if (!ret && cpu >= CPUClass::ARM_NEON)
			ret = select_pixel_io_func_neon(in, out);
	}

	return ret;
}

interpolate_func select_interpolate_func_arm(CPUClass cpu)
{
	ArmCapabilities caps = query_arm_capabilities();
	interpolate_func ret = nullptr;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.neon)
			ret = cubic_interpolation_neon;
	} else {
		if (!ret && cpu >= CPUClass::ARM_NEON)
			ret = cubic_interpolation_neon;
	}

	return ret;
}

std::unique_ptr<Prescreener> create_prescreener_old_arm(const PrescreenerOldCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	ArmCapabilities caps = query_arm_capabilities();
	std::unique_ptr<Prescreener> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.neon)
			ret = create_prescreener_old_neon(coeffs, pixel_half);
	} else {
		if (!ret && cpu >= CPUClass::ARM_NEON)
			ret = create_prescreener_old_neon(coeffs, pixel_half);
	}

	return ret;
}

std::unique_ptr<Prescreener> create_prescreener_new_arm(const PrescreenerNewCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	ArmCapabilities caps = query_arm_capabilities();
	std::unique_ptr<Prescreener> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.neon)
			ret = create_prescreener_new_neon(coeffs, pixel_half);
	} else {
		if (!ret && cpu >= CPUClass::ARM_NEON)
			ret = create_prescreener_new_neon(coeffs, pixel_half);
	}

	return ret;
}

std::unique_ptr<Predictor> create_predictor_arm(const PredictorModel &model, bool use_q2, CPUClass cpu)
{
	ArmCapabilities caps = query_arm_capabilities();
	std::unique_ptr<Predictor> ret;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_ARM_SME
		// The SME unit is shared between cores. Treat it like the 512-bit
		// x86 case and require explicit opt-in through AUTO_64B.
		if (!ret && cpu == CPUClass::AUTO_64B && caps.sme2)
			ret = create_predictor_sme(model, use_q2);
#endif
#ifdef ZNEDI3_ARM_SVE
		// At 128-bit vector length SVE has no advantage over NEON.
		if (!ret && caps.sve && caps.sve_vl >= 32)
			ret = create_predictor_sve(model, use_q2);
#endif
		if (!ret && caps.neon)
			ret = create_predictor_neon(model, use_q2);
	} else {
#ifdef ZNEDI3_ARM_SME
		if (!ret && cpu >= CPUClass::ARM_SME)
			ret = create_predictor_sme(model, use_q2);
#endif
#ifdef ZNEDI3_ARM_SVE
		if (!ret && cpu >= CPUClass::ARM_SVE)
			ret = create_predictor_sve(model, use_q2);
#endif
		if (!ret && cpu >= CPUClass::ARM_NEON)
			ret = create_predictor_neon(model, use_q2);
	}

	return ret;
}

} // namespace znedi3

#endif // ZNEDI3_ARM
