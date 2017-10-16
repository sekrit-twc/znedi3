#pragma once

#ifndef ZNEDI3_CPUINFO_H_
#define ZNEDI3_CPUINFO_H_

namespace znedi3 {

/**
 * Enum for CPU type.
 */
enum class CPUClass {
	NONE,
	AUTO,
	AUTO_64B,
#ifdef ZNEDI3_X86
	X86_SSE,
	X86_SSE2,
	X86_AVX,
	X86_F16C,
	X86_AVX2,
	X86_AVX512, // F, CD, BW, DQ, VL
#endif // ZNEDI3_X86
};

constexpr bool cpu_is_autodetect(CPUClass cpu) noexcept
{
	return cpu == CPUClass::AUTO || cpu == CPUClass::AUTO_64B;
}

bool cpu_has_fast_f16(CPUClass cpu) noexcept;
bool cpu_requires_64b_alignment(CPUClass cpu) noexcept;

} // namespace znedi3

#endif // ZNEDI3_CPUINFO_H_
