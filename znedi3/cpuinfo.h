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
	AUTO_64B, // AUTO, also allowing 512-bit (64-byte) AVX-512 on x86, SME on arm.
#if defined(ZNEDI3_X86)
	X86_SSE,
	X86_SSE2,
	X86_AVX,
	X86_F16C,
	X86_AVX2,
	X86_AVX512, // F, CD, BW, DQ, VL
#endif
#if defined(ZNEDI3_ARM)
	ARM_NEON,
	ARM_SVE,
	ARM_SME,
#endif
};

constexpr bool cpu_is_autodetect(CPUClass cpu) noexcept
{
	return cpu == CPUClass::AUTO || cpu == CPUClass::AUTO_64B;
}

} // namespace znedi3

#endif // ZNEDI3_CPUINFO_H_
