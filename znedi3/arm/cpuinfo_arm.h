#pragma once

#ifdef ZNEDI3_ARM

#ifndef ZNEDI3_ARM_CPUINFO_ARM_H_
#define ZNEDI3_ARM_CPUINFO_ARM_H_

namespace znedi3 {

enum class CPUClass;

/**
 * Selected AArch64 feature flags.
 */
struct ArmCapabilities {
	unsigned neon : 1; /**< Advanced SIMD. Mandatory in AArch64. */
	unsigned sve  : 1; /**< Non-streaming SVE. */
	unsigned sve2 : 1; /**< Non-streaming SVE2. */
	unsigned sme  : 1; /**< Scalable Matrix Extension. */
	unsigned sme2 : 1; /**< SME version 2. */
	unsigned sve_vl; /**< Non-streaming SVE vector length in bytes, 0 if no SVE. */
};

/**
 * Get the AArch64 feature flags on the current CPU.
 *
 * @return capabilities
 */
ArmCapabilities query_arm_capabilities() noexcept;

} // namespace znedi3

#endif // ZNEDI3_ARM_CPUINFO_ARM_H_

#endif // ZNEDI3_ARM
