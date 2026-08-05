#ifdef ZNEDI3_ARM

#include "cpuinfo_arm.h"

#if defined(__APPLE__)
  #include <sys/sysctl.h>
#elif defined(__linux__)
  #include <sys/auxv.h>
  #include <sys/prctl.h>

  // Stable kernel ABI constants. Provide fallbacks so that detection does not
  // depend on the build environment: musl does not define the HWCAP bits at
  // all, and glibc/kernel headers predating the extensions lack them too.
  #ifndef HWCAP_SVE
    #define HWCAP_SVE (1UL << 22)
  #endif
  #ifndef HWCAP2_SVE2
    #define HWCAP2_SVE2 (1UL << 1)
  #endif
  #ifndef HWCAP2_SME
    #define HWCAP2_SME (1UL << 23)
  #endif
  #ifndef HWCAP2_SME2
    #define HWCAP2_SME2 (1UL << 37)
  #endif
  #ifndef PR_SVE_GET_VL
    #define PR_SVE_GET_VL 51
  #endif
  #ifndef PR_SVE_VL_LEN_MASK
    #define PR_SVE_VL_LEN_MASK 0xffff
  #endif
#endif

namespace znedi3 {
namespace {

#if defined(__APPLE__)
bool query_sysctl_flag(const char *name) noexcept
{
	int val = 0;
	size_t len = sizeof(val);

	if (sysctlbyname(name, &val, &len, nullptr, 0))
		return false;
	return val != 0;
}
#endif

ArmCapabilities do_query_arm_capabilities() noexcept
{
	ArmCapabilities caps = { 0 };

	caps.neon = 1;

#if defined(__APPLE__)
	// Apple Silicon has no non-streaming SVE.
	caps.sme = query_sysctl_flag("hw.optional.arm.FEAT_SME");
	caps.sme2 = query_sysctl_flag("hw.optional.arm.FEAT_SME2");
#elif defined(__linux__)
	unsigned long hwcap = getauxval(AT_HWCAP);
	unsigned long hwcap2 = getauxval(AT_HWCAP2);

	caps.sve = !!(hwcap & HWCAP_SVE);
	caps.sve2 = !!(hwcap2 & HWCAP2_SVE2);
	caps.sme = !!(hwcap2 & HWCAP2_SME);
	caps.sme2 = !!(hwcap2 & HWCAP2_SME2);

	if (caps.sve) {
		int vl = prctl(PR_SVE_GET_VL);
		if (vl > 0)
			caps.sve_vl = static_cast<unsigned>(vl & PR_SVE_VL_LEN_MASK);
	}
#endif

	return caps;
}

} // namespace


ArmCapabilities query_arm_capabilities() noexcept
{
	static const ArmCapabilities caps = do_query_arm_capabilities();
	return caps;
}

} // namespace znedi3

#endif // ZNEDI3_ARM
