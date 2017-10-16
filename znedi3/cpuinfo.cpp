#include "cpuinfo.h"

#ifdef ZNEDI3_X86
  #include "x86/cpuinfo_x86.h"
#endif

namespace znedi3 {

bool cpu_has_fast_f16(CPUClass cpu) noexcept
{
	bool ret = false;
#ifdef ZNEDI3_X86
	ret = cpu_has_fast_f16_x86(cpu);
#endif
	return ret;
}

bool cpu_requires_64b_alignment(CPUClass cpu) noexcept
{
	bool ret = false;
#ifdef ZNEDI3_X86
	ret = cpu_requires_64b_alignment_x86(cpu);
#endif
	return ret;
}

} // namespace znedi3
