#ifdef ZNEDI3_X86

#include "kernel_x86.h"
#include "znedi3_impl.h"

namespace znedi3 {

std::unique_ptr<Predictor> create_predictor_x86(const std::pair<const PredictorTraits, PredictorCoefficients> &model, bool use_q2, CPUClass cpu)
{
	return nullptr;
}

} // namespace znedi3

#endif // ZNEDI3_X86
