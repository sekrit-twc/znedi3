#pragma once

#ifdef ZNEDI3_X86

#ifndef X86_KERNEL_X86_H_
#define X86_KERNEL_X86_H_

#include <memory>
#include <utility>
#include "weights.h"

namespace znedi3 {

enum class CPUClass;


std::unique_ptr<Predictor> create_predictor_x86(const std::pair<const PredictorTraits, PredictorCoefficients> &model, bool use_q2, CPUClass cpu);

} // namespace znedi3

#endif // X86_KERNEL_X86_H_

#endif // ZNEDI3_X86
