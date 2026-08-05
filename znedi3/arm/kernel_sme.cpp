// Streaming-mode SME entry points. This file is the only unit compiled with
// SME target features enabled. It must not contain any non-streaming vector
// code. Apple Silicon has no non-streaming SVE, and enabling the SME feature
// tempts the auto-vectorizer into emitting streaming-compatible SVE
// instructions in regular functions, which segfault outside streaming mode.

#if defined(ZNEDI3_ARM) && defined(ZNEDI3_ARM_SME)

#include <cstddef>
#include <arm_sme.h>
#include "kernel_arm.h"

namespace znedi3 {
namespace {

// Accumulate the matrix product
//   activation[batch][rows] = matrix[cols][rows]^T x input[cols][batch]
// as a sequence of rank-1 updates on the ZA tiles, one row block of up to
// four tiles at a time.
inline void sme_matmul_block(const float *matrix, const float *inputT, float *activation, unsigned rows, unsigned cols,
                             unsigned batch, unsigned num, unsigned i, unsigned nt) __arm_streaming __arm_inout("za")
{
	const svbool_t ptrue = svptrue_b32();
	unsigned vw = static_cast<unsigned>(svcntw());

	svbool_t p0 = svwhilelt_b32(i + 0 * vw, rows);
	svbool_t p1 = svwhilelt_b32(i + 1 * vw, rows);
	svbool_t p2 = svwhilelt_b32(i + 2 * vw, rows);
	svbool_t p3 = svwhilelt_b32(i + 3 * vw, rows);

	svzero_za();

	for (unsigned j = 0; j < cols; ++j) {
		const float *col = matrix + static_cast<size_t>(j) * rows + i;
		svfloat32_t x = svld1_f32(ptrue, inputT + static_cast<size_t>(j) * batch);

		svmopa_za32_f32_m(0, p0, ptrue, svld1_f32(p0, col + 0 * vw), x);
		if (nt >= 2)
			svmopa_za32_f32_m(1, p1, ptrue, svld1_f32(p1, col + 1 * vw), x);
		if (nt >= 3)
			svmopa_za32_f32_m(2, p2, ptrue, svld1_f32(p2, col + 2 * vw), x);
		if (nt >= 4)
			svmopa_za32_f32_m(3, p3, ptrue, svld1_f32(p3, col + 3 * vw), x);
	}

	// The vertical tile slices hold one gathered pixel each.
	for (unsigned p = 0; p < num; ++p) {
		float *dst = activation + static_cast<size_t>(p) * rows + i;

		svst1_ver_za32(0, p, p0, dst + 0 * vw);
		if (nt >= 2)
			svst1_ver_za32(1, p, p1, dst + 1 * vw);
		if (nt >= 3)
			svst1_ver_za32(2, p, p2, dst + 2 * vw);
		if (nt >= 4)
			svst1_ver_za32(3, p, p3, dst + 3 * vw);
	}
}

} // namespace


unsigned sme_streaming_vector_length_words() noexcept
{
	return static_cast<unsigned>(svcntsw());
}

__arm_locally_streaming __arm_new("za")
void sme_matmul(const float *matrix, const float *inputT, float *activation,
                unsigned rows, unsigned cols, unsigned batch, unsigned num) noexcept
{
	unsigned vw = static_cast<unsigned>(svcntw());

	for (unsigned i = 0; i < rows; i += 4 * vw) {
		unsigned nt = rows - i >= 4 * vw ? 4 : (rows - i + vw - 1) / vw;
		sme_matmul_block(matrix, inputT, activation, rows, cols, batch, num, i, nt);
	}
}

} // namespace znedi3

#endif // ZNEDI3_ARM && ZNEDI3_ARM_SME
