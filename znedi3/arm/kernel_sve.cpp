// Non-streaming SVE predictor for cores with native SVE (e.g. Graviton3 =
// Neoverse V1 @ 256-bit, Graviton4 = Neoverse V2 @ 128-bit). This must never
// be selected on Apple Silicon, which has no non-streaming SVE. The SVE
// target feature is confined to this translation unit.

#if defined(ZNEDI3_ARM) && defined(ZNEDI3_ARM_SVE)

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <arm_neon.h>
#include <arm_sve.h>
#include "alloc.h"
#include "ccdep.h"
#include "kernel.h"
#include "kernel_arm.h"
#include "kernel_neon_common.h"

namespace znedi3 {
namespace {

inline svfloat32_t sve_expf_f32(svbool_t pg, svfloat32_t x)
{
	x = svmad_f32_x(pg, x, svdup_f32(EXPF_LN2_INV_SCALED), svdup_f32(EXPF_ONE_SCALED));
	svint32_t xi = svcvt_s32_f32_x(pg, x);

	// Clear the mantissa. This represents exp2(floor(x)).
	svfloat32_t i = svreinterpret_f32_s32(svand_n_s32_x(pg, xi, 0x7F800000L));
	// Reset the exponent to zero. This represents exp2(x - floor(x)).
	svfloat32_t f = svreinterpret_f32_s32(svorr_n_s32_x(pg, svand_n_s32_x(pg, xi, 0x007FFFFFL), 0x3F800000L));

	x = svdup_f32(EXP2F_X_PLUS1_REMEZ[4]);
	x = svmad_f32_x(pg, x, f, svdup_f32(EXP2F_X_PLUS1_REMEZ[3]));
	x = svmad_f32_x(pg, x, f, svdup_f32(EXP2F_X_PLUS1_REMEZ[2]));
	x = svmad_f32_x(pg, x, f, svdup_f32(EXP2F_X_PLUS1_REMEZ[1]));
	x = svmad_f32_x(pg, x, f, svdup_f32(EXP2F_X_PLUS1_REMEZ[0]));

	return svmul_f32_x(pg, i, x);
}

inline svfloat32_t sve_softmax_exp_f32(svbool_t pg, svfloat32_t x)
{
	svuint32_t xsign = svand_n_u32_x(pg, svreinterpret_u32_f32(x), 0x80000000U);

	x = svmin_n_f32_x(pg, svabs_f32_x(pg, x), 80.0f);
	x = svreinterpret_f32_u32(svorr_u32_x(pg, svreinterpret_u32_f32(x), xsign));
	return sve_expf_f32(pg, x);
}

inline svfloat32_t sve_elliott_f32(svbool_t pg, svfloat32_t x)
{
	svfloat32_t den = svadd_n_f32_x(pg, svabs_f32_x(pg, x), 1.0f);
	return svdiv_f32_x(pg, x, den);
}


// Scale, bias, and store one row chunk of the four per-pixel accumulators.
// nns is a multiple of 16 and vw <= 16, so a chunk never straddles the
// softmax and elliott halves of the matrix.
inline FORCE_INLINE void sgemv_store_chunk_sve(svbool_t pc, const float *bias, unsigned row_base, unsigned nns,
                                               float *activation_softmax, float *activation_elliott, const float *scale,
                                               svfloat32_t a0, svfloat32_t a1, svfloat32_t a2, svfloat32_t a3)
{
	const svbool_t pg = svptrue_b32();
	svfloat32_t bias_v = svld1_f32(pc, bias + row_base);
	float *dst = row_base >= nns ? activation_elliott + (row_base - nns) : activation_softmax + row_base;

	svst1_f32(pc, dst + 0 * static_cast<ptrdiff_t>(nns), svmla_n_f32_x(pg, bias_v, a0, scale[0]));
	svst1_f32(pc, dst + 1 * static_cast<ptrdiff_t>(nns), svmla_n_f32_x(pg, bias_v, a1, scale[1]));
	svst1_f32(pc, dst + 2 * static_cast<ptrdiff_t>(nns), svmla_n_f32_x(pg, bias_v, a2, scale[2]));
	svst1_f32(pc, dst + 3 * static_cast<ptrdiff_t>(nns), svmla_n_f32_x(pg, bias_v, a3, scale[3]));
}

inline FORCE_INLINE void sgemv_x4_sve(const float *matrix, const float *inputT, const float *bias, unsigned matrix_rows, unsigned matrix_cols,
                                      float *activation_softmax, float *activation_elliott, unsigned nns, const float *scale)
{
	const svbool_t pg = svptrue_b32();
	const unsigned vw = static_cast<unsigned>(svcntw());

	for (unsigned i = 0; i < matrix_rows; i += 4 * vw) {
		svbool_t p0 = svwhilelt_b32(i + 0 * vw, matrix_rows);
		svbool_t p1 = svwhilelt_b32(i + 1 * vw, matrix_rows);
		svbool_t p2 = svwhilelt_b32(i + 2 * vw, matrix_rows);
		svbool_t p3 = svwhilelt_b32(i + 3 * vw, matrix_rows);

		svfloat32_t accum00 = svdup_f32(0.0f), accum01 = svdup_f32(0.0f), accum02 = svdup_f32(0.0f), accum03 = svdup_f32(0.0f);
		svfloat32_t accum10 = svdup_f32(0.0f), accum11 = svdup_f32(0.0f), accum12 = svdup_f32(0.0f), accum13 = svdup_f32(0.0f);
		svfloat32_t accum20 = svdup_f32(0.0f), accum21 = svdup_f32(0.0f), accum22 = svdup_f32(0.0f), accum23 = svdup_f32(0.0f);
		svfloat32_t accum30 = svdup_f32(0.0f), accum31 = svdup_f32(0.0f), accum32 = svdup_f32(0.0f), accum33 = svdup_f32(0.0f);

		for (unsigned j = 0; j < matrix_cols; ++j) {
			const float *col = matrix + static_cast<size_t>(j) * matrix_rows + i;
			const float *x = inputT + static_cast<size_t>(j) * 4;

			svfloat32_t c0 = svld1_f32(p0, col + 0 * vw);
			svfloat32_t c1 = svld1_f32(p1, col + 1 * vw);
			svfloat32_t c2 = svld1_f32(p2, col + 2 * vw);
			svfloat32_t c3 = svld1_f32(p3, col + 3 * vw);

			accum00 = svmla_n_f32_x(pg, accum00, c0, x[0]);
			accum01 = svmla_n_f32_x(pg, accum01, c1, x[0]);
			accum02 = svmla_n_f32_x(pg, accum02, c2, x[0]);
			accum03 = svmla_n_f32_x(pg, accum03, c3, x[0]);

			accum10 = svmla_n_f32_x(pg, accum10, c0, x[1]);
			accum11 = svmla_n_f32_x(pg, accum11, c1, x[1]);
			accum12 = svmla_n_f32_x(pg, accum12, c2, x[1]);
			accum13 = svmla_n_f32_x(pg, accum13, c3, x[1]);

			accum20 = svmla_n_f32_x(pg, accum20, c0, x[2]);
			accum21 = svmla_n_f32_x(pg, accum21, c1, x[2]);
			accum22 = svmla_n_f32_x(pg, accum22, c2, x[2]);
			accum23 = svmla_n_f32_x(pg, accum23, c3, x[2]);

			accum30 = svmla_n_f32_x(pg, accum30, c0, x[3]);
			accum31 = svmla_n_f32_x(pg, accum31, c1, x[3]);
			accum32 = svmla_n_f32_x(pg, accum32, c2, x[3]);
			accum33 = svmla_n_f32_x(pg, accum33, c3, x[3]);
		}

		sgemv_store_chunk_sve(p0, bias, i + 0 * vw, nns, activation_softmax, activation_elliott, scale, accum00, accum10, accum20, accum30);
		if (i + 1 * vw < matrix_rows)
			sgemv_store_chunk_sve(p1, bias, i + 1 * vw, nns, activation_softmax, activation_elliott, scale, accum01, accum11, accum21, accum31);
		if (i + 2 * vw < matrix_rows)
			sgemv_store_chunk_sve(p2, bias, i + 2 * vw, nns, activation_softmax, activation_elliott, scale, accum02, accum12, accum22, accum32);
		if (i + 3 * vw < matrix_rows)
			sgemv_store_chunk_sve(p3, bias, i + 3 * vw, nns, activation_softmax, activation_elliott, scale, accum03, accum13, accum23, accum33);
	}
}

inline FORCE_INLINE void softmax_exp_sve(float *ptr, unsigned n)
{
	const unsigned vw = static_cast<unsigned>(svcntw());

	for (unsigned i = 0; i < n; i += vw) {
		svbool_t pr = svwhilelt_b32(i, n);
		svst1_f32(pr, ptr + i, sve_softmax_exp_f32(pr, svld1_f32(pr, ptr + i)));
	}
}

inline FORCE_INLINE void wae5_x4_sve(const float *softmax, const float *elliott, unsigned n, float *mstd)
{
	const unsigned vw = static_cast<unsigned>(svcntw());

	for (unsigned p = 0; p < 4; ++p) {
		svfloat32_t vsum = svdup_f32(0.0f);
		svfloat32_t wsum = svdup_f32(0.0f);

		for (unsigned i = 0; i < n; i += vw) {
			svbool_t pr = svwhilelt_b32(i, n);

			svfloat32_t s = svld1_f32(pr, softmax + p * static_cast<ptrdiff_t>(n) + i);
			svfloat32_t e = sve_elliott_f32(pr, svld1_f32(pr, elliott + p * static_cast<ptrdiff_t>(n) + i));

			vsum = svmla_f32_m(pr, vsum, s, e);
			wsum = svadd_f32_m(pr, wsum, s);
		}

		float vsum_reduced = svaddv_f32(svptrue_b32(), vsum);
		float wsum_reduced = svaddv_f32(svptrue_b32(), wsum);

		if (wsum_reduced > 1e-10f)
			mstd[3 * 4 + p] += (5.0f * vsum_reduced) / wsum_reduced * mstd[1 * 4 + p] + mstd[0 * 4 + p];
		else
			mstd[3 * 4 + p] += mstd[0 * 4 + p];
	}
}


class PredictorSVE final : public Predictor {
	InterleavedPredictorModel m_model;
	double m_inv_filter_size;
	bool m_use_q2;

	void apply_model(const float *input, float *activation, float *mstd, const double *sums, const double *sumsqs) const
	{
		unsigned filter_size = m_model.xdim * m_model.ydim;
		unsigned nns = m_model.nns;

		float *activation_softmax = activation;
		float *activation_elliott = activation + 4 * static_cast<ptrdiff_t>(nns);

		input_stddev_neon(sums, sumsqs, mstd, 4, 4, m_inv_filter_size);

		for (unsigned q = 0; q < (m_use_q2 ? 2U : 1U); ++q) {
			const float *neurons = q ? m_model.neurons_q2 : m_model.neurons_q1;
			const float *bias = q ? m_model.bias_q2 : m_model.bias_q1;

			sgemv_x4_sve(neurons, input, bias, nns * 2, filter_size, activation_softmax, activation_elliott, nns, mstd + 2 * 4);
			softmax_exp_sve(activation_softmax, 4 * nns);
			wae5_x4_sve(activation_softmax, activation_elliott, nns, mstd);
		}
	}
public:
	PredictorSVE(const PredictorModel &model, bool use_q2) :
		m_model(create_interleaved_predictor_model(model)),
		m_inv_filter_size{ 1.0 / (m_model.xdim * m_model.ydim) },
		m_use_q2{ use_q2 }
	{
		assert(model.first.xdim * model.first.ydim <= 48 * 6);
	}

	size_t get_tmp_size() const noexcept override
	{
		FakeAllocator alloc;

		alloc.allocate_n<float>(48 * 6 * 4);
		alloc.allocate_n<float>(256 * 2 * 4);
		alloc.allocate_n<float>(4 * 4);

		return alloc.count();
	}

	void process(const float * const src[6], float *dst, const unsigned char *prescreen, void *tmp, unsigned n) const noexcept override
	{
		LinearAllocator alloc{ tmp };

		ptrdiff_t window_offset_y = 3 - static_cast<ptrdiff_t>(m_model.ydim / 2);
		ptrdiff_t window_offset_x = static_cast<ptrdiff_t>(m_model.xdim) / 2 - 1;

		float *input = alloc.allocate_n<float>(48 * 6 * 4);
		float *activation = alloc.allocate_n<float>(256 * 2 * 4);
		float *mstd = alloc.allocate_n<float>(4 * 4);

		double sums[4] = {};
		double sumsqs[4] = {};
		unsigned gathered_idx[4];
		size_t num_gathered = 0;

		for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
			if (prescreen[i])
				continue;

			gather_pixels_neon(src + window_offset_y, i - window_offset_x, m_model.xdim, m_model.ydim,
			                   input, 4, static_cast<unsigned>(num_gathered), &sums[num_gathered], &sumsqs[num_gathered]);
			gathered_idx[num_gathered] = static_cast<unsigned>(i);
			++num_gathered;

			if (num_gathered == 4) {
				apply_model(input, activation, mstd, sums, sumsqs);

				for (ptrdiff_t idx = 0; idx < 4; ++idx) {
					dst[gathered_idx[idx]] = mstd[3 * 4 + idx] * (m_use_q2 ? 0.5f : 1.0f);
				}

				num_gathered = 0;
			}
		}
		if (num_gathered) {
			apply_model(input, activation, mstd, sums, sumsqs);

			for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(num_gathered); ++idx) {
				dst[gathered_idx[idx]] = mstd[3 * 4 + idx] * (m_use_q2 ? 0.5f : 1.0f);
			}
		}
	}
};

} // namespace


std::unique_ptr<Predictor> create_predictor_sve(const PredictorModel &model, bool use_q2)
{
	// A chunk of vw rows must not straddle the softmax/elliott boundary,
	// which is a multiple of 16.
	if (16 % svcntw() != 0)
		return nullptr;

	return std::make_unique<PredictorSVE>(model, use_q2);
}

} // namespace znedi3

#endif // ZNEDI3_ARM && ZNEDI3_ARM_SVE
