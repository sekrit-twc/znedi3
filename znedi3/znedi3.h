#ifndef ZNEDI3_H_
#define ZNEDI3_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum znedi3_cpu_type_e {
	ZNEDI3_CPU_NONE     = 0, /**< Portable C-based implementation. */
	ZNEDI3_CPU_AUTO     = 1, /**< Runtime CPU detection. */
	ZNEDI3_CPU_AUTO_64B = 2  /**< Allow use of 64-byte (512-bit) instructions. */
#if defined(__i386) || defined(_M_IX86) || defined(_M_X64) || defined(__x86_64__)
	,ZNEDI3_CPU_X86_MMX       = 1000,
	ZNEDI3_CPU_X86_SSE        = 1001,
	ZNEDI3_CPU_X86_SSE2       = 1002,
	ZNEDI3_CPU_X86_SSE3       = 1003,
	ZNEDI3_CPU_X86_SSSE3      = 1004,
	ZNEDI3_CPU_X86_SSE41      = 1005,
	ZNEDI3_CPU_X86_SSE42      = 1006,
	ZNEDI3_CPU_X86_AVX        = 1007,
	ZNEDI3_CPU_X86_F16C       = 1008, /**< AVX with F16C extension (e.g. Ivy Bridge). */
	ZNEDI3_CPU_X86_AVX2       = 1009,
	ZNEDI3_CPU_X86_AVX512F    = 1010,
	ZNEDI3_CPU_X86_AVX512_SKX = 1011, /**< AVX-512 {F,CD,VL,BW,DQ} (e.g. Skylake-X/SP). */
	ZNEDI3_CPU_X86_AVX512_CLX = 1012, /**< SKX + VNNI */
	ZNEDI3_CPU_X86_AVX512_PMC = 1013, /**< SKX + VBMI + IFMA52 */
	ZNEDI3_CPU_X86_AVX512_SNC = 1014, /**< PMC + VPOPCNTDQ + BITALG + VBMI2 + VNNI */
	ZNEDI3_CPU_X86_AVX512_WLC = 1015, /**< SNC + VP2INTERSECT */
	ZNEDI3_CPU_X86_AVX512_GLC = 1016  /**< WLC + FP16 + BF16 */
#endif
} znedi3_cpu_type_e;

typedef enum znedi3_pixel_type_e {
	ZNEDI3_PIXEL_BYTE,
	ZNEDI3_PIXEL_WORD,
	ZNEDI3_PIXEL_HALF,
	ZNEDI3_PIXEL_FLOAT
} znedi3_pixel_type_e;

typedef enum znedi3_nsize_e {
	ZNEDI3_NSIZE_8x6  = 0,
	ZNEDI3_NSIZE_16x6 = 1,
	ZNEDI3_NSIZE_32x6 = 2,
	ZNEDI3_NSIZE_48x6 = 3,
	ZNEDI3_NSIZE_8x4  = 4,
	ZNEDI3_NSIZE_16x4 = 5,
	ZNEDI3_NSIZE_32x4 = 6
} znedi3_nsize_e;

typedef enum znedi3_nns_e {
	ZNEDI3_NNS_16  = 0,
	ZNEDI3_NNS_32  = 1,
	ZNEDI3_NNS_64  = 2,
	ZNEDI3_NNS_128 = 3,
	ZNEDI3_NNS_256 = 4
} znedi3_nns_e;

typedef enum znedi3_qual_e {
	ZNEDI3_QUAL_1 = 1,
	ZNEDI3_QUAL_2 = 2
} znedi3_qual_e;

typedef enum znedi3_etype_e {
	ZNEDI3_ETYPE_ABS = 0,
	ZNEDI3_ETYPE_MSE = 1
} znedi3_etype_e;

typedef enum znedi3_prescreen_e {
	ZNEDI3_PRESCREEN_NONE   = 0,
	ZNEDI3_PRESCREEN_OLD    = 1,
	ZNEDI3_PRESCREEN_NEW_L0 = 2,
	ZNEDI3_PRESCREEN_NEW_L1 = 3,
	ZNEDI3_PRESCREEN_NEW_L2 = 4
} znedi3_prescreen_e;


typedef struct znedi3_weights znedi3_weights;

znedi3_weights *znedi3_weights_read(const void *data, size_t size);

znedi3_weights *znedi3_weights_from_file(const char *path);

void znedi3_weights_free(znedi3_weights *ptr);


typedef struct znedi3_filter_params {
	znedi3_pixel_type_e pixel_type;
	unsigned bit_depth;
	znedi3_nsize_e nsize;
	znedi3_nns_e nns;
	znedi3_qual_e qual;
	znedi3_etype_e etype;
	znedi3_prescreen_e prescreen;
	znedi3_cpu_type_e cpu;
	unsigned char int16_prescreen;
	unsigned char int16_predict;
	unsigned char slow_exp;
	unsigned char show_mask;
} znedi3_filter_params;

typedef struct znedi3_filter znedi3_filter;

void znedi3_filter_params_default(znedi3_filter_params *params);

znedi3_filter *znedi3_filter_create(const znedi3_weights *weights, const znedi3_filter_params *params, unsigned width, unsigned height);

void znedi3_filter_free(znedi3_filter *ptr);

size_t znedi3_filter_get_tmp_size(const znedi3_filter *ptr);

void znedi3_filter_process(const znedi3_filter *ptr, const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride, void *tmp, int parity);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ZNEDI3_H_ */
