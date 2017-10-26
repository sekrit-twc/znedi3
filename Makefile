PREFIX ?= /usr
CXX ?= g++
CXXFLAGS := -DZNEDI3_X86 -I./znedi3 -I./vsxx $(CPPFLAGS) -fPIC -Wall -O3 $(CXXFLAGS)
LDFLAGS := $(LDFLAGS)
LIBEXT ?= so

NOSIMD_OBJ := znedi3/cpuinfo.o znedi3/kernel.o znedi3/weights.o znedi3/znedi3.o znedi3/znedi3_impl.o \
vsxx/vsxx_pluginmain.o vsznedi3/vsznedi3.o \
znedi3/x86/cpuinfo_x86.o znedi3/x86/kernel_x86.o

SSE_OBJ := znedi3/x86/kernel_sse.o
SSE2_OBJ := znedi3/x86/kernel_sse2.o
F16C_OBJ := znedi3/x86/kernel_f16c.o
AVX_OBJ := znedi3/x86/kernel_avx.o
AVX2_OBJ := znedi3/x86/kernel_avx2.o
AVX512_OBJ := znedi3/x86/kernel_avx512.o

ALL_OBJ := $(NOSIMD_OBJ) $(SSE_OBJ) $(SSE2_OBJ) $(F16C_OBJ) $(AVX_OBJ) $(AVX2_OBJ) $(AVX512_OBJ)

all: libznedi3.$(LIBEXT)

clean:
	rm -f $(ALL_OBJ)
	rm -f libznedi3.$(LIBEXT)

libznedi3.$(LIBEXT): $(ALL_OBJ)
	$(CXX) -shared -s -o libznedi3.$(LIBEXT) $(ALL_OBJ) $(LDFLAGS)

$(NOSIMD_OBJ):
	$(CXX) -c $(CXXFLAGS) -o $*.o $*.cpp

$(SSE_OBJ):
	$(CXX) -c -msse $(CXXFLAGS) -o $*.o $*.cpp

$(SSE2_OBJ):
	$(CXX) -c -msse -msse2 $(CXXFLAGS) -o $*.o $*.cpp

$(F16C_OBJ):
	$(CXX) -c -mf16c $(CXXFLAGS) -o $*.o $*.cpp

$(AVX_OBJ):
	$(CXX) -c -DUSE_FMA=0 -mavx $(CXXFLAGS) -o $*.o $*.cpp

$(AVX2_OBJ):
	$(CXX) -c -DUSE_FMA=1 -mavx2 -mfma $(CXXFLAGS) -o $*.o $*.cpp

$(AVX512_OBJ):
	$(CXX) -c -DUSE_FMA=1 -DZNEDI3_X86_AVX512 -mfma -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq $(CXXFLAGS) -o $*.o $*.cpp

install: libznedi3.$(LIBEXT) nnedi3_weights.bin
	install -c -d $(PREFIX)/lib/vapoursynth
	install -c -m 755 libznedi3.$(LIBEXT) $(PREFIX)/lib/vapoursynth/
	install -c -m 644 nnedi3_weights.bin $(PREFIX)/lib/vapoursynth/
