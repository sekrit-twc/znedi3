MY_CFLAGS := -O2 -fPIC $(CFLAGS)
MY_CXXFLAGS := -std=c++14 -O2 -fPIC $(CXXFLAGS)
MY_CPPFLAGS := -Iznedi3 -Ivsxx $(CPPFLAGS)
MY_LDFLAGS := $(LDFLAGS)
MY_LIBS := $(LIBS)

znedi3_HDRS = \
	znedi3/align.h \
	znedi3/alloc.h \
	znedi3/ccdep.h \
	znedi3/cpuinfo.h \
	znedi3/kernel.h \
	znedi3/weights.h \
	znedi3/x86/kernel_x86.h \
	znedi3/x86/kernel_avx_common.h \
	znedi3/x86/kernel_sse_commmon.h \
	znedi3/x86/cpuinfo_x86.h \
	znedi3/znedi3.h \
	znedi3/znedi3_impl.h

znedi3_OBJS = \
	znedi3/cpuinfo.o \
	znedi3/kernel.o \
	znedi3/weights.o \
	znedi3/x86/cpuinfo_x86.o \
	znedi3/x86/kernel_avx.o \
	znedi3/x86/kernel_avx2.o \
	znedi3/x86/kernel_avx512.o \
	znedi3/x86/kernel_f16c.o \
	znedi3/x86/kernel_sse.o \
	znedi3/x86/kernel_sse2.o \
	znedi3/x86/kernel_x86.o \
	znedi3/znedi3.o \
	znedi3/znedi3_impl.o

testapp_HDRS = \
	testapp/argparse.h \
	testapp/mmap.h \
	testapp/timer.h \
	testapp/win32_bitmap.h

testapp_OBJS = \
	testapp/argparse.o \
	testapp/main.o \
	testapp/mmap.o \
	testapp/win32_bitmap.o

vsxx_HDRS = \
	vsxx/VapourSynth.h \
	vsxx/VapourSynth++.hpp \
	vsxx/VSScript.h \
	vsxx/VSHelper.h \
	vsxx/vsxx_pluginmain.h

ifeq ($(X86), 1)
  znedi3/x86/kernel_avx.o: EXTRA_CXXFLAGS := -mavx
  znedi3/x86/kernel_avx2.o: EXTRA_CXXFLAGS := -mavx2 -mfma
  znedi3/x86/kernel_f16c.o: EXTRA_CXXFLAGS := -mavx -mf16c
  znedi3/x86/kernel_sse.o: EXTRA_CXXFLAGS := -msse
  znedi3/x86/kernel_sse2.o: EXTRA_CXXFLAGS := -msse2
  MY_CPPFLAGS := -DZNEDI3_X86 $(MY_CPPFLAGS)
endif

ifeq ($(X86_AVX512), 1)
  znedi3/x86/kernel_avx512.o: EXTRA_CXXFLAGS := -mavx512f -mfma
  MY_CPPFLAGS := -DZNEDI3_X86_AVX512 $(MY_CPPFLAGS)
endif

all: vsznedi3.so

testapp/testapp: $(testapp_OBJS) $(znedi3_OBJS)
	$(CXX) $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@

vsznedi3.so: vsznedi3/vsznedi3.o vsxx/vsxx_pluginmain.o $(znedi3_OBJS)
	$(CXX) -shared $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@

clean:
	rm -f *.a *.o *.so testapp/testapp testapp/*.o znedi3/*.o znedi3/x86/*.o vsznedi3/*.o vsxx/*.o

%.o: %.cpp $(znedi3_HDRS) $(testapp_HDRS) $(vsxx_HDRS)
	$(CXX) -c $(EXTRA_CXXFLAGS) $(MY_CXXFLAGS) $(MY_CPPFLAGS) $< -o $@

.PHONY: clean
