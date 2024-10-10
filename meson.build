project('znedi3', 'c', 'cpp',
        version: '2.2',
        default_options: ['c_std=c99', 'cpp_std=c++14', 'buildtype=release', 'b_lto=true'],
        meson_version: '>=0.46')


warnings = [
  '-Wall',
  '-Wextra',
  '-Wshadow',
]

cflags = [
  warnings,
  '-fvisibility=hidden',
  '-DGRAPHENGINE_IMPL_NAMESPACE=znedi3',
  '-DPACKAGE_VERSION="@0@"'.format(meson.project_version()),
  '-DNDEBUG',
  '-Igraphengine/include',
  '-Iznedi3',
  '-Ivsxx',
  '-Ivsxx/vapoursynth',
]

ldflags = [
]

includes = include_directories(
  'graphengine/include',
  'znedi3',
  'vsxx',
  'vsxx/vapoursynth',
)

znedi3_sources = [
  'znedi3/kernel.cpp',
  'znedi3/weights.cpp',
  'znedi3/znedi3.cpp',
  'znedi3/znedi3_impl.cpp',
]

vsznedi3_sources = [
  'vsznedi3/vsznedi3.cpp',
  'vsxx/vsxx4_pluginmain.cpp',
]

graphengine_sources = [
  'graphengine/graphengine/cpuinfo.cpp',
  'graphengine/graphengine/graph.cpp',
  'graphengine/graphengine/node.cpp',
]

testapp_sources = [
  'testapp/argparse.cpp',
  'testapp/main.cpp',
  'testapp/mmap.cpp',
  'testapp/win32_bitmap.cpp',
]

znedi3_helper_libs = []

host_cpu_family = host_machine.cpu_family()

if host_cpu_family.startswith('x86')
  cflags += ['-mfpmath=sse', '-msse2', '-DZNEDI3_X86=1']

  znedi3_sources += [
    'znedi3/x86/cpuinfo_x86.cpp',
    'znedi3/x86/kernel_x86.cpp',
    'znedi3/x86/kernel_sse.cpp',
    'znedi3/x86/kernel_sse2.cpp',
  ]

  znedi3_sources_x86_avx = [
    'znedi3/x86/kernel_avx.cpp',
  ]

  znedi3_sources_x86_f16c = [
    'znedi3/x86/kernel_f16c.cpp',
  ]

  znedi3_sources_x86_avx2 = [
    'znedi3/x86/kernel_avx2.cpp',
  ]

  znedi3_sources_x86_avx512 = [
    'znedi3/x86/kernel_avx512.cpp',
  ]

  znedi3_helper_libs += static_library('avx',
                                znedi3_sources_x86_avx,
                                include_directories: includes,
                                cpp_args: [cflags, '-mavx', '-mtune=sandybridge'],
                                pic: true,
                                install: false)

  znedi3_helper_libs += static_library('f16c',
                                znedi3_sources_x86_f16c,
                                include_directories: includes,
                                cpp_args: [cflags, '-mavx', '-mf16c', '-mtune=ivybridge'],
                                pic: true,
                                install: false)

  znedi3_helper_libs += static_library('avx2',
                                znedi3_sources_x86_avx2,
                                include_directories: includes,
                                cpp_args: [cflags, '-mavx2', '-mfma', '-mtune=haswell'],
                                pic: true,
                                install: false)

  znedi3_helper_libs += static_library('avx512',
                                znedi3_sources_x86_avx2,
                                include_directories: includes,
                                cpp_args: [cflags, '-mavx512f', '-mfma', '-mtune=skylake-avx512'],
                                pic: true,
                                install: false)

  graphengine_sources += [
    'graphengine/graphengine/x86/cpuinfo_x86.cpp'
  ]

endif

if host_cpu_family.startswith('aarch') or host_cpu_family.startswith('arm')
  cflags += ['-DZNEDI3_ARM=1']

  znedi3_sources += [
    'znedi3/arm/kernel_arm.cpp',
    'znedi3/arm/kernel_neon.cpp',
  ]
endif

znedi3_helper_libs += static_library('graphengine',
                              graphengine_sources,
                              include_directories: includes,
                              cpp_args: [cflags],
                              pic: true,
                              install: false)

cxx = meson.get_compiler('cpp')


shared_module('vsznedi3',
              znedi3_sources + vsznedi3_sources,
              include_directories: includes,
              link_args: ldflags,
              c_args: cflags,
              cpp_args: cflags,
              link_with: znedi3_helper_libs,
              install: true)

executable   ('testapp',
              znedi3_sources + testapp_sources,
              include_directories: includes,
              link_args: ldflags,
              c_args: cflags,
              cpp_args: cflags,
              link_with: znedi3_helper_libs,
              install: false)
