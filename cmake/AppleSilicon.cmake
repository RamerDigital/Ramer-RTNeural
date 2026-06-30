set(RTNEURAL_APPLE_M_CPU "apple-m1" CACHE STRING "Apple M-series CPU to target for microarchitecture optimization")
option(RTNEURAL_LTO "Enable ThinLTO for compilation and linking" ON)
option(RTNEURAL_APPLE_FAST_MATH "Enable fast-math compilation flags" OFF)

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "RTNeural -- Configuring optimizations for Apple Silicon (${RTNEURAL_APPLE_M_CPU})")
    
    # Target CPU instruction scheduling
    target_compile_options(RTNeural PUBLIC "-mcpu=${RTNEURAL_APPLE_M_CPU}")
    
    # ThinLTO Support
    if(RTNEURAL_LTO)
        target_compile_options(RTNeural PUBLIC "-flto=thin")
        target_link_options(RTNeural PUBLIC "-flto=thin")
        message(STATUS "RTNeural -- ThinLTO enabled")
    endif()

    # Fast-Math Option
    if(RTNEURAL_APPLE_FAST_MATH)
        target_compile_options(RTNeural PUBLIC "-ffast-math")
        message(STATUS "RTNeural -- Fast-Math enabled")
    endif()
endif()
