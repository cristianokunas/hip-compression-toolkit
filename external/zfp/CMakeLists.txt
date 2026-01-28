# ZFP external library CMakeLists.txt
# This builds the ZFP library with HIP support

cmake_minimum_required(VERSION 3.18)

# ZFP source files for core compression
set(ZFP_CORE_SOURCES
    src/bitstream.c
    src/zfp.c
)

# ZFP encode/decode sources (all types and dimensions)
file(GLOB ZFP_CODEC_SOURCES 
    src/encode*.c
    src/decode*.c
)

# ZFP HIP sources
set(ZFP_HIP_SOURCES
    src/hip/interface.cpp
)

# All ZFP sources
set(ZFP_SOURCES
    ${ZFP_CORE_SOURCES}
    ${ZFP_CODEC_SOURCES}
)

# Build ZFP as object library
add_library(zfp_core OBJECT ${ZFP_SOURCES})

target_include_directories(zfp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(zfp_core PRIVATE
    ZFP_WITH_HIP
    BIT_STREAM_WORD_TYPE=uint64
)

# Set position independent code
set_property(TARGET zfp_core PROPERTY POSITION_INDEPENDENT_CODE ON)

# HIP interface (compiled separately with hipcc)
if(NOT CUDA_BACKEND)
    set_source_files_properties(${ZFP_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(zfp_hip OBJECT ${ZFP_HIP_SOURCES})
    
    target_include_directories(zfp_hip PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hip
        ${CMAKE_CURRENT_SOURCE_DIR}/src/share
    )
    
    target_compile_definitions(zfp_hip PRIVATE
        ZFP_WITH_HIP
        BIT_STREAM_WORD_TYPE=uint64
    )
    
    set_property(TARGET zfp_hip PROPERTY POSITION_INDEPENDENT_CODE ON)
    
    if(DEFINED CMAKE_HIP_ARCHITECTURES)
        set_property(TARGET zfp_hip PROPERTY HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
    endif()
endif()
