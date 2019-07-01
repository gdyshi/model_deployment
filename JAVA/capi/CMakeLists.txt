cmake_minimum_required(VERSION 3.5)

project(jni_model)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_compile_options(-std=c++11)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_VERBOSE_MAKEFILE ON)
    add_compile_options(-Wall -Wextra -pedantic-errors)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4)
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.11)
        add_compile_options(/permissive-)
    endif()
endif()
include_directories(${CMAKE_SOURCE_DIR}/../include/tensorflow/c
										/opt/jdk-12.0.1/include
										/opt/jdk-12.0.1/include/linux)
link_directories(${CMAKE_SOURCE_DIR}/../../C++/lib
                ${CMAKE_SOURCE_DIR}/../../C++/src/build)


add_library(jni_model SHARED jni_utils.cpp jni_utils.h jni_adapterc.cpp model.h)
target_link_libraries(jni_model model  tensorflow)



