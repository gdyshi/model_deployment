cmake_minimum_required(VERSION 3.5)

project(hello_tf)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_VERBOSE_MAKEFILE ON)
    add_compile_options(-Wall -Wextra -pedantic-errors)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4)
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.11)
        add_compile_options(/permissive-)
    endif()
endif()
include_directories(${CMAKE_SOURCE_DIR}/../include/tensorflow/c)
link_directories(${CMAKE_SOURCE_DIR}/../lib)


add_library(model SHARED tf_utils.cpp tf_utils.hpp model.cpp model.h)
target_link_libraries(model  tensorflow)

add_executable(hello_tf example.cpp model.h)
target_link_libraries(hello_tf tensorflow model)
add_dependencies(hello_tf model)

#add_executable(test example.cpp model.h)


