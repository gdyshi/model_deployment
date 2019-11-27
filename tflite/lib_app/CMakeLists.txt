cmake_minimum_required(VERSION 3.5)

project(myapp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -DNDEBUG -fPIC")

include_directories(${CMAKE_SOURCE_DIR}/tflib/include)
link_directories(${CMAKE_SOURCE_DIR}/tflib/lib)


add_library(mylib SHARED model.cc model.h)
target_link_libraries(mylib  tensorflow-lite)

add_executable(myapp example.cc model.h)
target_link_libraries(myapp tensorflow-lite mylib pthread)
add_dependencies(myapp mylib)
