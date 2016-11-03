cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(merge_and_downsample)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(merge_and_downsample merge_and_downsample.cpp)
target_link_libraries(merge_and_downsample ${PCL_LIBRARIES})

