cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(ground_filter)

find_package(PCL 1.7.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (ground_filter ground_filter.cpp)
target_link_libraries (ground_filter ${PCL_LIBRARIES})