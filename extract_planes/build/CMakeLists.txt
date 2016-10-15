cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(extract_planes)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (extract_planes extract_planes.cpp)
target_link_libraries (extract_planes ${PCL_LIBRARIES})

