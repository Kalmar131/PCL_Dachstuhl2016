# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/build

# Include any dependencies generated for this target.
include CMakeFiles/Extracting.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Extracting.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Extracting.dir/flags.make

CMakeFiles/Extracting.dir/extracting.cpp.o: CMakeFiles/Extracting.dir/flags.make
CMakeFiles/Extracting.dir/extracting.cpp.o: /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/src/extracting.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Extracting.dir/extracting.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Extracting.dir/extracting.cpp.o -c /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/src/extracting.cpp

CMakeFiles/Extracting.dir/extracting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Extracting.dir/extracting.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/src/extracting.cpp > CMakeFiles/Extracting.dir/extracting.cpp.i

CMakeFiles/Extracting.dir/extracting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Extracting.dir/extracting.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/src/extracting.cpp -o CMakeFiles/Extracting.dir/extracting.cpp.s

CMakeFiles/Extracting.dir/extracting.cpp.o.requires:
.PHONY : CMakeFiles/Extracting.dir/extracting.cpp.o.requires

CMakeFiles/Extracting.dir/extracting.cpp.o.provides: CMakeFiles/Extracting.dir/extracting.cpp.o.requires
	$(MAKE) -f CMakeFiles/Extracting.dir/build.make CMakeFiles/Extracting.dir/extracting.cpp.o.provides.build
.PHONY : CMakeFiles/Extracting.dir/extracting.cpp.o.provides

CMakeFiles/Extracting.dir/extracting.cpp.o.provides.build: CMakeFiles/Extracting.dir/extracting.cpp.o

# Object files for target Extracting
Extracting_OBJECTS = \
"CMakeFiles/Extracting.dir/extracting.cpp.o"

# External object files for target Extracting
Extracting_EXTERNAL_OBJECTS =

Extracting: CMakeFiles/Extracting.dir/extracting.cpp.o
Extracting: CMakeFiles/Extracting.dir/build.make
Extracting: /usr/lib/x86_64-linux-gnu/libboost_system.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_thread.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
Extracting: /usr/lib/x86_64-linux-gnu/libpthread.so
Extracting: /usr/lib/libpcl_common.so
Extracting: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
Extracting: /usr/lib/libpcl_kdtree.so
Extracting: /usr/lib/libpcl_octree.so
Extracting: /usr/lib/libpcl_search.so
Extracting: /usr/lib/x86_64-linux-gnu/libqhull.so
Extracting: /usr/lib/libpcl_surface.so
Extracting: /usr/lib/libpcl_sample_consensus.so
Extracting: /usr/lib/libOpenNI.so
Extracting: /usr/lib/libOpenNI2.so
Extracting: /usr/lib/libvtkCommon.so.5.8.0
Extracting: /usr/lib/libvtkFiltering.so.5.8.0
Extracting: /usr/lib/libvtkImaging.so.5.8.0
Extracting: /usr/lib/libvtkGraphics.so.5.8.0
Extracting: /usr/lib/libvtkGenericFiltering.so.5.8.0
Extracting: /usr/lib/libvtkIO.so.5.8.0
Extracting: /usr/lib/libvtkRendering.so.5.8.0
Extracting: /usr/lib/libvtkVolumeRendering.so.5.8.0
Extracting: /usr/lib/libvtkHybrid.so.5.8.0
Extracting: /usr/lib/libvtkWidgets.so.5.8.0
Extracting: /usr/lib/libvtkParallel.so.5.8.0
Extracting: /usr/lib/libvtkInfovis.so.5.8.0
Extracting: /usr/lib/libvtkGeovis.so.5.8.0
Extracting: /usr/lib/libvtkViews.so.5.8.0
Extracting: /usr/lib/libvtkCharts.so.5.8.0
Extracting: /usr/lib/libpcl_io.so
Extracting: /usr/lib/libpcl_filters.so
Extracting: /usr/lib/libpcl_features.so
Extracting: /usr/lib/libpcl_keypoints.so
Extracting: /usr/lib/libpcl_registration.so
Extracting: /usr/lib/libpcl_segmentation.so
Extracting: /usr/lib/libpcl_recognition.so
Extracting: /usr/lib/libpcl_visualization.so
Extracting: /usr/lib/libpcl_people.so
Extracting: /usr/lib/libpcl_outofcore.so
Extracting: /usr/lib/libpcl_tracking.so
Extracting: /usr/lib/libpcl_apps.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_system.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_thread.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
Extracting: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
Extracting: /usr/lib/x86_64-linux-gnu/libpthread.so
Extracting: /usr/lib/x86_64-linux-gnu/libqhull.so
Extracting: /usr/lib/libOpenNI.so
Extracting: /usr/lib/libOpenNI2.so
Extracting: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
Extracting: /usr/lib/libvtkCommon.so.5.8.0
Extracting: /usr/lib/libvtkFiltering.so.5.8.0
Extracting: /usr/lib/libvtkImaging.so.5.8.0
Extracting: /usr/lib/libvtkGraphics.so.5.8.0
Extracting: /usr/lib/libvtkGenericFiltering.so.5.8.0
Extracting: /usr/lib/libvtkIO.so.5.8.0
Extracting: /usr/lib/libvtkRendering.so.5.8.0
Extracting: /usr/lib/libvtkVolumeRendering.so.5.8.0
Extracting: /usr/lib/libvtkHybrid.so.5.8.0
Extracting: /usr/lib/libvtkWidgets.so.5.8.0
Extracting: /usr/lib/libvtkParallel.so.5.8.0
Extracting: /usr/lib/libvtkInfovis.so.5.8.0
Extracting: /usr/lib/libvtkGeovis.so.5.8.0
Extracting: /usr/lib/libvtkViews.so.5.8.0
Extracting: /usr/lib/libvtkCharts.so.5.8.0
Extracting: /usr/lib/libpcl_common.so
Extracting: /usr/lib/libpcl_kdtree.so
Extracting: /usr/lib/libpcl_octree.so
Extracting: /usr/lib/libpcl_search.so
Extracting: /usr/lib/libpcl_surface.so
Extracting: /usr/lib/libpcl_sample_consensus.so
Extracting: /usr/lib/libpcl_io.so
Extracting: /usr/lib/libpcl_filters.so
Extracting: /usr/lib/libpcl_features.so
Extracting: /usr/lib/libpcl_keypoints.so
Extracting: /usr/lib/libpcl_registration.so
Extracting: /usr/lib/libpcl_segmentation.so
Extracting: /usr/lib/libpcl_recognition.so
Extracting: /usr/lib/libpcl_visualization.so
Extracting: /usr/lib/libpcl_people.so
Extracting: /usr/lib/libpcl_outofcore.so
Extracting: /usr/lib/libpcl_tracking.so
Extracting: /usr/lib/libpcl_apps.so
Extracting: /usr/lib/libvtkViews.so.5.8.0
Extracting: /usr/lib/libvtkInfovis.so.5.8.0
Extracting: /usr/lib/libvtkWidgets.so.5.8.0
Extracting: /usr/lib/libvtkVolumeRendering.so.5.8.0
Extracting: /usr/lib/libvtkHybrid.so.5.8.0
Extracting: /usr/lib/libvtkParallel.so.5.8.0
Extracting: /usr/lib/libvtkRendering.so.5.8.0
Extracting: /usr/lib/libvtkImaging.so.5.8.0
Extracting: /usr/lib/libvtkGraphics.so.5.8.0
Extracting: /usr/lib/libvtkIO.so.5.8.0
Extracting: /usr/lib/libvtkFiltering.so.5.8.0
Extracting: /usr/lib/libvtkCommon.so.5.8.0
Extracting: /usr/lib/libvtksys.so.5.8.0
Extracting: CMakeFiles/Extracting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable Extracting"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Extracting.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Extracting.dir/build: Extracting
.PHONY : CMakeFiles/Extracting.dir/build

CMakeFiles/Extracting.dir/requires: CMakeFiles/Extracting.dir/extracting.cpp.o.requires
.PHONY : CMakeFiles/Extracting.dir/requires

CMakeFiles/Extracting.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Extracting.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Extracting.dir/clean

CMakeFiles/Extracting.dir/depend:
	cd /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/src /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/src /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/build /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/build /home/loren/Entwicklung/C++/PCL_Dachstuhl2016/Extracting/build/CMakeFiles/Extracting.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Extracting.dir/depend

