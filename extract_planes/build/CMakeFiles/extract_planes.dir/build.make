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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kalmar/PCL_Dachstuhl2016/extract_planes/build

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kalmar/PCL_Dachstuhl2016/extract_planes/build

# Include any dependencies generated for this target.
include CMakeFiles/extract_planes.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/extract_planes.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/extract_planes.dir/flags.make

CMakeFiles/extract_planes.dir/extract_planes.cpp.o: CMakeFiles/extract_planes.dir/flags.make
CMakeFiles/extract_planes.dir/extract_planes.cpp.o: extract_planes.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/kalmar/PCL_Dachstuhl2016/extract_planes/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/extract_planes.dir/extract_planes.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/extract_planes.dir/extract_planes.cpp.o -c /home/kalmar/PCL_Dachstuhl2016/extract_planes/build/extract_planes.cpp

CMakeFiles/extract_planes.dir/extract_planes.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/extract_planes.dir/extract_planes.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/kalmar/PCL_Dachstuhl2016/extract_planes/build/extract_planes.cpp > CMakeFiles/extract_planes.dir/extract_planes.cpp.i

CMakeFiles/extract_planes.dir/extract_planes.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/extract_planes.dir/extract_planes.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/kalmar/PCL_Dachstuhl2016/extract_planes/build/extract_planes.cpp -o CMakeFiles/extract_planes.dir/extract_planes.cpp.s

CMakeFiles/extract_planes.dir/extract_planes.cpp.o.requires:
.PHONY : CMakeFiles/extract_planes.dir/extract_planes.cpp.o.requires

CMakeFiles/extract_planes.dir/extract_planes.cpp.o.provides: CMakeFiles/extract_planes.dir/extract_planes.cpp.o.requires
	$(MAKE) -f CMakeFiles/extract_planes.dir/build.make CMakeFiles/extract_planes.dir/extract_planes.cpp.o.provides.build
.PHONY : CMakeFiles/extract_planes.dir/extract_planes.cpp.o.provides

CMakeFiles/extract_planes.dir/extract_planes.cpp.o.provides.build: CMakeFiles/extract_planes.dir/extract_planes.cpp.o

# Object files for target extract_planes
extract_planes_OBJECTS = \
"CMakeFiles/extract_planes.dir/extract_planes.cpp.o"

# External object files for target extract_planes
extract_planes_EXTERNAL_OBJECTS =

extract_planes: CMakeFiles/extract_planes.dir/extract_planes.cpp.o
extract_planes: CMakeFiles/extract_planes.dir/build.make
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_system.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_thread.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
extract_planes: /usr/lib/x86_64-linux-gnu/libpthread.so
extract_planes: /usr/lib/libpcl_common.so
extract_planes: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
extract_planes: /usr/lib/libpcl_kdtree.so
extract_planes: /usr/lib/libpcl_octree.so
extract_planes: /usr/lib/libpcl_search.so
extract_planes: /usr/lib/x86_64-linux-gnu/libqhull.so
extract_planes: /usr/lib/libpcl_surface.so
extract_planes: /usr/lib/libpcl_sample_consensus.so
extract_planes: /usr/lib/libOpenNI.so
extract_planes: /usr/lib/libOpenNI2.so
extract_planes: /usr/lib/libvtkCommon.so.5.8.0
extract_planes: /usr/lib/libvtkFiltering.so.5.8.0
extract_planes: /usr/lib/libvtkImaging.so.5.8.0
extract_planes: /usr/lib/libvtkGraphics.so.5.8.0
extract_planes: /usr/lib/libvtkGenericFiltering.so.5.8.0
extract_planes: /usr/lib/libvtkIO.so.5.8.0
extract_planes: /usr/lib/libvtkRendering.so.5.8.0
extract_planes: /usr/lib/libvtkVolumeRendering.so.5.8.0
extract_planes: /usr/lib/libvtkHybrid.so.5.8.0
extract_planes: /usr/lib/libvtkWidgets.so.5.8.0
extract_planes: /usr/lib/libvtkParallel.so.5.8.0
extract_planes: /usr/lib/libvtkInfovis.so.5.8.0
extract_planes: /usr/lib/libvtkGeovis.so.5.8.0
extract_planes: /usr/lib/libvtkViews.so.5.8.0
extract_planes: /usr/lib/libvtkCharts.so.5.8.0
extract_planes: /usr/lib/libpcl_io.so
extract_planes: /usr/lib/libpcl_filters.so
extract_planes: /usr/lib/libpcl_features.so
extract_planes: /usr/lib/libpcl_keypoints.so
extract_planes: /usr/lib/libpcl_registration.so
extract_planes: /usr/lib/libpcl_segmentation.so
extract_planes: /usr/lib/libpcl_recognition.so
extract_planes: /usr/lib/libpcl_visualization.so
extract_planes: /usr/lib/libpcl_people.so
extract_planes: /usr/lib/libpcl_outofcore.so
extract_planes: /usr/lib/libpcl_tracking.so
extract_planes: /usr/lib/libpcl_apps.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_system.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_thread.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
extract_planes: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
extract_planes: /usr/lib/x86_64-linux-gnu/libpthread.so
extract_planes: /usr/lib/x86_64-linux-gnu/libqhull.so
extract_planes: /usr/lib/libOpenNI.so
extract_planes: /usr/lib/libOpenNI2.so
extract_planes: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
extract_planes: /usr/lib/libvtkCommon.so.5.8.0
extract_planes: /usr/lib/libvtkFiltering.so.5.8.0
extract_planes: /usr/lib/libvtkImaging.so.5.8.0
extract_planes: /usr/lib/libvtkGraphics.so.5.8.0
extract_planes: /usr/lib/libvtkGenericFiltering.so.5.8.0
extract_planes: /usr/lib/libvtkIO.so.5.8.0
extract_planes: /usr/lib/libvtkRendering.so.5.8.0
extract_planes: /usr/lib/libvtkVolumeRendering.so.5.8.0
extract_planes: /usr/lib/libvtkHybrid.so.5.8.0
extract_planes: /usr/lib/libvtkWidgets.so.5.8.0
extract_planes: /usr/lib/libvtkParallel.so.5.8.0
extract_planes: /usr/lib/libvtkInfovis.so.5.8.0
extract_planes: /usr/lib/libvtkGeovis.so.5.8.0
extract_planes: /usr/lib/libvtkViews.so.5.8.0
extract_planes: /usr/lib/libvtkCharts.so.5.8.0
extract_planes: /usr/lib/libpcl_common.so
extract_planes: /usr/lib/libpcl_kdtree.so
extract_planes: /usr/lib/libpcl_octree.so
extract_planes: /usr/lib/libpcl_search.so
extract_planes: /usr/lib/libpcl_surface.so
extract_planes: /usr/lib/libpcl_sample_consensus.so
extract_planes: /usr/lib/libpcl_io.so
extract_planes: /usr/lib/libpcl_filters.so
extract_planes: /usr/lib/libpcl_features.so
extract_planes: /usr/lib/libpcl_keypoints.so
extract_planes: /usr/lib/libpcl_registration.so
extract_planes: /usr/lib/libpcl_segmentation.so
extract_planes: /usr/lib/libpcl_recognition.so
extract_planes: /usr/lib/libpcl_visualization.so
extract_planes: /usr/lib/libpcl_people.so
extract_planes: /usr/lib/libpcl_outofcore.so
extract_planes: /usr/lib/libpcl_tracking.so
extract_planes: /usr/lib/libpcl_apps.so
extract_planes: /usr/lib/libvtkViews.so.5.8.0
extract_planes: /usr/lib/libvtkInfovis.so.5.8.0
extract_planes: /usr/lib/libvtkWidgets.so.5.8.0
extract_planes: /usr/lib/libvtkVolumeRendering.so.5.8.0
extract_planes: /usr/lib/libvtkHybrid.so.5.8.0
extract_planes: /usr/lib/libvtkParallel.so.5.8.0
extract_planes: /usr/lib/libvtkRendering.so.5.8.0
extract_planes: /usr/lib/libvtkImaging.so.5.8.0
extract_planes: /usr/lib/libvtkGraphics.so.5.8.0
extract_planes: /usr/lib/libvtkIO.so.5.8.0
extract_planes: /usr/lib/libvtkFiltering.so.5.8.0
extract_planes: /usr/lib/libvtkCommon.so.5.8.0
extract_planes: /usr/lib/libvtksys.so.5.8.0
extract_planes: CMakeFiles/extract_planes.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable extract_planes"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/extract_planes.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/extract_planes.dir/build: extract_planes
.PHONY : CMakeFiles/extract_planes.dir/build

CMakeFiles/extract_planes.dir/requires: CMakeFiles/extract_planes.dir/extract_planes.cpp.o.requires
.PHONY : CMakeFiles/extract_planes.dir/requires

CMakeFiles/extract_planes.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/extract_planes.dir/cmake_clean.cmake
.PHONY : CMakeFiles/extract_planes.dir/clean

CMakeFiles/extract_planes.dir/depend:
	cd /home/kalmar/PCL_Dachstuhl2016/extract_planes/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kalmar/PCL_Dachstuhl2016/extract_planes/build /home/kalmar/PCL_Dachstuhl2016/extract_planes/build /home/kalmar/PCL_Dachstuhl2016/extract_planes/build /home/kalmar/PCL_Dachstuhl2016/extract_planes/build /home/kalmar/PCL_Dachstuhl2016/extract_planes/build/CMakeFiles/extract_planes.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/extract_planes.dir/depend

