# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

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
CMAKE_SOURCE_DIR = /home/hui/study/bow4book/DBow3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hui/study/bow4book/DBow3/build

# Include any dependencies generated for this target.
include src/CMakeFiles/DBoW3.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/DBoW3.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/DBoW3.dir/flags.make

src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o: src/CMakeFiles/DBoW3.dir/flags.make
src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o: ../src/FeatureVector.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hui/study/bow4book/DBow3/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DBoW3.dir/FeatureVector.cpp.o -c /home/hui/study/bow4book/DBow3/src/FeatureVector.cpp

src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW3.dir/FeatureVector.cpp.i"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hui/study/bow4book/DBow3/src/FeatureVector.cpp > CMakeFiles/DBoW3.dir/FeatureVector.cpp.i

src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW3.dir/FeatureVector.cpp.s"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hui/study/bow4book/DBow3/src/FeatureVector.cpp -o CMakeFiles/DBoW3.dir/FeatureVector.cpp.s

src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o.requires:
.PHONY : src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o.requires

src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o.provides: src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/DBoW3.dir/build.make src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o.provides.build
.PHONY : src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o.provides

src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o.provides.build: src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o

src/CMakeFiles/DBoW3.dir/BowVector.cpp.o: src/CMakeFiles/DBoW3.dir/flags.make
src/CMakeFiles/DBoW3.dir/BowVector.cpp.o: ../src/BowVector.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hui/study/bow4book/DBow3/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/DBoW3.dir/BowVector.cpp.o"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DBoW3.dir/BowVector.cpp.o -c /home/hui/study/bow4book/DBow3/src/BowVector.cpp

src/CMakeFiles/DBoW3.dir/BowVector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW3.dir/BowVector.cpp.i"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hui/study/bow4book/DBow3/src/BowVector.cpp > CMakeFiles/DBoW3.dir/BowVector.cpp.i

src/CMakeFiles/DBoW3.dir/BowVector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW3.dir/BowVector.cpp.s"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hui/study/bow4book/DBow3/src/BowVector.cpp -o CMakeFiles/DBoW3.dir/BowVector.cpp.s

src/CMakeFiles/DBoW3.dir/BowVector.cpp.o.requires:
.PHONY : src/CMakeFiles/DBoW3.dir/BowVector.cpp.o.requires

src/CMakeFiles/DBoW3.dir/BowVector.cpp.o.provides: src/CMakeFiles/DBoW3.dir/BowVector.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/DBoW3.dir/build.make src/CMakeFiles/DBoW3.dir/BowVector.cpp.o.provides.build
.PHONY : src/CMakeFiles/DBoW3.dir/BowVector.cpp.o.provides

src/CMakeFiles/DBoW3.dir/BowVector.cpp.o.provides.build: src/CMakeFiles/DBoW3.dir/BowVector.cpp.o

src/CMakeFiles/DBoW3.dir/quicklz.c.o: src/CMakeFiles/DBoW3.dir/flags.make
src/CMakeFiles/DBoW3.dir/quicklz.c.o: ../src/quicklz.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hui/study/bow4book/DBow3/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object src/CMakeFiles/DBoW3.dir/quicklz.c.o"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/DBoW3.dir/quicklz.c.o   -c /home/hui/study/bow4book/DBow3/src/quicklz.c

src/CMakeFiles/DBoW3.dir/quicklz.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/DBoW3.dir/quicklz.c.i"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/hui/study/bow4book/DBow3/src/quicklz.c > CMakeFiles/DBoW3.dir/quicklz.c.i

src/CMakeFiles/DBoW3.dir/quicklz.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/DBoW3.dir/quicklz.c.s"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/hui/study/bow4book/DBow3/src/quicklz.c -o CMakeFiles/DBoW3.dir/quicklz.c.s

src/CMakeFiles/DBoW3.dir/quicklz.c.o.requires:
.PHONY : src/CMakeFiles/DBoW3.dir/quicklz.c.o.requires

src/CMakeFiles/DBoW3.dir/quicklz.c.o.provides: src/CMakeFiles/DBoW3.dir/quicklz.c.o.requires
	$(MAKE) -f src/CMakeFiles/DBoW3.dir/build.make src/CMakeFiles/DBoW3.dir/quicklz.c.o.provides.build
.PHONY : src/CMakeFiles/DBoW3.dir/quicklz.c.o.provides

src/CMakeFiles/DBoW3.dir/quicklz.c.o.provides.build: src/CMakeFiles/DBoW3.dir/quicklz.c.o

src/CMakeFiles/DBoW3.dir/DescManip.cpp.o: src/CMakeFiles/DBoW3.dir/flags.make
src/CMakeFiles/DBoW3.dir/DescManip.cpp.o: ../src/DescManip.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hui/study/bow4book/DBow3/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/DBoW3.dir/DescManip.cpp.o"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DBoW3.dir/DescManip.cpp.o -c /home/hui/study/bow4book/DBow3/src/DescManip.cpp

src/CMakeFiles/DBoW3.dir/DescManip.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW3.dir/DescManip.cpp.i"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hui/study/bow4book/DBow3/src/DescManip.cpp > CMakeFiles/DBoW3.dir/DescManip.cpp.i

src/CMakeFiles/DBoW3.dir/DescManip.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW3.dir/DescManip.cpp.s"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hui/study/bow4book/DBow3/src/DescManip.cpp -o CMakeFiles/DBoW3.dir/DescManip.cpp.s

src/CMakeFiles/DBoW3.dir/DescManip.cpp.o.requires:
.PHONY : src/CMakeFiles/DBoW3.dir/DescManip.cpp.o.requires

src/CMakeFiles/DBoW3.dir/DescManip.cpp.o.provides: src/CMakeFiles/DBoW3.dir/DescManip.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/DBoW3.dir/build.make src/CMakeFiles/DBoW3.dir/DescManip.cpp.o.provides.build
.PHONY : src/CMakeFiles/DBoW3.dir/DescManip.cpp.o.provides

src/CMakeFiles/DBoW3.dir/DescManip.cpp.o.provides.build: src/CMakeFiles/DBoW3.dir/DescManip.cpp.o

src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o: src/CMakeFiles/DBoW3.dir/flags.make
src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o: ../src/QueryResults.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hui/study/bow4book/DBow3/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DBoW3.dir/QueryResults.cpp.o -c /home/hui/study/bow4book/DBow3/src/QueryResults.cpp

src/CMakeFiles/DBoW3.dir/QueryResults.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW3.dir/QueryResults.cpp.i"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hui/study/bow4book/DBow3/src/QueryResults.cpp > CMakeFiles/DBoW3.dir/QueryResults.cpp.i

src/CMakeFiles/DBoW3.dir/QueryResults.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW3.dir/QueryResults.cpp.s"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hui/study/bow4book/DBow3/src/QueryResults.cpp -o CMakeFiles/DBoW3.dir/QueryResults.cpp.s

src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o.requires:
.PHONY : src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o.requires

src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o.provides: src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/DBoW3.dir/build.make src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o.provides.build
.PHONY : src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o.provides

src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o.provides.build: src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o

src/CMakeFiles/DBoW3.dir/Database.cpp.o: src/CMakeFiles/DBoW3.dir/flags.make
src/CMakeFiles/DBoW3.dir/Database.cpp.o: ../src/Database.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hui/study/bow4book/DBow3/build/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/DBoW3.dir/Database.cpp.o"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DBoW3.dir/Database.cpp.o -c /home/hui/study/bow4book/DBow3/src/Database.cpp

src/CMakeFiles/DBoW3.dir/Database.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW3.dir/Database.cpp.i"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hui/study/bow4book/DBow3/src/Database.cpp > CMakeFiles/DBoW3.dir/Database.cpp.i

src/CMakeFiles/DBoW3.dir/Database.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW3.dir/Database.cpp.s"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hui/study/bow4book/DBow3/src/Database.cpp -o CMakeFiles/DBoW3.dir/Database.cpp.s

src/CMakeFiles/DBoW3.dir/Database.cpp.o.requires:
.PHONY : src/CMakeFiles/DBoW3.dir/Database.cpp.o.requires

src/CMakeFiles/DBoW3.dir/Database.cpp.o.provides: src/CMakeFiles/DBoW3.dir/Database.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/DBoW3.dir/build.make src/CMakeFiles/DBoW3.dir/Database.cpp.o.provides.build
.PHONY : src/CMakeFiles/DBoW3.dir/Database.cpp.o.provides

src/CMakeFiles/DBoW3.dir/Database.cpp.o.provides.build: src/CMakeFiles/DBoW3.dir/Database.cpp.o

src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o: src/CMakeFiles/DBoW3.dir/flags.make
src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o: ../src/Vocabulary.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hui/study/bow4book/DBow3/build/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DBoW3.dir/Vocabulary.cpp.o -c /home/hui/study/bow4book/DBow3/src/Vocabulary.cpp

src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW3.dir/Vocabulary.cpp.i"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hui/study/bow4book/DBow3/src/Vocabulary.cpp > CMakeFiles/DBoW3.dir/Vocabulary.cpp.i

src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW3.dir/Vocabulary.cpp.s"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hui/study/bow4book/DBow3/src/Vocabulary.cpp -o CMakeFiles/DBoW3.dir/Vocabulary.cpp.s

src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o.requires:
.PHONY : src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o.requires

src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o.provides: src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/DBoW3.dir/build.make src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o.provides.build
.PHONY : src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o.provides

src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o.provides.build: src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o

src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o: src/CMakeFiles/DBoW3.dir/flags.make
src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o: ../src/ScoringObject.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hui/study/bow4book/DBow3/build/CMakeFiles $(CMAKE_PROGRESS_8)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DBoW3.dir/ScoringObject.cpp.o -c /home/hui/study/bow4book/DBow3/src/ScoringObject.cpp

src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW3.dir/ScoringObject.cpp.i"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hui/study/bow4book/DBow3/src/ScoringObject.cpp > CMakeFiles/DBoW3.dir/ScoringObject.cpp.i

src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW3.dir/ScoringObject.cpp.s"
	cd /home/hui/study/bow4book/DBow3/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hui/study/bow4book/DBow3/src/ScoringObject.cpp -o CMakeFiles/DBoW3.dir/ScoringObject.cpp.s

src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o.requires:
.PHONY : src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o.requires

src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o.provides: src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/DBoW3.dir/build.make src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o.provides.build
.PHONY : src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o.provides

src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o.provides.build: src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o

# Object files for target DBoW3
DBoW3_OBJECTS = \
"CMakeFiles/DBoW3.dir/FeatureVector.cpp.o" \
"CMakeFiles/DBoW3.dir/BowVector.cpp.o" \
"CMakeFiles/DBoW3.dir/quicklz.c.o" \
"CMakeFiles/DBoW3.dir/DescManip.cpp.o" \
"CMakeFiles/DBoW3.dir/QueryResults.cpp.o" \
"CMakeFiles/DBoW3.dir/Database.cpp.o" \
"CMakeFiles/DBoW3.dir/Vocabulary.cpp.o" \
"CMakeFiles/DBoW3.dir/ScoringObject.cpp.o"

# External object files for target DBoW3
DBoW3_EXTERNAL_OBJECTS =

src/libDBoW3.so.0.0.1: src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o
src/libDBoW3.so.0.0.1: src/CMakeFiles/DBoW3.dir/BowVector.cpp.o
src/libDBoW3.so.0.0.1: src/CMakeFiles/DBoW3.dir/quicklz.c.o
src/libDBoW3.so.0.0.1: src/CMakeFiles/DBoW3.dir/DescManip.cpp.o
src/libDBoW3.so.0.0.1: src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o
src/libDBoW3.so.0.0.1: src/CMakeFiles/DBoW3.dir/Database.cpp.o
src/libDBoW3.so.0.0.1: src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o
src/libDBoW3.so.0.0.1: src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o
src/libDBoW3.so.0.0.1: src/CMakeFiles/DBoW3.dir/build.make
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_xphoto.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_xobjdetect.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_tracking.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_surface_matching.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_structured_light.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_stereo.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_sfm.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_saliency.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_rgbd.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_reg.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_plot.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_optflow.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_line_descriptor.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_hdf.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_fuzzy.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_dpm.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_dnn.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_datasets.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_ccalib.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_bioinspired.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_bgsegm.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_aruco.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_viz.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_videostab.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_superres.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_stitching.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_photo.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudastereo.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudaoptflow.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudaobjdetect.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudalegacy.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudaimgproc.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudafeatures2d.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudacodec.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudabgsegm.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_text.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_face.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_ximgproc.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_xfeatures2d.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_shape.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudawarping.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_objdetect.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudafilters.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudaarithm.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_calib3d.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_features2d.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_ml.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_highgui.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_videoio.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_flann.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_video.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_imgproc.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_core.so.3.1.0
src/libDBoW3.so.0.0.1: /usr/local/lib/libopencv_cudev.so.3.1.0
src/libDBoW3.so.0.0.1: src/CMakeFiles/DBoW3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library libDBoW3.so"
	cd /home/hui/study/bow4book/DBow3/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DBoW3.dir/link.txt --verbose=$(VERBOSE)
	cd /home/hui/study/bow4book/DBow3/build/src && $(CMAKE_COMMAND) -E cmake_symlink_library libDBoW3.so.0.0.1 libDBoW3.so.0.0 libDBoW3.so

src/libDBoW3.so.0.0: src/libDBoW3.so.0.0.1
	@$(CMAKE_COMMAND) -E touch_nocreate src/libDBoW3.so.0.0

src/libDBoW3.so: src/libDBoW3.so.0.0.1
	@$(CMAKE_COMMAND) -E touch_nocreate src/libDBoW3.so

# Rule to build all files generated by this target.
src/CMakeFiles/DBoW3.dir/build: src/libDBoW3.so
.PHONY : src/CMakeFiles/DBoW3.dir/build

src/CMakeFiles/DBoW3.dir/requires: src/CMakeFiles/DBoW3.dir/FeatureVector.cpp.o.requires
src/CMakeFiles/DBoW3.dir/requires: src/CMakeFiles/DBoW3.dir/BowVector.cpp.o.requires
src/CMakeFiles/DBoW3.dir/requires: src/CMakeFiles/DBoW3.dir/quicklz.c.o.requires
src/CMakeFiles/DBoW3.dir/requires: src/CMakeFiles/DBoW3.dir/DescManip.cpp.o.requires
src/CMakeFiles/DBoW3.dir/requires: src/CMakeFiles/DBoW3.dir/QueryResults.cpp.o.requires
src/CMakeFiles/DBoW3.dir/requires: src/CMakeFiles/DBoW3.dir/Database.cpp.o.requires
src/CMakeFiles/DBoW3.dir/requires: src/CMakeFiles/DBoW3.dir/Vocabulary.cpp.o.requires
src/CMakeFiles/DBoW3.dir/requires: src/CMakeFiles/DBoW3.dir/ScoringObject.cpp.o.requires
.PHONY : src/CMakeFiles/DBoW3.dir/requires

src/CMakeFiles/DBoW3.dir/clean:
	cd /home/hui/study/bow4book/DBow3/build/src && $(CMAKE_COMMAND) -P CMakeFiles/DBoW3.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/DBoW3.dir/clean

src/CMakeFiles/DBoW3.dir/depend:
	cd /home/hui/study/bow4book/DBow3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hui/study/bow4book/DBow3 /home/hui/study/bow4book/DBow3/src /home/hui/study/bow4book/DBow3/build /home/hui/study/bow4book/DBow3/build/src /home/hui/study/bow4book/DBow3/build/src/CMakeFiles/DBoW3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/DBoW3.dir/depend

