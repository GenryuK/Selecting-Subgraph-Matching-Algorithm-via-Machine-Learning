# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv

# Include any dependencies generated for this target.
include graph/CMakeFiles/graph.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include graph/CMakeFiles/graph.dir/compiler_depend.make

# Include the progress variables for this target.
include graph/CMakeFiles/graph.dir/progress.make

# Include the compile flags for this target's objects.
include graph/CMakeFiles/graph.dir/flags.make

graph/CMakeFiles/graph.dir/graph.cpp.o: graph/CMakeFiles/graph.dir/flags.make
graph/CMakeFiles/graph.dir/graph.cpp.o: ../graph/graph.cpp
graph/CMakeFiles/graph.dir/graph.cpp.o: graph/CMakeFiles/graph.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object graph/CMakeFiles/graph.dir/graph.cpp.o"
	cd /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv/graph && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT graph/CMakeFiles/graph.dir/graph.cpp.o -MF CMakeFiles/graph.dir/graph.cpp.o.d -o CMakeFiles/graph.dir/graph.cpp.o -c /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/graph/graph.cpp

graph/CMakeFiles/graph.dir/graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/graph.dir/graph.cpp.i"
	cd /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv/graph && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/graph/graph.cpp > CMakeFiles/graph.dir/graph.cpp.i

graph/CMakeFiles/graph.dir/graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/graph.dir/graph.cpp.s"
	cd /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv/graph && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/graph/graph.cpp -o CMakeFiles/graph.dir/graph.cpp.s

# Object files for target graph
graph_OBJECTS = \
"CMakeFiles/graph.dir/graph.cpp.o"

# External object files for target graph
graph_EXTERNAL_OBJECTS =

graph/libgraph.so: graph/CMakeFiles/graph.dir/graph.cpp.o
graph/libgraph.so: graph/CMakeFiles/graph.dir/build.make
graph/libgraph.so: graph/CMakeFiles/graph.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libgraph.so"
	cd /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv/graph && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/graph.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
graph/CMakeFiles/graph.dir/build: graph/libgraph.so
.PHONY : graph/CMakeFiles/graph.dir/build

graph/CMakeFiles/graph.dir/clean:
	cd /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv/graph && $(CMAKE_COMMAND) -P CMakeFiles/graph.dir/cmake_clean.cmake
.PHONY : graph/CMakeFiles/graph.dir/clean

graph/CMakeFiles/graph.dir/depend:
	cd /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/graph /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv/graph /home/kuraya/SubgraphMatchingSurvey-1/PILOS/PILOS2/SubgraphMatchingSurvey/vlabel/buildv/graph/CMakeFiles/graph.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : graph/CMakeFiles/graph.dir/depend

