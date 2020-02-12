cmake_minimum_required(VERSION 2.8.3)
project(cuda_exploration)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

# https://stackoverflow.com/questions/25748039/add-cuda-to-ros-package

find_package(CUDA REQUIRED) 

 # set CUDA_NVCC_FLAGS as you would do with CXX/C FLAGS         
set(CUDA_NVCC_FLAGS CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE) 
set(LIB_TYPE STATIC)
cuda_add_library(vector_math_lib ${LIB_TYPE} src/vector_addition.cu)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES vector_math_lib
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(vector_math_node src/main.cpp)

add_dependencies(vector_math_node vector_math_lib)

TARGET_LINK_LIBRARIES(vector_math_node
	${catkin_LIBRARIES}
	vector_math_lib
)