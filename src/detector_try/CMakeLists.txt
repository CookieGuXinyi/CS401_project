cmake_minimum_required(VERSION 3.0.2)
project(detector_try)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

# 强制使用系统的OpenCV
set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
find_package(OpenCV 4.2 REQUIRED)

# 设置环境变量以使用系统的库
set(ENV{LD_LIBRARY_PATH} "/usr/lib/x86_64-linux-gnu:$ENV{LD_LIBRARY_PATH}")

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs cv_bridge image_transport
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(detector_node src/detector_node.cpp)
target_link_libraries(detector_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
