cmake_minimum_required(VERSION 3.0.2)
project(detector_try)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

find_package(OpenCV REQUIRED)

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
