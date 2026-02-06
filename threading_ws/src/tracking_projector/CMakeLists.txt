cmake_minimum_required(VERSION 3.5)

project(tracking_projector VERSION 0.1 LANGUAGES CXX)
add_compile_options(-g)


set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Qt6 REQUIRED COMPONENTS Core Gui OpenGL OpenGLWidgets Widgets)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cherry_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(
 include/tracking_projector
 ${rclcpp_INCLUDE_DIRS}
 ${std_msgs_INCLUDE_DIRS}
 )


file(GLOB SOURCE_FILES
 “src/*.cpp”
)

add_executable(tracking_projector 
src/main.cpp
src/tracker.cpp
include/tracking_projector/tracker.h
include/tracking_projector/helper.h
src/helper.cpp
src/cherry_cpp.cpp
include/tracking_projector/cherry_cpp.hpp
include/tracking_projector/frame.h
src/frame.cpp
include/tracking_projector/conveyor.h
src/conveyor.cpp
include/tracking_projector/history.h
src/history.cpp
include/tracking_projector/tracker_node.h
src/tracker_node.cpp
)

target_link_libraries(tracking_projector
    Qt::Core
    Qt::Gui
    Qt::OpenGL
    Qt::OpenGLWidgets
    Qt::Widgets)

ament_target_dependencies(tracking_projector
 cherry_interfaces
 rclcpp
 std_msgs
 OpenCV
 )


install(TARGETS tracking_projector
  DESTINATION lib/${PROJECT_NAME})

ament_package()
