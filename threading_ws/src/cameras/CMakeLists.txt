cmake_minimum_required(VERSION 3.8)
project(cameras)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find Vimba SDK lib
#
# exports:
#
#   Vimba_FOUND
#   Vimba_INCLUDE_DIRS
#   Vimba_LIBRARIES
#


set(Vimba_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include/VimbaX_2023-4)

# # Include dir
# find_path(Vimba_INCLUDE_DIR 
#     NAMES VmbCPP.h
#     PATHS ${Vimba_ROOT}/api/include/VmbCPP
# )

# # The library itself
# find_library(Vimba_LIBRARY
#   NAMES libVmbCPP.so
#   PATHS ${Vimba_ROOT}/api/lib
# )

# # Library for image transformation
# find_library(Vimba_TRANSFORM
#   NAMES libVmbImageTransform.so
#   PATHS ${Vimba_ROOT}/api/lib
# )

# Library for image transformation
find_path(Vimba_CONFIGS
  NAMES vmb-config.cmake
  PATHS ${Vimba_ROOT}/api/lib/cmake/vmb
)


# set( VMB_PATH ${Vimba_ROOT}api/lib/cmake/vmb)
# set( VMB_CONFIGS ${Vimba_ROOT}api/lib/cmake/vmb/configs)
message("root : ${Vimba_ROOT}")
message("Vimba_CONFIGS: (${Vimba_CONFIGS}")
# message("Vimba_LIBRARY: (${Vimba_LIBRARY}")
# message("Vimba_INCLUDE_DIR: (${Vimba_INCLUDE_DIR}")

# list(APPEND CMAKE_PREFIX_PATH "${Vimba_CONFIGS}")
set (Vmb_DIR ${Vimba_CONFIGS})


find_package(Vmb REQUIRED COMPONENTS CPP NAMES Vmb VmbC VmbCPP VmbImageTransform)

# # Set the cmake variable
# set(Vimba_LIBRARIES ${Vimba_LIBRARY} ${Vimba_TRANSFORM})
# set(Vimba_INCLUDE_DIRS ${Vimba_INCLUDE_DIR} ${Vimba_ROOT})
# set(Vimba_FOUND yes)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cherry_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# include_directories(${Vimba_INCLUDE_DIRS})


add_executable(
  cognex_cam 
  src/main.cpp  
  src/cognex_cam.cpp
  src/FrameObserver.cpp
  src/EventObserver.cpp
)
add_executable(
  cognex_hdr
  src/mainhdr.cpp  
  src/cognex_hdr.cpp
  src/FrameObserver.cpp
  src/EventObserver.cpp
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/cameras)

target_include_directories(cognex_cam PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )


ament_target_dependencies(cognex_cam rclcpp rclcpp_action cherry_interfaces sensor_msgs)
ament_target_dependencies(cognex_hdr rclcpp rclcpp_action cherry_interfaces sensor_msgs)

# add_library( vimba_lib   
#   ${VIMBA_LIB}/libVmbCPP.so
#   ${VIMBA_LIB}/libVmbC.so
#   ${VIMBA_LIB}/libVmbImageTransform.so
# )

target_link_libraries(cognex_cam Vmb::CPP )
target_link_libraries(cognex_hdr Vmb::CPP )



install(TARGETS cognex_cam cognex_hdr
  DESTINATION lib/${PROJECT_NAME})

# install(FILES
#   ${Vimba_TRANSFORM}
#   ${Vimba_LIBRARY}
#   DESTINATION /usr/lib
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
