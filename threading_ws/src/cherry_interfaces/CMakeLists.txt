cmake_minimum_required(VERSION 3.8)
project(cherry_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)        
# find_package(sensor_msgs REQUIRED)
# find_package(std_msgs REQUIRED)
# find_package(message_generation REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(rosidl_default_generators REQUIRED)



set(msg_files
  "msg/ImageSet.msg"
  "msg/ImageLayer.msg"
  "msg/ImageSetHdr.msg"
  "msg/Cherry.msg"
  "msg/CherryArray.msg"
  "msg/CherryArrayStamped.msg"
  "msg/EncoderCount.msg"
  "msg/Inputs.msg"
  "msg/Outputs.msg"
  "msg/HSC.msg"
  "msg/Temperature.msg"
  "msg/Trigger.msg"
  "msg/PickMode.msg"
)
set(srv_files
  "srv/CombineImages.srv"
  "srv/SetLights.srv"
  "srv/Detection.srv"
  "srv/Detectionhdr.srv"
  "srv/ImageSave.srv"
  "srv/EncoderCount.srv"
  "srv/SetLatch.srv"
  "srv/EncoderLatches.srv"
  "srv/ResetLatches.srv"
  "srv/GetCherryBuffer.srv"
  "srv/LatchRobot.srv"
)

set(action_files
  "action/Acquisition.action"
  "action/Acquisitionhdr.action"
  "action/FindCherries.action"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ImageSet.msg"
  "msg/ImageLayer.msg"
  "msg/ImageSetHdr.msg"
  "msg/Cherry.msg"
  "msg/CherryArray.msg"
  "msg/CherryArrayStamped.msg"
  "msg/EncoderCount.msg"
  "msg/Inputs.msg"
  "msg/Outputs.msg"
  "msg/HSC.msg"
  "msg/Temperature.msg"
  "msg/Trigger.msg"
  "msg/PickMode.msg"
  "srv/CombineImages.srv"
  "srv/SetLights.srv"
  "srv/Detection.srv"
  "srv/Detectionhdr.srv"
  "srv/ImageSave.srv"
  "srv/EncoderCount.srv"
  "srv/EncoderLatches.srv"
  "srv/SetLatch.srv"
  "srv/ResetLatches.srv"
  "srv/ImageCounts.srv"
  "srv/Trigger.srv"
  "srv/LatchRobot.srv"
  "srv/GetCherryBuffer.srv"
  "action/Acquisition.action"
  "action/Acquisitionhdr.action"
  "action/FindCherries.action"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

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
