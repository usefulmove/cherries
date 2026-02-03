(Google Drive: "Cherry Line Files/notes.txt")


avt vimba camera inferace

use this to start the camera node:
source /opt/ros/humble/setup.bash
source install/setup.bash <- do thise from ~/cherry_ws/ folder

		ros2 launch avt_vimba_camera Mako_G-507.launch.xml 

use this to view images published by the mono_camera_node:

		ros2 run image_view image_view --ros-args --remap image:=/image_color

settings for the camera can be modified in the Mako_G-507.launch.xml file found in:
~/user/cherry_ws/src/avt_vimba_camera/avt_vimba_camera/launch/
After modifiying, the project will need to be rebuilt at ~/cherry_ws/
There is probably a way to make modifications in the Build folder, but
those changes will go away if the project is rebuilt.  On the other hand,
you don;t have to rebuild - so maybe that is useful?

trigger node startup:

		ros2 launch avt_vimba_camera trigger_node.launch.xml

trigger settings:
    feature/ActionDeviceKey: 1
    feature/ActionGroupKey: 1
    feature/ActionGroupMask: 1
    
    firmware was updated to latest version on camera using utlity on allied vision's website.
    
    
    
PC ip address:
172.16.1.223/16

camera ip address:
172.16.2.2/16

https://superuser.com/questions/493319/can-i-have-graphics-on-linux-without-a-desktop-manager

https://www.cairographics.org/Xlib/





ros2 run --prefix 'gdb -ex run --args' tracking_projector tracking_projector 



ros2 run image_view image_view --ros-args --remap image:=/detection_server/keypoint_image_processed
ros2 run image_view image_view --ros-args --remap image:=/detection_server/keypoint_image_color

ros2 run image_view image_view --ros-args --remap image:=/single_acquisition_node/image_back
ros2 run image_view image_view --ros-args --remap image:=/single_acquisition_node/image_top


ros2 run image_view image_view --ros-args --remap image:=/detection_server/keypoint_image_processed
