from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Node(
        #     package='image_combine_python',
        #     # namespace='turtlesim1',
        #     executable='combiner',
        #     # name='sim'
        #     parameters=[
        #         {"img_scaling": 0.590909090909091}
        #     ]
        # ),
        Node(
            package='cherry_detection',
            executable='detection_service',
        ),
        Node(
            package='cameras',
            executable='cognex_hdr',
        ),
        Node(
            package='plc_eip',
            executable='plc_eip',
        ),
        Node(
            package='cherry_buffer',
            executable='cherry_buffer',
        ),
        Node(
            package='image_service',
            executable='image_services',
        ),
        Node(
            package='system_monitor',
            executable='temperature_node',
        ),
        Node(
            package='trigger_node',
            executable='trigger_node',
        ),
        Node(
            package='tracking_projector',
            executable='tracking_projector',
            name='tracking_projector',
            # parameters=[
            #     {"x": 2.45},
            #     {"y": -0.31},
            #     {"scaling_factor": 839.0},
            #     {"rotation": 0.0},
            #     {"screen": 2}
            # ]
        ),
        Node(
            package='fanuc_comms',
            executable='fanuc_comms',
        ),
    ])