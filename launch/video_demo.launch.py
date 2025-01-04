import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix

def generate_launch_description():

    package_name = "nomadrt"

    try:
        package_path = get_package_prefix(package_name)
    except KeyError:
        raise RuntimeError(f"Please install the package '{package_name}'")

    weights_folder = os.path.join(package_path, 'share', package_name, 'weights')

    return LaunchDescription([

        Node(
            package='nomadrt',
            namespace='video_demo',
            executable='test_video_node.py',
            name='publish_video'
        ),

        Node(
            package='nomadrt',
            namespace='video_demo',
            executable='receiver_node.py',
            name='receive_video',
            parameters=[
                {"action_path": os.path.join(weights_folder, 'action.engine') },
                {"encoder_path": os.path.join(weights_folder, 'encoder.engine') },
                {"distance_path": os.path.join(weights_folder, 'distance.engine') }
            ]
        )

    ])