# Copyright 2020 Andy McEvoy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

package_name = 'urdf_to_dh'
robot_name = 'random'
urdf_filename = 'random.urdf'
world_tf_arg = ['0', '0', '0', '0', '0', '0', 'world', 'link0']


def generate_launch_description():
    bringup_dir = get_package_share_directory(package_name)
    rviz_filename = 'test_urdf.rviz'
    rviz_config_file = ['-d', os.path.join(bringup_dir, rviz_filename)]
    print(rviz_config_file)

    # Publish the static World coordinate system
    world_tf_cmd = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=world_tf_arg
    )

    # Robot planning state publisher
    urdf_file = os.path.join(bringup_dir, 'urdf', urdf_filename)
    print(urdf_file)
    robot_state_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        arguments=[urdf_file],
    )

    # Joint state publisher
    joint_state_cmd = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen'
    )

    # RViz
    rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=rviz_config_file,
        output='screen'
    )

    return LaunchDescription([
        world_tf_cmd,
        robot_state_cmd,
        joint_state_cmd,
        rviz_cmd,
    ])
