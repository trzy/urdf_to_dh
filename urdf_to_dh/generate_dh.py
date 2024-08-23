# Copyright 2020 Andy McEvoy.
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

import argparse
import xml.etree.ElementTree as ET
from anytree import AnyNode, LevelOrderIter
from anytree import RenderTree
import numpy as np
import os
import pandas as pd
import pprint
import math


import urdf_to_dh.kinematics_helpers as kh
import urdf_to_dh.geometry_helpers as gh
import urdf_to_dh.urdf_helpers as uh


class GenerateDhParams:

    def __init__(self, filepath: str):
        self.urdf_joints = {}
        self.urdf_links = {}
        self.urdf_file = filepath
        self.urdf_tree_nodes = []
        self.root_link = None
        self.verbose = False


    def parse_urdf(self):
        # Get the root of the URDF and extract all of the joints
        urdf_root = uh.get_urdf_root(self.urdf_file)

        # Parse all links first and add to tree
        for child in urdf_root:
            if child.tag == 'link':
                self.urdf_links[child.get('name')] = {'rel_tf': np.eye(4), 'abs_tf': np.eye(4), 'dh_tf': np.eye(4), 'abs_dh_tf': np.eye(4), 'dh_found': False}
                node = AnyNode(id=child.get('name'), parent=None, children=None, type='link')
                self.urdf_tree_nodes.append(node)

        # Parse all joints and add to tree
        for child in urdf_root:
            if child.tag == 'joint':
                joint_name, joint_data = uh.process_joint(child)
                self.urdf_joints[joint_name] = joint_data
                node = AnyNode(id=joint_name, parent=None, children=None, type='joint')

                # Find parent and child link
                for n in self.urdf_tree_nodes:
                    if n.id == joint_data['parent']:
                        node.parent = n
                    if n.id == joint_data['child']:
                        n.parent = node
                self.urdf_tree_nodes.append(node)

        # Find root link
        num_nodes_no_parent = 0
        for n in self.urdf_tree_nodes:
            if n.parent == None:
                num_nodes_no_parent += 1
                self.root_link = n

        if num_nodes_no_parent == 1:
            # Root link DH will be identity, set dh_found = True
            # TODO: Probably not needed since order iter is used
            self.urdf_links[self.root_link.id]['dh_found'] = True
            print("URDF Tree:")
            for pre, _, node in RenderTree(self.root_link):
                print('%s%s' % (pre, node.id))

            print("Joint Info:")
            pprint.pprint(self.urdf_joints)
        else:
            print("Error: Should only be one root link")


    def produce_simplified_urdf(self, filepath: str):
        joints_by_id = {}               # original joint data for each joint id
        joint_tf_by_id = {}             # new joint transform by ID
        joint_ancestor_id_by_id = {}    # for each moving joint, which moving joint (if any) comes before it

        # Collapse fixed joints and preserve only moving joints
        for node in LevelOrderIter(self.root_link):
            if self.is_moving_joint(node):
                current_joint = node

                # Collapse the transforms of all the preceding fixed joints into this one so we can
                # eliminate them
                prior_fixed_joints = self.find_consecutive_ancestral_fixed_joints(current_joint)
                prior_fixed_joints.reverse()    # start from farthest ancestor so we can walk down the tree
                transform = np.eye(4)
                for fixed_joint in prior_fixed_joints:
                    transform = transform @ self.urdf_joints[fixed_joint.id]["tf"]
                transform = transform @ self.urdf_joints[current_joint.id]["tf"]

                # Find ancestral moving joint, if one exists
                ancestral_moving_joint = self.find_ancestral_moving_joint(current_joint)
                print(ancestral_moving_joint)

                # Store this moving joint
                joints_by_id[current_joint.id] = self.urdf_joints[current_joint.id]
                joint_tf_by_id[current_joint.id] = transform
                if ancestral_moving_joint is not None:
                    joint_ancestor_id_by_id[current_joint.id] = ancestral_moving_joint.id

        # Convert each transform into xyz and rpy parameters (axis shouldn't change because it is
        # *local* to the joint)
        joint_xyz_by_id = {}
        joint_rpy_by_id = {}
        for id, joint_data in joints_by_id.items():
            transform = joint_tf_by_id[id]
            xyz = transform[0:3,3]
            rpy = kh.rotation_matrix_to_rpy(transform[0:3,0:3])
            joint_xyz_by_id[id] = xyz
            joint_rpy_by_id[id] = rpy

        # Create a link following every joint
        child_link_name_by_joint_id = {}
        i = 1
        for id, joint_data in joints_by_id.items():
            child_link_name_by_joint_id[id] = f"link_{i}"
            i += 1

        # Assign a parent link to everyone. Root joint gets a parent link named "base_link"
        parent_link_name_by_joint_id = {}
        for id, joint_data in joints_by_id.items():
            is_root = id not in joint_ancestor_id_by_id
            if is_root:
                parent_link_name_by_joint_id[id] = "base_link"
            else:
                joint_ancestor_id = joint_ancestor_id_by_id[id]
                parent_link_name_by_joint_id[id] = child_link_name_by_joint_id[joint_ancestor_id]

        # All link
        all_link_names = set(list(parent_link_name_by_joint_id.values()) + list(child_link_name_by_joint_id.values()))

        # Produce the URDF file contents. Links first, then joints.
        contents = """<?xml version="1.0"?>
<robot name="robot">
"""
        for name in all_link_names:
            contents += f"  <link name=\"{name}\"></link>\n"
        for id, joint_data in joints_by_id.items():
            joint_type = joint_data["joint_type"]
            xyz = joint_xyz_by_id[id]
            rpy = joint_rpy_by_id[id]
            axis = self.urdf_joints[id]["axis"]
            limits = self.urdf_joints[id]["limits"]
            parent_link_name = parent_link_name_by_joint_id[id]
            child_link_name = child_link_name_by_joint_id[id]
            contents += f"  <joint name=\"{id}\" type=\"{joint_type}\">\n"
            contents += f"      <origin xyz=\"{xyz[0]} {xyz[1]} {xyz[2]}\" rpy=\"{rpy[0]} {rpy[1]} {rpy[2]}\" />\n"
            contents += f"      <axis xyz=\"{axis[0]} {axis[1]} {axis[2]}\" />\n"
            if limits is not None:
                contents += f"      <limit lower=\"{limits['lower']}\" upper=\"{limits['upper']}\" velocity=\"{limits['velocity']}\" effort=\"{limits['effort']}\" />\n"
            contents += f"      <parent link=\"{parent_link_name}\" />\n"
            contents += f"      <child link=\"{child_link_name}\" />\n"
            contents += f"  </joint>\n"

        contents += "</robot>"

        with open(filepath, "w") as fp:
            fp.write(contents)
        print(f"Wrote simplified URDF to: {filepath}")

    # Returns a list of direct ancestral fixed joints up to and excluding the first moving joint.
    # E.g., [ parent_joint, grandparent_joint, ... ]
    # Links are not included.
    def find_consecutive_ancestral_fixed_joints(self, node):
        ancestors = []
        next_node = node.parent
        while (next_node is not None) and (not self.is_moving_joint(next_node)):
            if self.is_fixed_joint(next_node):
                ancestors.append(next_node)
            next_node = next_node.parent
        print(f"Ancestral fixed joints of {node.id}: {[ n.id for n in ancestors ]}")
        return ancestors

    def find_ancestral_moving_joint(self, node) -> AnyNode | None:
        next_node = node.parent
        while next_node is not None:
            if self.is_moving_joint(next_node):
                return next_node
            next_node = next_node.parent
        return None

    def is_moving_joint(self, node):
        return node.type == "joint" and self.urdf_joints[node.id]["joint_type"] != "fixed"

    def is_fixed_joint(self, node):
        return node.type == "joint" and self.urdf_joints[node.id]["joint_type"] == "fixed"

    def calculate_tfs_in_world_frame(self):
        print("Calculate world tfs:")
        for n in LevelOrderIter(self.root_link):
            if n.type == 'link' and n.parent != None:
                print("\nget tf from ", n.parent.parent.id, " to ", n.id)
                parent_tf_world = self.urdf_links[n.parent.parent.id]['abs_tf']
                tf = self.urdf_joints[n.parent.id]['tf']
                self.urdf_links[n.id]['rel_tf'] = tf

                abs_tf = np.eye(4)
                abs_tf = np.matmul(parent_tf_world, tf)
                self.urdf_links[n.id]['abs_tf'] = abs_tf

        # print("Link Info:")
        # for link_name, link_data in self.urdf_links.items():
        #     print("\n=====", link_name)
        #     print("rel_tf")
        #     print(link_data['rel_tf'])
        #     print("abs_tf")
        #     print(link_data['abs_tf'])
        #     print("dh_tf")
        #     print(link_data['dh_tf'])
        #     print("abs_dh_tf")
        #     print(link_data['abs_dh_tf'])


    def calculate_dh_params(self):
        print("calculate_dh_params")
        # Node process order:
        print("process_order = \n", [urdf_node.id for urdf_node in LevelOrderIter(self.root_link)])
        robot_dh_params = []

        # First pass: compute DH params for all joints relative to their parents, regardless of
        # whether those parents are fixed joints or moveable joints
        for urdf_node in LevelOrderIter(self.root_link):
            if urdf_node.type == 'link' and self.urdf_links[urdf_node.id]['dh_found'] == False:
                print("\n\nprocess dh params for ", urdf_node.id)

                # TF from current link frame to world frame
                link_to_world = self.urdf_links[urdf_node.id]['abs_tf']

                # DH frame from parent link frame to world frame
                parent_to_world_dh = self.urdf_links[urdf_node.parent.parent.id]['abs_dh_tf']

                # TF from link frame to parent dh frame
                link_to_parent_dh = np.matmul(kh.inv_tf(parent_to_world_dh), link_to_world)

                # Find DH parameters
                # Publish Joint axis for visual verification
                #self.marker_pub.publish_arrow(urdf_node.id, np.zeros(3), self.urdf_joints[urdf_node.parent.id]['axis'], [1.0, 0.0, 1.0, 0.2])
                axis = np.matmul(link_to_parent_dh[0:3, 0:3], self.urdf_joints[urdf_node.parent.id]['axis'])

                dh_params = self.get_joint_dh_params(link_to_parent_dh, axis)

                dh_frame = kh.get_dh_frame(dh_params)
                abs_dh_frame = np.matmul(parent_to_world_dh, dh_frame)

                self.urdf_links[urdf_node.id]['dh_tf'] = dh_frame

                self.urdf_links[urdf_node.id]['abs_dh_tf'] = abs_dh_frame
                #self.marker_pub.publish_frame('world', abs_dh_frame)
                robot_dh_params.append([urdf_node.parent.id, urdf_node.parent.parent.id, urdf_node.id] + list(dh_params.round(5)))

        pd_frame = pd.DataFrame(robot_dh_params, columns=['joint', 'parent', 'child', 'd', 'theta', 'r', 'alpha'])
        pd_frame['theta'] = pd_frame['theta'] * 180.0 / math.pi
        pd_frame['alpha'] = pd_frame['alpha'] * 180.0 / math.pi
        print("\nDH Parameters: (csv)")
        print(pd_frame.to_csv())
        print("\nDH Parameters: (markdown)")
        print(pd_frame.to_markdown())


    def get_joint_dh_params(self, rel_link_frame, axis):
        dh_params = np.zeros(4)

        # Get the joint axis in the parent frame
        # for joint_name, joint_data in self.urdf_joints.items():
        #     print(joint_name)
        #     parent_tf_to_child_tf = kh.get_extrinsic_rotation(joint_data['rpy'])
        #     # print(parent_tf_to_child_tf)

        #     axis_in_parent_tf = np.matmul(parent_tf_to_child_tf, joint_data['axis'])
        #     self.publish_arrow(joint_data['parent'], joint_data['xyz'], axis_in_parent_tf)
        #     # print(axis_in_parent_tf)
        origin_xyz = rel_link_frame[0:3, 3]
        z_axis = np.array([0, 0, 1])
        print(axis)
        # Collinear case
        if gh.are_collinear(np.zeros(3), z_axis, origin_xyz, axis):
            print("  Process collinear case.")
            dh_params = self.process_collinear_case(origin_xyz, rel_link_frame[0:3, 0])
            # continue

        # Parallel case
        elif gh.are_parallel(z_axis, axis):
            print("  Process parallel case.")
            dh_params = self.process_parallel_case(origin_xyz)
            # continue

        # Intersect case
        elif gh.lines_intersect(np.zeros(3), z_axis, origin_xyz, axis)[0]:
            print("  Process intersection case.")
            print(rel_link_frame)
            dh_params = self.process_intersection_case(origin_xyz, axis)
            # continue

        # Skew case
        else:
            print("  Process skew case.")
            dh_params = self.process_skew_case(origin_xyz, axis)


        # Visualize the "d" component
        # self.publish_arrow(joint_data['parent'], np.zeros(3), pointA, 0.0, 0.0, 1.0, 0.5)

        # # Visualize the "r" component
        # self.publish_arrow(joint_data['parent'], pointA, pointB-pointA, 1.0, 0.0, 1.0, 1.0)

        # # Visualize the intersection and alignment with the next joint axis
        # self.publish_arrow(joint_data['parent'], pointB, joint_data['xyz']-pointB, 0.0, 1.0, 1.0, 0.5)
        print(dh_params)
        return dh_params

    def process_collinear_case(self, origin, xaxis):
        dh_params = np.zeros(4)
        dh_params[0] = origin[2]
        # dh_params[1] = math.atan2(xaxis[1], xaxis[0])
        return dh_params

    def process_parallel_case(self, origin):
        dh_params = np.zeros(4)
        dh_params[0] = origin[2]
        dh_params[1] = math.atan2(origin[1], origin[0])
        dh_params[2] = math.sqrt(origin[0]**2 + origin[1]**2)
        return dh_params

    def process_intersection_case(self, origin, axis):
        dh_params = np.zeros(4)
        solution = gh.lines_intersect(np.zeros(3), np.array([0, 0, 1]), origin, axis)[1][0]
        dh_params[0] = solution[0]

        zaxis = np.array([0., 0., 1.])
        xaxis = np.array([1., 0., 0.])

        for i in range(0,3):
            if abs(axis[i]) < 1.e-5:
                axis[i] = 0

        cn = np.cross(zaxis, axis)
        for i in range(0,3):
            if abs(cn[i]) < 1.e-6:
                cn[i] = 0
        if (cn[0] < 0):
            cn = cn * -1
        dh_params[1] = math.atan2(cn[1], cn[0])
        print(math.atan2(np.dot(np.cross(xaxis, cn), zaxis), np.dot(xaxis, cn)))

        dh_params[2] = 0

        vn = cn / np.linalg.norm(cn)
        dh_params[3] = math.atan2(np.dot(np.cross(zaxis, axis), vn), np.dot(zaxis, axis))

        return dh_params

    def process_skew_case(self, origin, direction):
        pointA = np.zeros(3)
        pointB = np.zeros(3)
        dh_params = np.zeros(4)

        # Find closest points along parent z-axis (pointA) and joint axis (pointB)
        t = -1.0 * (origin[0] * direction[0] + origin[1] * direction[1]) / (direction[0]**2 + direction[1]**2)
        pointB = origin + t * direction
        pointA[2] = pointB[2]

        # 'd' is offset along parent z axis
        dh_params[0] = pointA[2]

        # 'r' is the length of the common normal
        dh_params[2] = np.linalg.norm(pointB - pointA)

        # 'theta' is the angle between the x-axis and the common normal
        dh_params[1] = math.atan2(pointB[1], pointB[0])

        # 'alpha' is the angle between the current z-axis and the joint axis
        # Awesome way to get signed angle:
        # https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane/33920320#33920320
        cn = pointB - pointA
        vn = cn / np.linalg.norm(cn)
        zaxis = np.array([0, 0, 1])
        dh_params[3] = math.atan2(np.dot(np.cross(zaxis, direction), vn), np.dot(zaxis, direction))

        # print('points = ', pointA, pointB)
        # print('dh params = ', dh_params)
        return dh_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser("generate_dh")
    parser.add_argument("file", nargs="+")
    parser.add_argument("--simplified-urdf", metavar="filepath", action="store", type=str, help="Prune fixed joints and return a chain of mobile joints only")
    options = parser.parse_args()

    dh_generator = GenerateDhParams(filepath=options.file[0])
    dh_generator.parse_urdf()
    if options.simplified_urdf:
        dh_generator.produce_simplified_urdf(filepath=options.simplified_urdf)
        exit()
    dh_generator.calculate_tfs_in_world_frame()
    dh_generator.calculate_dh_params()
