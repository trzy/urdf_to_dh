import xml.etree.ElementTree as ET
import numpy as np

from . import kinematics_helpers as kh

# Helper functions for parsing the URDF
def get_urdf_root(urdf_file):
    """Parse a URDF for joints.

    Args:
        urdf_path (string): The absolute path to the URDF to be analyzed.

    Returns:
        root (xml object): root node of the URDF.
    """
    try:
        tree = ET.parse(urdf_file)
    except ET.ParseError:
        print('ERROR: Could not parse urdf file.')

    return tree.getroot()

def process_joint(joint):
    """Extracts the relevant joint info into a dictionary.
    Args:
    Returns:
    """
    axis = np.array([1, 0, 0])
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    parent_link = ''
    child_link = ''

    joint_name = joint.get('name')

    limits = None
    for child in joint:
        if child.tag == 'axis':
            axis = np.array(child.get('xyz').split(), dtype=float)
        elif child.tag == 'origin':
            xyz_tag = child.get('xyz')
            rpy_tag = child.get('rpy')
            xyz_tag = "0 0 0" if xyz_tag is None else xyz_tag
            rpy_tag = "0 0 0" if rpy_tag is None else rpy_tag
            xyz = np.array(xyz_tag.split(), dtype=float)
            rpy = np.array(rpy_tag.split(), dtype=float)
        elif child.tag == 'limit':
            lower_limit = child.get('lower')
            upper_limit = child.get('upper')
            effort = child.get('effort')
            velocity = child.get('velocity')
            lower_limit = lower_limit if not None else 0
            upper_limit = upper_limit if not None else 0
            effort = effort if not None else 0
            velocity = velocity if not None else 0
            limits = { 'lower': lower_limit, 'upper': upper_limit, 'effort': effort, 'velocity': velocity }
        elif child.tag == 'parent':
            parent_link = child.get('link')
        elif child.tag == 'child':
            child_link = child.get('link')


    tf = np.eye(4)
    tf[0:3, 0:3] = kh.get_extrinsic_rotation(rpy)
    tf[0:3, 3] = xyz

    return joint_name, {'axis': axis, 'tf': tf, 'parent': parent_link, 'child': child_link, 'dh': np.zeros(4), 'joint_type': joint.get('type'), 'limits': limits }
