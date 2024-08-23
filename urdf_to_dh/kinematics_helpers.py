import numpy as np
import math

# Kinematics helper functions
def x_rotation(theta):
    """The 3x3 rotation matrix for a rotation of `theta` radians about the x-axis."""
    return np.array([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]])


def y_rotation(theta):
    """The 3x3 rotation matrix for a rotation of `theta` radians about the y-axis."""
    return np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]])


def z_rotation(theta):
    """The 3x3 rotation matrix for a rotation of `theta` radians about the z-axis."""
    return np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]])


def get_extrinsic_rotation(rpy):
    """Gets the extrinsic rotation matrix defined by roll about x, then pitch about y, then yaw
    about z. This is the rotation matrix used in URDF.

    Args:
        rpy: A numpy array containing the roll, pitch, and yaw angles in radians.

    Returns:
        A 3x3 numpy array for the resulting extrinsic rotation.
    """

    x_rot = x_rotation(rpy[0])
    y_rot = y_rotation(rpy[1])
    z_rot = z_rotation(rpy[2])
    return np.matmul(z_rot, np.matmul(y_rot, x_rot))

def inv_tf(tf):
    """Get the inverse of a homogeneous transform"""
    inv_tf = np.eye(4)
    inv_tf[0:3, 0:3] = np.transpose(tf[0:3, 0:3])
    inv_tf[0:3, 3] = -1.0 * np.matmul(np.transpose(tf[0:3, 0:3]), tf[0:3, 3])
    return inv_tf

def get_dh_frame(dh_params):
    """Get the tf for the given dh parameters."""
    d = dh_params[0]
    theta = dh_params[1]
    r = dh_params[2]
    alpha = dh_params[3]

    dh_frame = np.eye(4)

    dh_frame[0, 0] = math.cos(theta)
    dh_frame[0, 1] = -math.sin(theta) * math.cos(alpha)
    dh_frame[0, 2] = math.sin(theta) * math.sin(alpha)
    dh_frame[0, 3] = r * math.cos(theta)

    dh_frame[1, 0] = math.sin(theta)
    dh_frame[1, 1] = math.cos(theta) * math.cos(alpha)
    dh_frame[1, 2] = -math.cos(theta) * math.sin(alpha)
    dh_frame[1, 3] = r * math.sin(theta)

    dh_frame[2, 1] = math.sin(alpha)
    dh_frame[2, 2] = math.cos(alpha)
    dh_frame[2, 3] = d
    return dh_frame


def rotation_matrix_to_rpy(R):
    # https://learnopencv.com/rotation-matrix-to-euler-angles/

    # Check if it really is a rotation matrix
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    assert n < 1e-6

    # Compute rpy vector
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

