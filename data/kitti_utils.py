#-*-coding:utf-8-*- 
"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 2019/5/14 下午5:23 
  description: some utils for processing kitti data
"""
import numpy as np


# transform euler angle to rotation matrix
def rot_x(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def rot_y(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rot_z(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


# Transform the data of oxts to pose 4*4
def pose_from_oxts_packet(metadata, scale):

    lat, lon, alt, roll, pitch, yaw = metadata
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    Taken from https://github.com/utiasSTARS/pykitti
    """

    er = 6378137.  # earth radius (approx.) in meters
    # Use a Mercator projection to get the translation vector
    ty = lat * np.pi * er / 180.

    tx = scale * lon * np.pi * er / 180.
    # ty = scale * er * \
    #     np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz]).reshape(-1,1)

    # Use the Euler angles to get the rotation matrix
    Rx = rot_x(roll)
    Ry = rot_y(pitch)
    Rz = rot_z(yaw)
    R = Rz.dot(Ry.dot(Rx))
    return transform_from_rot_trans(R, t)


# read calibration files, return dictionary
def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


# concatenate rotation matrix and translation matrix 3*3+3*1->4*4
def transform_from_rot_trans(R, t):
    """Transformation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


# Get the P_rect of cam2cam
def get_p_rect(cam2cam,c,zoom_x,zoom_y):
    P_rect = cam2cam['P_rect_'+str(c)].reshape(3,4)
    P_rect[0]*=zoom_x
    P_rect[1]*=zoom_y
    return P_rect


