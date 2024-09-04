# This file is meant to hold functions used in the other sections of code within this directory
#
# creator: thomas stevens
# created: 2024/08/13

import time
import numpy as np
import yaml


def write_poni(fn, name, pixel1, pixel2, max_shape, distance, PONI1, PONI2, rot1, rot2, rot3, wavelength, version=2.1,
               orientation=3):
    # Writes a new .poni file from a .yml heXRD format file.
    txt = ["# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis",
           f"# Calibration converted from heXRD format at {time.ctime()}",
           f"poni_version: {version}",
           f"Detector: {name}",
           f'Detector_config: '
           f'{{"pixel1": {pixel1}, "pixel2": {pixel2}, "max_shape": {max_shape}, "orientation": {orientation}}}',
           f"Distance: {distance}",
           f"Poni1: {PONI1}",
           f"Poni2: {PONI2}",
           f"Rot1: {rot1}",
           f"Rot2: {rot2}",
           f"Rot3: {rot3}",
           f"Wavelength: {wavelength}", ""]
    fn.write("\n".join(txt))


def read_hexrd(file):
    read = yaml.load(file, Loader=yaml.FullLoader)
    energy = read['beam']['energy']
    detectors = list(read['detectors'].keys())
    data = {'energy': energy}
    for detector_name in detectors:
        data[detector_name] = {}
        data[detector_name]['location'] = np.array(read['detectors'][detector_name]['transform']['translation'], dtype=float)
        data[detector_name]['rotation'] = np.array(read['detectors'][detector_name]['transform']['tilt'], dtype=float)
        data[detector_name]['detector_shape'] = np.array([read['detectors'][detector_name]['pixels']['columns'], read['detectors'][detector_name]['pixels']['rows']], dtype=float)
        data[detector_name]['pixel_size'] = np.array(read['detectors'][detector_name]['pixels']['size'], dtype=float)
    return data


def hexrd_rotation_matrix(rotation_array):
    """ The rotation order is X -> Y -> Z, I believe this matches the rotation array order.
    All rotations are right-handed.
    The heXRD lab coordinates map to the pyFAI coordinates as follows:
    Z = -X_3
    Y = X_1
    X = X_2
    """
    cosrotx = np.cos(rotation_array[0])
    sinrotx = np.sin(rotation_array[0])
    cosroty = np.cos(rotation_array[1])
    sinroty = np.sin(rotation_array[1])
    cosrotz = np.cos(rotation_array[2])
    sinrotz = np.sin(rotation_array[2])

    rotz = np.array([[cosrotz, -sinrotz, 0],
                     [sinrotz, cosrotz, 0],
                     [0, 0, 1]])

    roty = np.array([[cosroty, 0, sinroty],
                     [0, 1, 0],
                     [-sinroty, 0, cosroty]])

    rotx = np.array([[1, 0, 0],
                     [0, cosrotx, -sinrotx],
                     [0, sinrotx, cosrotx]])

    rotation_matrix = np.dot(np.dot(rotz, roty), rotx)
    # order is Z, Y, X because matrices

    return rotation_matrix


def hexrd_to_pyfai_params(params):
    # Changes local rotation and lab coordinates of the centre of the detector to a point of normal incidence location in detector coordinates and a pyFAI rotation
    # find unit vectors of rotated axis in the hexrd lab frame (done)  ->  transform to pyFAI coords (done)  ->  find rotation matrix in pyFAI coords (done)
    # rotate detector location vector using inverse pyFAI rotation (done)  ->  move to detector corner (done)  ->  this coordinate is [-poni1, -poni2, dist] in pyFAI coords (done)
    rotation_matrix_hexrd = hexrd_rotation_matrix(params['VAREX1']['rotation'])
    hexrd_to_pyfai_mat = np.array([[0, 1, 0],
                                   [1, 0, 0],
                                   [0, 0, -1]])
    rotation_matrix_pyfai = np.dot(np.dot(hexrd_to_pyfai_mat, rotation_matrix_hexrd), hexrd_to_pyfai_mat)
    theta2 = np.arcsin(rotation_matrix_pyfai[2, 0])
    cos_theta2 = np.cos(theta2)
    theta1 = np.arccos(np.around(rotation_matrix_pyfai[2, 2] / cos_theta2, 8))
    theta3 = np.arcsin(np.around(rotation_matrix_pyfai[1, 0] / cos_theta2, 8))

    inverse_rot = np.linalg.inv(rotation_matrix_pyfai)
    detector_centre = np.dot(hexrd_to_pyfai_mat, np.array(params['VAREX1']['location'])) / 1000
    detector_centre_un_rotated = np.dot(inverse_rot, detector_centre)
    centre_to_corner = np.array([-params['VAREX1']['pixel_size'][1] * params['VAREX1']['detector_shape'][1] / 2000,
                                 -params['VAREX1']['pixel_size'][0] * params['VAREX1']['detector_shape'][0] / 2000,
                                 0])  # Axis 1 in pyFAI is vertical which is axis 2 in heXRD
    detector_corner_un_rotated = detector_centre_un_rotated + centre_to_corner
    poni1 = -detector_corner_un_rotated[0]
    poni2 = -detector_corner_un_rotated[1]
    distance = detector_corner_un_rotated[2]
    return distance, poni1, poni2, theta1, theta2, theta3


def pyfai_rotation_matrix(rot1, rot2, rot3):
    """ The rotation order is 1 -> 2 -> 3.
    Rotation 3 is right-handed, Rotations 1 and 2 are left-handed.
    The heXRD lab coordinates map to the pyFAI coordinates as follows:
    Z = -X_3
    Y = X_1
    X = X_2
    """
    cosrot1 = np.cos(rot1)
    sinrot1 = np.sin(rot1)
    cosrot2 = np.cos(rot2)
    sinrot2 = np.sin(rot2)
    cosrot3 = np.cos(rot3)
    sinrot3 = np.sin(rot3)

    rot3_mat = np.array([[cosrot3, -sinrot3, 0],
                     [sinrot3, cosrot3, 0],
                     [0, 0, 1]])

    rot2_mat = np.array([[cosrot2, 0, -sinrot2],
                     [0, 1, 0],
                     [sinrot2, 0, cosrot2]])

    rot1_mat = np.array([[1, 0, 0],
                     [0, cosrot1, sinrot1],
                     [0, -sinrot1, cosrot1]])

    rotation_matrix = np.dot(np.dot(rot3_mat, rot2_mat), rot1_mat)
    # order is 3, 2, 1 because matrices

    return rotation_matrix


def pyfai_to_hexrd_params(distance, poni1, poni2, theta1, theta2, theta3, detector_shape, pixel_size):
    # find unit vectors of rotated axis in the pyFAI lab frame (done)  ->  transform to heXRD coords (done)  ->  find detector rotation in heXRD coords (done)
    # find location of detector centre before rotation in pyFAI coords (done)  ->  rotate to real location in pyFAI coords (done)  ->  transform into heXRD coords (done)
    rotation_matrix_pyfai = pyfai_rotation_matrix(theta1, theta2, theta3)
    corner_to_centre = np.array([pixel_size[0] * detector_shape[0] / 2,
                                 pixel_size[1] * detector_shape[1] / 2,
                                 0])

    pyfai_to_hexrd_mat = np.array([[0, 1, 0],
                                   [1, 0, 0],
                                   [0, 0, -1]])
    detector_corner_un_rotated = np.array([-poni1, -poni2, distance])
    detector_centre_un_rotated = detector_corner_un_rotated + corner_to_centre
    location = np.around(np.dot(pyfai_to_hexrd_mat, np.dot(rotation_matrix_pyfai, detector_centre_un_rotated)) * 1000,
                         5)

    rotation_matrix_hexrd = np.dot(np.dot(pyfai_to_hexrd_mat, rotation_matrix_pyfai), pyfai_to_hexrd_mat)
    theta_y = np.arcsin(-rotation_matrix_hexrd[2, 0])
    cos_theta_y = np.cos(theta_y)
    theta_x = np.arccos(np.around(rotation_matrix_hexrd[2, 2] / cos_theta_y, 8))
    theta_z = np.arcsin(np.around(rotation_matrix_hexrd[1, 0] / cos_theta_y, 8))
    rotation = np.array([theta_x, theta_y, theta_z])
    return location, rotation


def get_coords(pixel_size, side_length, poni1, poni2, distance, rotation_matrix):
    id1, id2 = np.arange(0, side_length, 1) * pixel_size, np.arange(0, side_length, 1) * pixel_size
    pixel1, pixel2 = np.meshgrid(id1, id2)

    size = pixel1.size

    p1, p2, p3 = (pixel1 - poni1).ravel(), (pixel2 - poni2).ravel(), np.zeros(size) + distance

    coords_un_rotated = np.vstack((p1, p2, p3))

    coords = np.dot(rotation_matrix, coords_un_rotated)
    return coords

