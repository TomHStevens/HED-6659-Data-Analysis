# This file is meant to hold functions used in the other sections of code within this directory
#
# creator: thomas stevens
# created: 2024/08/13

import time
import numpy as np
import yaml


def get_coords(pixel_size, side_length, poni1, poni2, distance, rotation_matrix):
    id1, id2 = np.arange(0, side_length, 1) * pixel_size, np.arange(0, side_length, 1) * pixel_size
    pixel1, pixel2 = np.meshgrid(id1, id2)

    size = pixel1.size

    p1, p2, p3 = (pixel1 - poni1).ravel(), (pixel2 - poni2).ravel(), np.zeros(size) + distance

    coords_un_rotated = np.vstack((p1, p2, p3))

    coords = np.dot(rotation_matrix, coords_un_rotated)
    return coords

