# This file is meant to hold functions used in the other sections of code within this directory
#
# creator: thomas stevens
# created: 2024/08/13

import time


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
