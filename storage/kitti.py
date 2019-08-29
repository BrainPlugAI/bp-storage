'''Loads Kitti data into the same beard format.

author: Felix Geilert
'''

import os
from . import utils
from . import beard


#--------------------------------------------------------------------------------------------------

def load(folder, classes=None, only=None, size=None, show_btype=True, resize=utils.ResizeMode.FIT, pad_color=(0,0,0), pad_mode=utils.PadMode.EDGE, beard_style=False, debug=False):
    '''Loads the kitti data and returns generator.

    Args:
        folder (str): the path to load the kitti data
        classes (list): List of classes expected in the dataset (if None, assume default kitti classes)
        only (list): List of `BeardDataType` to limit the loading of the dataset
        size (int): Either single int of tuple of ints to indiciate the size of the image. If None image will not be resized. Data is provided as `[Height, Width]`.
        pad (bool): If image is resized, use pad to change data
        beard_style (bool): Converts the config to beard style for easier compatibility
        debug (bool): Gives debug output

    Returns:
        config (dict): Configuration loaded for the generator/dataset
        gen (Generator): Generator that returns tuples of data: `(img, global, metadata, DataType)`.
            Whereby `global` and `metadata` are dicts (metadata is an array of dicts) that contain the names of the
            elements in the config json.
    '''
    # safty: check if the folder exists
    if not os.path.exists(folder):
        raise IOError("Specified folder ({}) does not exist!".format(folder))

    # load the config file
    if classes is None:
        classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
    config = create_config(classes, beard_style)

    # create the generator and return data
    return config, beard._gen_beard(folder, config, only, size, show_btype, resize, pad_color, pad_mode, debug=debug)

def store(folder, debug=False):
    '''Stores data in the kitti format.'''
    raise NotImplementedError

def create_config(classes, beard_style=False):
    '''Generates a beard-like config for the Kitti dataset (to work with various input functions).
    Args:
        classes (list): List of class names that are used for the config

    Returns:
        config (dict): Dictionary that contains the config in the style of beard
    '''
    return {
        "global": [],
        "boxes": [
            {
                "type": "enum",
                "pos": 0,
                "name": "class" if beard_style else "type",
                "dtype": "str",
                "values": classes,
                "_comment": "Describes the type of object"
            },
            {
                "type": "value",
                "pos": 1,
                "name": "truncated",
                "dtype": "float",
                "_comment": "Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries"
            },
            {
                "type": "value",
                "pos": 2,
                "name": "occluded",
                "dtype": "int",
                "_comment": "Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown"
            },
            {
                "type": "value",
                "pos": 3,
                "name": "alpha",
                "dtype": "float",
                "_comment": "Observation angle of object, ranging [-pi..pi]"
            },
            {
                "type": "box-array",
                "length": 4,
                "name": "bbox",
                "bb_type": "absolute",
                "order": "x-y",
                "dtype": "int",
                "pos": 4,
                "_comment": "2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates"
            },
            {
                "type": "array",
                "length": 3,
                "name": "dimensions",
                "dtype": "float",
                "pos": 8,
                "_comment": "3D object dimensions: height, width, length (in meters)"
            },
            {
                "type": "array",
                "length": 3,
                "name": "location",
                "dtype": "float",
                "pos": 11,
                "_comment": "3D object location x,y,z in camera coordinates (in meters)"
            },
            {
                "type": "value",
                "pos": 14,
                "name": "rotation_y",
                "dtype": "float",
                "_comment": "Rotation ry around Y-axis in camera coordinates [-pi..pi]"
            },
            {
                "type": "value",
                "pos": 15,
                "name": "score",
                "dtype": "float",
                "optional": True,
                "_comment": "Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better."
            }
        ]
    }
