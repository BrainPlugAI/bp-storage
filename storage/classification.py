'''Data Loader for classification dataset.

author: Felix Geilert
'''

from random import randint
import numpy as np
import os, glob, math
import shutil
from . import utils

def _img_gen(folder):
    imgs = utils.search_imgs(folder)
    for img in imgs:
        yield utils.imread(img)
        
def _find_classes(folder, only):
    '''Retrieves the relevant classes from the '''
    # generate data
    folders = utils.only_folders(only)
    
    # find all classes (iterate through all relevant folders)
    rel_classes = []
    for btype in folders:
        # iterate through all possible dirs
        for dir in folders[btype]:
            # generate the folder name and check if exists
            dir = os.path.join(folder, dir)
            if not os.path.exists(dir):
                continue
            # iterate through all subdirs
            _, dirs, _ = next(os.walk(dir))
            for cls_dir in dirs:
                cls_name = cls_dir.upper()
                cls_dir = os.path.join(dir, cls_dir)
                if not os.path.isdir(cls_dir):
                    continue
                if cls_name not in rel_classes:
                    rel_classes.append(cls_name)
    # return the generated classess
    return rel_classes

def _gen_single(img, cls_name, classes, btype, one_hot=True, beard_format=False, show_btype=False):
    '''Generate tuple for a single output.'''
    if one_hot:
        cls_name = np.eye(len(classes))[classes.index(cls_name)]
    if beard_format:
        if show_btype: return img, {}, [{utils.const.ITEM_CLASS: cls_name, utils.const.ITEM_BBOX: [0,0,0,0]}], btype
        else: return img, {}, [{utils.const.ITEM_CLASS: cls_name, utils.const.ITEM_BBOX: [0,0,0,0]}]
    else:
        if show_btype: return img, cls_name, btype
        else: return img, cls_name

def _gen_cls(folder, classes, shuffle=True, only=None, size=None, one_hot=True, beard_format=False, show_btype=False, resize=utils.ResizeMode.FIT, pad_color=(0,0,0), pad_mode=utils.PadMode.EDGE, debug=False):
    '''Loads the images from the given folder.
    
    Default output format is img, label, btype
    
    Args:
        folder (str): folder to load the data from
        classes (list): list of classes (prefered upper case)
        shuffle (bool): defines if the data should be shuffled (default: True)
        only (DataType):
        size (int):
        one_hot (bool): defines if the classes should be given as one_hot vectors
        beard_format (bool): defines if the generator should output in the same format as the beard & kitti generators
    '''
    # generate data
    folders = utils.only_folders(only)
    if classes is None:
        raise ValueError("Expected list of classes, but got None!")
    classes = [x.upper() for x in classes]

    # iterate through folders
    for btype in folders:
        found = False
        for dir in folders[btype]:
            # check if folder exists
            dir = os.path.join(folder, dir)
            if not os.path.exists(dir):
                continue
            found = True

            # check class folders
            cls_gens = []
            _, dirs, _ = next(os.walk(dir))
            for cls_dir in dirs:
                cls_name = cls_dir.upper()
                cls_dir = os.path.join(dir, cls_dir)
                if not os.path.isdir(cls_dir):
                    continue
                if classes is None or cls_name in classes:
                    cls_gens.append((cls_name, _img_gen(cls_dir)))

            # shuffle the classes
            if not shuffle:
                for cls_name, gen in cls_gens:
                    for img in gen:
                        img, _, _ = lib.resize(img, size, resize, pad_color, pad_mode)
                        yield _gen_single(img, cls_name, classes, btype, one_hot, beard_format, show_btype)
            else:
                while len(cls_gens) > 0:
                    id = randint(0, len(cls_gens) - 1)
                    try:
                        cls_name, gen = cls_gens[id]
                        img = next(gen)
                        img, _, _ = utils.resize(img, size, resize, pad_color, pad_mode)
                        yield _gen_single(img, cls_name, classes, btype, one_hot, beard_format, show_btype)
                    except StopIteration:
                        del cls_gens[id]

        # debug output
        if not found:
            if debug: print("Could not find folder for type: {}".format(btype.name))

def load(folder, classes=None, only=None, size=None, one_hot=True, beard_format=False, show_btype=False, resize=utils.ResizeMode.FIT, pad_color=(0,0,0), pad_mode=utils.PadMode.EDGE, debug=False, shuffle=True):
    '''Loads the classification data from file.

    Returns:
        folder (str): Folder that contains the classification structure
        debug (bool): Defines if debugs messages should be shown
    '''
    # safty: check if the folder exists
    if not os.path.exists(folder):
        raise IOError("Specified folder ({}) does not exist!".format(folder))
        
    # load the relevant classes
    if classes is None:
        classes = _find_classes(folder, only)

    return classes, _gen_cls(folder, classes, shuffle, only, size, one_hot, beard_format, show_btype, resize, pad_color, pad_mode, debug)

def load_sample_imgs(folder, only, size=None, count=10, classes=None, resize=utils.ResizeMode.FIT, pad_color=(0,0,0), pad_mode=utils.PadMode.EDGE):
    '''Loads relevant count of sample images'''
    # safty: check if the folder exists
    if not os.path.exists(folder):
        raise IOError("Specified folder ({}) does not exist!".format(folder))
        
    # load the relevant classes
    if classes is None:
        classes = _find_classes(folder, only)
        
    gen = _gen_cls(folder, classes, True, only, size, True, False, False, resize, pad_color, pad_mode, False)
    
    imgs = []
    labels = []
    for img, lbl in gen:
        # check for end
        if len(imgs) >= count: break
        if np.random.randint(0, 10) > 5: continue
        # add data
        imgs.append(img)
        labels.append(lbl)
        
    # compress
    imgs = np.stack(imgs, axis=0)
    labels = np.stack(labels, axis=0)
    return imgs, labels

def write(folder, debug=False):
    raise NotImplementedError
