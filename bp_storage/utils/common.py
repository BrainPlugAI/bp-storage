'''Holds some common helper functions and structs that are used in storage.'''


from enum import Enum
import os, glob


class DataType(Enum):
    '''Types of possible input data.'''
    TRAINING    = 0
    DEVELOPMENT = 1
    TESTING     = 2

class ResizeMode(Enum):
    '''Types of resize functions.'''
    FIT         = 0
    STRETCH     = 1
    PAD_COLOR   = 2
    PAD_MEAN    = 3
    PAD_EDGE    = 4
    PAD_RANDOM  = 5

class FillMode(Enum):
    '''Fill Mode for image patches.'''
    COLOR  = 0
    MEAN   = 1
    RANDOM = 2

class PadMode(Enum):
    '''Defines the type of padding mode.'''
    EDGE    = 0
    CENTER  = 1

def dict_folders():
    '''Returns dict with all folder combinations.'''
    return {
        DataType.TRAINING: ["train", "training"],
        DataType.DEVELOPMENT: ["val", "validation", "valid", "dev", "develop", "development"],
        DataType.TESTING: ["test", "testing"]
    }

def only_folders(only):
    # generate data
    folders = dict_folders()

    # filter the data if relevant
    if only is not None:
        if isinstance(only, DataType): only = [only]
        folders = {k: v for k, v in folders.items() if k in only}
    return folders

def detect_folders(path):
    '''Retrieves the path for the images and labels folders.'''
    img = None
    for folder in ['image', 'images', 'img', 'imgs']:
        img = os.path.join(path, folder)
        if os.path.exists(img):
            break

    lbl = None
    for folder in ['label', 'labels', 'lbl', 'lbls']:
        lbl = os.path.join(path, folder)
        if os.path.exists(lbl):
            break

    return img, lbl

def search_imgs(img_dir):
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs += glob.glob(os.path.join(img_dir, ext), recursive=True)
    return imgs

def num_imgs(img_dir):
    return len(_search_imgs(img_dir))
