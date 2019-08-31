'''Various image handling functions.'''


from .common import *
import math
import numpy as np


# HANDLE IMAGE LOADING
try:
    import lycon

    # define functions
    def imwrite(img_path, img):
        '''Stores image to disk.'''
        img = np.ascontiguousarray(img, dtype=np.uint8)
        lycon.save(img_path, img)

    def imread(img_path, channels=3):
        '''Loads an image from the given path.'''
        img = lycon.load(img_path)
        if channels == 3:
            img = img[...,[2,1,0]]
        elif channels == 1:
            img = np.mean(img, axis=-1, keepdims=True)
        return img

    def imresize(img, width, height):
        return lycon.resize(img, width=width, height=height, interpolation=lycon.Interpolation.LINEAR)

except ImportError:
    print("WARNING: Could not find lycon, using cv2 instead!")
    try:
        import cv2
    except ImportError:
        raise RuntimeError("storage library requires either cv2 or lycon to be installed!")

    # define functions
    def imwrite(img_path, img):
        '''Stores image to disk.'''
        cv2.imwrite(img_path, img)

    def imread(img_path, channels=3):
        '''Loads an image from the given path.'''
        img = cv2.imread(img_path, 1)
        if channels == 3:
            img = img[...,[2,1,0]]
        elif channels == 1:
            img = np.mean(img, axis=-1, keepdims=True)
        return img

    def imresize(img, width, height):
        return cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)

# ----

def get_padding(params):
    if "padding" not in params.training:
        raise KeyError("Could not find value 'padding' in 'training'!")
    pad = params.training.padding
    mode = PadMode.EDGE
    resize = ResizeMode.FIT
    color = (0,0,0)
    if pad[0] == "center":
        mode = PadMode.CENTER
    if pad[1] == "stretch":
        resize = ResizeMode.STRETCH
    elif pad[1] == "black":
        resize = ResizeMode.PAD_COLOR
        color = (0,0,0)
    elif pad[1] == "blue":
        resize = ResizeMode.PAD_COLOR
        color = (0,0,255)
    elif pad[1] == "red":
        resize = ResizeMode.PAD_COLOR
        color = (255,0,0)
    elif pad[1] == "green":
        resize = ResizeMode.PAD_COLOR
        color = (0,255,0)
    elif pad[1] == "color":
        resize = ResizeMode.PAD_COLOR
        color = params.training.pad_color
    elif pad[1] == "random":
        resize = ResizeMode.PAD_RANDOM
    elif pad[1] == "mean":
        resize = ResizeMode.PAD_MEAN
    elif pad[1] == "edge":
        resize = ResizeMode.PAD_EDGE

    return mode, resize, color

def imread_resize(img_path, params):
    '''Loads an image from the given path and resizes it according to configuration.'''
    img = imread(img_path, params.network.color_channels)
    mode, res_mode, pad_color = get_padding(params)
    img, _, _ = resize(img, params.network.input_size, res_mode, pad_color, mode)
    return img

def pad(img, size, resize=ResizeMode.FIT, pad_color=(0,0,0), pad_mode=PadMode.EDGE):
    '''Pads an image to a new size.

    Returns:
        img (np.array): padded image
        offset (tuple): integer tuple that stores the offset from the upper left corner in format `[TOP, LEFT]`
    '''
    # retrieve general parameter
    pad_size = [(size[0] - img.shape[0]), (size[1] - img.shape[1])]
    padding = [(0, 0), (0, 0), (0, 0)]

    # add padding to the image
    if pad_mode == PadMode.EDGE:
        padding = [(0, int(pad_size[0])), (0, int(pad_size[1])), (0, 0)]
        pad_size = [0, 0]
    elif pad_mode == PadMode.CENTER:
        pad_size = [pad_size[0] / 2, pad_size[1] / 2]
        padding = [(math.floor(pad_size[0]), math.ceil(pad_size[0])), (math.floor(pad_size[1]), math.ceil(pad_size[1])), (0, 0)]

    # check additional padding modes
    if resize == ResizeMode.PAD_COLOR:
        img_new = np.stack([np.full(size, col) for col in pad_color], axis=-1)
        img_new[padding[0][0]:padding[0][0]+img.shape[0], padding[1][0]:padding[1][0]+img.shape[1], :] = img
        img = img_new
    elif resize == ResizeMode.PAD_MEAN:
        mode = "mean"
    elif resize == ResizeMode.PAD_EDGE:
        mode = "edge"
    elif resize == ResizeMode.PAD_RANDOM:
        img_new = np.random.randint(low=0, high=255, size=[size[0], size[1], 3])
        img_new[padding[0][0]:padding[0][0]+img.shape[0], padding[1][0]:padding[1][0]+img.shape[1], :] = img
        img = img_new
    else:
        return img, (0, 0)

    # update the image
    if resize not in (ResizeMode.PAD_COLOR, ResizeMode.PAD_RANDOM):
        padding = padding if len(img.shape)>=3 and img.shape[2]>1 else padding[:2]
        img = np.pad(img, padding, mode=mode)

    return img, (padding[0][0], padding[1][0])

def resize(img, size=None, resize=ResizeMode.FIT, pad_color=(0,0,0), pad_mode=PadMode.EDGE):
    '''Resizes the image and provides the scale.

    Returns:
        img (np.array): Array of the image
        scale (tuple): Tuple of float values containing the scale of the image in both dimensions
        offset (tuple): Tuple of int values containing the offset of the image from top left corner (through padding)
    '''
    # check if valid
    if size is None:
        return img, (1.0, 1.0), (0, 0)

    # retrieve some params
    img_size = img.shape[:2]
    offset = (0, 0)
    scale = (1.0, 1.0)

    # check the type of data
    if type(size) == tuple or type(size) == list or type(size) == np.ndarray:
        if resize == ResizeMode.FIT:
            frac = min((size[0] / img_size[0], size[1] / img_size[1]))
            scale = (frac, frac)
        elif resize == ResizeMode.STRETCH:
            scale = (size[0] / img_size[0], size[1] / img_size[1])
            frac = size
        else:
            frac = min((size[0] / img_size[0], size[1] / img_size[1]))
            scale = (frac, frac)
    elif type(size) == int:
        if resize == ResizeMode.FIT:
            frac = float(size) / max(img_size)
            scale = (frac, frac)
        elif resize == ResizeMode.STRETCH:
            frac = (size, size)
            scale = (frac[0] / img_size[0], frac[1] / img_size[1])
        else:
            frac = float(size) / max(img_size)
            scale = (frac, frac)
        size = (size, size)
    else:
        raise ValueError("Size has unkown type ({}: {})".format(type(size), size))

    # scale image and set padding
    #img = scipy.misc.imresize(img, frac)
    nsize = img.shape
    if isinstance(frac, float):
        nsize = [min(np.ceil(nsize[0] * frac), size[0]), min(np.ceil(nsize[1] * frac), size[1])]
    else:
        nsize = frac
    img = imresize(img, width=nsize[1], height=nsize[0])
    img, offset = pad(img, size, resize, pad_color, pad_mode)

    return img, scale, offset

def get_spaced_colors(n):
    '''Retrieves n colors distributed over the color space.'''
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors][:n]

def fill_patch(img, bbox, mode, color):
    '''Fills the given image patch in the given mode.

    Args:
        img (np.ndarray): Image array
        bbox (list): Bounding box for the patch in absolute coordinates and yx format
        mode (FillMode): FillMode that is used to fill the item
    '''
    # safty: check size of the box against image size
    bbox = [max(0, bbox[0]), max(0, bbox[1]), min(img.shape[0], bbox[2]), min(img.shape[1], bbox[3])]

    def _gen_patch(color):
        arr = []
        for i in range(len(color)):
            el = np.full([bbox[2] - bbox[0], bbox[3] - bbox[1]], color[i])
            arr.append(el)
        return np.stack(arr, axis=-1)

    # generate the element
    if mode == FillMode.MEAN:
        color = np.mean(img, (0, 1), dtype=np.float)
        patch = _gen_patch(color)
    elif mode == FillMode.COLOR:
        patch = _gen_patch(color)
    elif mode == FillMode.RANDOM:
        patch = np.random.randint(0, 255, [bbox[2] - bbox[0], bbox[3] - bbox[1], img.shape[2]])
    else:
        raise ValueError("Unkown value for fillmode ({})".format(mode))

    img[bbox[0]:bbox[2], bbox[1]:bbox[3], :] = patch

    return img
