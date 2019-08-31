'''Functions to handle dataset code.'''

from .common import *
from .images import *
from . import const
import numpy as np
import random

def set_dtype(value, dtype):
    '''Converts the value to the given dtype.'''
    if dtype == "float":
        value = float(value)
    elif dtype == "int":
        value = int(value)
    elif dtype == "str":
        value = value
    return value

def augment(gen, config, keep=False, stages=None, params=None, meta_udf=None):
    '''Augments the dataset if required.

    This function pays special respect to objects of type `box-array` to update them according to the transformations.
    It will also update the `complexity` (global value) accordingly.

    The `meta_udf` has a signature of `(mdata, gdata, config, params) -> (mdata, gdata)` in order to allow the user to define own transformations
    on the metadata.

    Args:
        gen (Generator): Beard-Style generator that should be augmented
        config (dict): Configuration of the dataset
        stages (list): List of dataset-stages (dev, train, etc.) that should be augmented (None=Augment all)
        keep (bool): Defines if the original image should be preserved
        params (dict): Dict of all relevant elements
        meta_udf (fct): user defined function that allows to update use-case specific metadata. Signature: (mdata, gdata, transform) => (mdata)
    '''
    # import relevant libs (do here, to avoid global crash if not installed!)
    import imgaug as ia
    from imgaug import augmenters as iaa

    # generate list of all augmentations
    augs = []
    if params is not None:
        if "flip" in params:
            augs.append(iaa.Fliplr(params["flip"]))
        if 'blur' in params:
            augs.append( iaa.GaussianBlur(sigma=(0, params["blur"])) )
        if 'crop' in params:
            augs.append( iaa.Crop(px=(0, params['crop'])) )
        if 'contrast' in params:
            augs.append( iaa.ContrastNormalization((params['contrast'][0], params['contrast'][1])) )
        if 'noise' in params:
            # strenght of the noise and amount added
            augs.append( iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, params['noise'][0]*255), per_channel=params['noise'][1]) )
        if 'transform' in params:
            transform_prob = lambda aug: iaa.Sometimes(params['transform'], aug)
            shear = params['shear'] if 'shear' in params else 0
            rot = params['rotate'] if 'rotate' in params else 0
            trans = params['translate'] if 'translate' in params else 0
            scale = params['scale'] if 'scale' in params else 0
            augs.append( transform_prob(iaa.Affine(
                scale={"x": (scale[0], scale[1]), "y": (scale[0], scale[1])},
                translate_percent={"x": (-trans, trans), "y": (-trans, trans)},
                rotate=(-rot, rot),
                shear=(-shear, shear),
                order=[0, 1],
                cval=(0, 255),
                mode=['constant', 'edge']
            )) )

    # create the augmentation model
    seq = iaa.Sequential(iaa.SomeOf((min(1, len(augs)), None), augs), random_order=True)

    # iterate through all data
    for img, gdata, mdata, btype in gen:
        # check if stage is augmented
        if stages is not None and btype not in stages:
            yield img, gdata, mdata, btype
            continue
        # output original if keep
        if keep:
            yield img, gdata, mdata, btype
        orig = img

        # iterate through all images that shall be generated
        for _ in range(params["per_img"]):
            # generate copy to not add multiple augmentations
            img = np.copy(orig)
            # retrieve augmentation standard
            seq_det = seq.to_deterministic()
            # augment the image
            aug_img = seq_det.augment_images([img])[0]

            # TODO: update metadata (todo: update for better search)
            aug_mdata = [x.copy() for x in mdata]
            bbs_dict = {const.ITEM_BBOX: [x[const.ITEM_BBOX] for x in aug_mdata]}

            # iterate through all elements
            for key in bbs_dict:
                bbs = np.copy(np.array(bbs_dict[key]))
                if len(bbs.shape) < 2 or bbs.shape[1] < 4:
                    print("\nERROR: WRONG BBS")
                    print(bbs)
                    continue
                # TODO: implement filter?
                #fltr = np.logical_and(bbs[:, 2] > bbs[:, 0], bbs[:, 3] > bbs[:, 1])
                #bbs = bbs[fltr]
                #cls = np.array(meta[0])[fltr]

                # TODO: implement order and format?
                # convert the bounding boxes to format
                bbs = [ia.BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]) for bb in bbs if bb[2] > bb[0] and bb[3] > bb[1]]
                bbs = ia.BoundingBoxesOnImage(bbs, shape=img.shape)

                # augment the bounding boxes
                aug_bbs = seq_det.augment_bounding_boxes([bbs])[0]
                aug_bbs = np.array([[bb.x1, bb.y1, bb.x2, bb.y2] for bb in aug_bbs.bounding_boxes])

                # update bounding boxes
                # load bounding boxes / check format
                for i in range(len(aug_bbs)):
                    aug_mdata[i][key] = aug_bbs[i]

            # TODO: update landmarks
            # TODO: update complexity based on transformations?

            # perform UDF transformations
            if meta_udf is not None:
                # TODO: update
                mdata = meta_udf(mdata, gdata, config, params)

            # send to gen
            yield aug_img, gdata, aug_mdata, btype

def merge(gens, shuffle=True, debug=True):
    '''Merges multiple given geneators (in beard format).

    Args:
        gens (List[Generator]): Generators in Beard format
        shuffle (bool): Defines if the two generators should be shuffled together

    Returns:
        gen (Generator): Generator in Beard format
    '''
    # merge the generators
    if not shuffle:
        for gen in gens:
            for img, gdata, mdata, btype in gen:
                yield img, gdata, mdata, btype
    else:
        while len(gens) > 0:
            id = random.randint(0, len(gens) - 1)
            try:
                img, gdata, mdata, btype = next(gens[id])
                yield img, gdata, mdata, btype
            except StopIteration as err:
                del gens[id]

def gen_negative(gen, mode=FillMode.COLOR, color=(0,0,0), is_rel=True, is_xy=False, scale=0.1, str_boxes=None, str_class=None):
    '''Converts a beard-style dataset into a negative dataset.

    Args:
        gen (Generator): Beard Style data generator
        mode (FillMode): The mode in which data should be filled
        color (tuple): Int tuple that defines a fill color (if mode is color)
        str_boxes (str): Name of the config element that contains the boxes

    Returns:
        gen (Generator): Beard-Style generator that contains only blacked images
    '''
    # update values
    str_boxes = const.ITEM_BBOX if str_boxes is None else str_boxes
    str_class = const.ITEM_CLASS if str_class is None else str_class
    # create new generator
    for img, gdata, mdata, btype in gen:
        for item in mdata:
            # retrieve boxes
            if str_boxes in item:
                bbox = item[str_boxes]
                if is_rel:
                    bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                if is_xy:
                    bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
                bbox = (np.array(bbox) * np.array([1-scale, 1-scale, 1+scale, 1+scale])).astype(np.int32)
                img = fill_patch(img, bbox, mode, color)
                # replace the element
                item[str_boxes] = [0,0,0,0]
                item[str_class] = "DONTCARE"

        # return data
        yield img, gdata, mdata, btype
