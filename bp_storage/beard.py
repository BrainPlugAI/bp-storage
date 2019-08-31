'''Generator for loading and saving beard data.

author: Felix Geilert
'''


import numpy as np
import os, glob, math
import shutil
import json, csv
from . import utils


#--------------------------------------------------------------------------------------------------
# PUBLIC HELPER FUNCTIONS

def select_config(name, config):
    '''Retrieves the element with the regarding name from the config.'''
    for item in config:
        if item["name"] == name:
            return item
    return None

#--------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS

# TODO: convert to class (to return meta-data struct?)
# TODO: provide static method to call directly to generator!

def _load_value(row, item, debug):
    '''Loads the item from the row and removes the regarding element from the row.'''
    value = None
    # safty: check if valid
    if len(row) == 0 and "optional" in item and item["optional"]:
        return None, row
    elif len(row) == 0:
        raise ValueError("Could not extract element ({}) from ({}) as it is empty".format(item["name"], row))
    # load the value
    if item["type"] == "enum":
        # get the values as upper case for checking
        value = row[0]
        value_list = [it.upper() for it in item["values"]]
        oval = value
        row = row[1:]

        # check dtype
        if item["dtype"] == "str":
            value = value.upper()
            value = value_list.index(value) if value in value_list else -1
        else:
            value = int(value)

        # generate output
        if value == -1 or (type(value) == int and value > len(value_list)):
            if debug: print("WARNING: the loaded class value ({}) is out of range ({}) or not in class list ({})".format(oval, len(item["values"]), item["values"]))
            value = -1 if item["dtype"] == "int" else "UNKOWN"
        else:
            # note: use default item here to restore cases
            value = value if item["dtype"] == "int" else item["values"][value]
    elif item["type"] == "array":
        value = []
        for i in range(item["length"]):
            value.append(utils.set_dtype(row[0], item["dtype"]))
            row = row[1:]
    elif item["type"] == "box-array":
        value = []
        for i in range(item["length"]):
            value.append(utils.set_dtype(row[0], item["dtype"]))
            row = row[1:]
        # TODO: convert to BBOX
    elif item["type"] == "value":
        value = utils.set_dtype(row[0], item["dtype"])
        row = row[1:]
    else:
        print("ERROR: data type ({}) is unkown!".format(item["type"]))
    return value, row

def _write_value(item, out, item_config, pos, debug):
    '''Creates the string output for a single item that should be written to the labels file.'''
    # safty: check if config exists
    if item_config is None:
        print("ERROR: The current element has no configuration!")
        return out

    # load the data
    value = None
    if item_config["type"] == "enum":
        # retrieve values
        value = item
        value_list = [it.upper() for it in item_config["values"]]
        oval = value

        # convert to int as base type
        if item_config["dtype"] == "str":
            value = value.upper()
            if value in value_list: value = value_list.index(value)
            else:
                if debug: print("WARNING: the specificed class value ({}) is out of range ({}) or not in class list ({})".format(oval, len(item_config["values"]), item_config["values"]))
                value = -1
        else:
            value = int(value)

        # convert back if required
        if item_config["dtype"] == "str":
            value = "UNKOWN" if value == -1 else item_config["values"][value]
    elif item_config["type"] == "array":
        value = []
        for i in range(item_config["length"]):
            value.append(utils.set_dtype(item[i], item_config["dtype"]))
        value = ' '.join(value)
    elif item_config["type"] == "box-array":
        value = []
        for i in range(item_config["length"]):
            value.append(str(utils.set_dtype(item[i], item_config["dtype"])))
        value = ' '.join(value)
    elif item_config["type"] == "value":
        value = str(utils.set_dtype(item, item_config["dtype"]))
    else:
        print("ERROR: data type ({}) is unkown!".format(item["type"]))

    # add the element with position information
    pos = item_config["pos"] if "pos" in item_config else len(out)
    out.append((pos, value))
    return out

def _gen_single(img, gdata, mdata, btype, show_btype=False):
    '''Generate tuple for a single output to adjust it to provided style.'''
    if show_btype: return img, gdata, mdata, btype
    else: return img, gdata, mdata

#--------------------------------------------------------------------------------------------------
# BEARD LOADING

def _gen_beard(folder, config, only=None, size=None, show_btype=True, resize=utils.ResizeMode.FIT, pad_color=(0,0,0), pad_mode=utils.PadMode.EDGE, classes=None, debug=False):
    # convert the global and boxes config to the right order
    global_config = config["global"]
    global_config.sort(key=lambda x: x["pos"] if "pos" in x else 0)
    boxes_config = config["boxes"]
    boxes_config.sort(key=lambda x: x["pos"] if "pos" in x else 0)

    # generate data
    folders = utils.only_folders(only)

    # iterate through folders
    for btype in folders:
        found = False
        for dir in folders[btype]:
            # check if folder exists
            dir = os.path.join(folder, dir)
            if not os.path.exists(dir):
                continue
            found = True

            # load the folder
            img_dir, lbl_dir = utils.detect_folders(dir)
            # search all relevant data
            imgs = utils.search_imgs(img_dir)

            # iterate through all data
            for img_path in imgs:
                # get the basename of the image
                lbl_path = os.path.splitext(os.path.basename(img_path))[0]
                lbl_path = os.path.join(lbl_dir, lbl_path + '.txt')

                # load the image (and convert it to RGB)
                img = utils.imread(img_path)
                gdata = {}
                mdata = []

                # resize the image
                img, scale, offset = utils.resize(img, size, resize, pad_color, pad_mode)

                # load the regarding labels
                with open(lbl_path, 'r') as csvfile:
                    lbl_reader = csv.reader(csvfile, delimiter=' ')

                    # load global data (if there is any)
                    if len(global_config) > 0:
                        row = next(lbl_reader)
                        for item in global_config:
                            # load the value
                            value, row = _load_value(row, item, debug)
                            # store the element
                            gdata[item["name"]] = value

                    # load metadata
                    for row in lbl_reader:
                        meta = {}
                        for item in boxes_config:
                            # load the value
                            value, row = _load_value(row, item, debug)

                            # check for transformation
                            if item["type"] == "box-array":
                                is_rel = (item["bb_type"].lower() == "relative")
                                # switch, as they are always stored in y-x format
                                bb_scale = (scale[1], scale[0])    if item["order"] == "x-y" else scale
                                bb_offset = (offset[1], offset[0]) if item["order"] == "x-y" else offset
                                if is_rel:
                                    value = [(value[0] * bb_scale[0]) + bb_offset[0],
                                             (value[1] * bb_scale[1]) + bb_offset[1],
                                             value[2] * bb_scale[0],
                                             value[3] * bb_scale[1]]
                                else:
                                    value = [(value[0] * bb_scale[0]) + bb_offset[0],
                                             (value[1] * bb_scale[1]) + bb_offset[1],
                                             (value[2] * bb_scale[0]) + bb_offset[0],
                                             (value[3] * bb_scale[1]) + bb_offset[1]]
                                value = np.array(value).astype(int)

                            # update classes according to limitations (IF: classes and config do not match, i.e. separate classes arg provided)
                            if item["name"] == "class" and classes is not None:
                                if value not in classes:
                                    value = classes[0]

                            # store the element
                            meta[item["name"]] = value
                        mdata.append(meta)

                # return the loaded elements
                yield _gen_single(img, gdata, mdata, btype, show_btype)

        # debug output
        if not found:
            if debug: print("Could not find folder for type: {}".format(btype.name))

    # debug output
    if debug: print("Loaded entire dataset")

# data loading
def load(folder, json_name="*.json", only=None, size=None, show_btype=True, resize=utils.ResizeMode.FIT, pad_color=(0,0,0), pad_mode=utils.PadMode.EDGE, classes=None, debug=False):
    '''Creates a generator for the beard dataset.

    Args:
        folder (str): the folder to load beard from (root folder containing the json file)
        json_name (str): Name of the json file. Uses first wildcard element by default.
        only (list): List of `BeardDataType` to limit the loading of the dataset
        size (int): Either single int of tuple of ints to indiciate the size of the image. If None image will not be resized. Data is provided as `[Height, Width]`.
        pad (bool): If image is resized, use pad to change data
        classes (list): List of classes to use (if specificed in the model - other elements will be moved to dontcare) [if none use all classes]
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

    # find the config file
    config_file = glob.glob(os.path.join(folder, json_name), recursive=False)
    if len(config_file) == 0:
        raise IOError("Cannot find the config file ({})!".format(json_name))

    # load the config file
    config = {}
    with open(config_file[0]) as f:
        config = json.load(f)

    # check the classes in the config file
    out_config = config
    if classes is not None:
        for item in out_config["boxes"]:
            if item["name"] == "class":
                item["values"] = classes
                break

    # create the generator and return data
    return out_config, _gen_beard(folder, config, only, size, show_btype, resize, pad_color, pad_mode, classes, debug)

# data storing
def store(gen, config, folder, clean=False, debug=False, start_id=0):
    '''Stores the data from the provided generator to folder.

    Args:
        gen (Generator): Beard generator that provides the relevant data.
        config (dict): Should contain both `global` and `boxes` data to be stored as config file.
        folder (str): folder to store the dataset into
        clean (bool): Defines clean storage (if true deletes any existing data in `folder`)
        debug (bool): If debug output should be shown
    '''
    # check to clean the folder
    if clean and os.path.exists(folder):
        shutil.rmtree(folder)

    # generate folder structure
    if not os.path.exists(folder):
        os.mkdir(folder)

    train_dir = os.path.join(folder, 'train')
    val_dir = os.path.join(folder, 'dev')
    test_dir = None

    # write the configuration
    with open(os.path.join(folder, 'config.json'.format(id)), 'w') as f:
        # store the file
        json.dump(config, f)

    # generate the folder for the training data
    for fldr_dir in [val_dir, train_dir]:
        if os.path.exists(fldr_dir) and clean:
            shutil.rmtree(fldr_dir)
        if not os.path.exists(fldr_dir):
            os.mkdir(fldr_dir)
        if not os.path.exists(os.path.join(fldr_dir, 'images')):
            os.mkdir(os.path.join(fldr_dir, 'images'))
        if not os.path.exists(os.path.join(fldr_dir, 'labels')):
            os.mkdir(os.path.join(fldr_dir, 'labels'))

    # iterate the counter
    counter = start_id
    for img, gdata, mdata, btype in gen:
        counter += 1
        if btype == utils.DataType.TRAINING: fldr = train_dir
        elif btype == utils.DataType.DEVELOPMENT: fldr = val_dir
        elif btype == utils.DataType.TESTING:
            # check folder and create new
            if test_dir is None:
                test_dir = os.path.join(folder, 'test')
                if os.path.exists(test_dir) and clean:
                    shutil.rmtree(test_dir)
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)
                if not os.path.exists(os.path.join(test_dir, 'images')):
                    os.mkdir(os.path.join(test_dir, 'images'))
                if not os.path.exists(os.path.join(test_dir, 'labels')):
                    os.mkdir(os.path.join(test_dir, 'labels'))
            fldr = test_dir

        #cv2.imwrite(os.path.join(fldr, 'images', '{:06d}.jpg'.format(counter)), img)
        img = img[...,[2,1,0]]
        utils.imwrite(os.path.join(fldr, 'images', '{:06d}.jpg'.format(counter)), img)
        #scipy.misc.imsave(os.path.join(fldr, 'images', '{:06d}.jpg'.format(counter)), img)
        with open(os.path.join(fldr, 'labels', '{:06d}.txt'.format(counter)), 'w+') as f:
            # write the global data
            gstr = []
            for i, gd in enumerate(gdata):
                conf = select_config(gd, config["global"])
                gstr = _write_value(gdata[gd], gstr, conf, i, debug)
            # convert data
            f.write( " ".join( [x[1] for x in sorted(gstr, key=lambda x: x[0])] ) )

            # write the metadata
            for items in mdata:
                f.write('\n')
                mstr = []
                # generate the data
                for i, item in enumerate(items):
                    conf = select_config(item, config["boxes"])
                    mstr = _write_value(items[item], mstr, conf, i, debug)
                f.write( " ".join( [x[1] for x in sorted(mstr, key=lambda x: x[0])] ) )

        # output current data as generator
        yield counter, btype
