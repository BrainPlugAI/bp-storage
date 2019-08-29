# BrainPlug Storage Library

Library to load, augment and different dataset formats for use in machine learning models.

> Note: The repo is still work in progress, so documentation might not be up to date

## Getting Started

Storage Library allows loading and augmentation in various data formats. It even allows to generate `Tensorflow` Tf-Records and load them (coming soon (TM)).

Simply install using (distribution via PyPi is planned):

```bash
pip3 install .
```

Currently the library has 4 parts:

* `storage.classification` - Allows to load simple classification datasets
* `storage.kitti` - Allows to load the kitti format for usage in detectors (3D Data not supported currently)
* `storage.beard` - Allows to load beard format (format optimized for localization tasks)
* `storage.utils` - Various helper functions

In general each data loader will create a python generator that can be used to loop over the data. Datasets in general are split into different types (defined in `storage.utils.DataType`):

* `TRAINING` - Used for general training purposes
* `DEVELOPMENT` - Used for validation and exploration during the development process (you will probably make certain assumptions about the structure of the dataset through this data)
* `VALIDATION` - Final validation data used to measure the performance of the trained model and to validate your assumptions

These datatypes are added to the end of the enum and the creation functions also allow you to filter the dataset for certain types.

## Examples

### General Concepts

Each load function allows to resize the images through 3 parameters, which contain enums:

* `resize` [`storage.utils.ResizeMode`] - Defines how the images are resized and if it should be padded
* `pad_mode` [`storage.utils.PadMode`] - In case of padding defines if the image should be pinned to top left corner or centered
* `pad_color` [Color Array] - Defines the color of the padding, if `ResizeMode.PAD_COLOR` is selected

Each load function also allows to specify the maximum size of the output image through `size` and if the dataset type (i.e. `storage.utils.DataType`) is provided for each element in the generator through `show_btype`. It also allows to filter only for a specific btype through the `only` argument, which expects a single or a list of multiple `DataType`.

### Classification

This is the simples type of dataset:

```python
import storage
from storage.utils import ResizeMode, DataType
# DEBUG: using cv2 for debug output
import cv2

# load the generator
classes, gen = storage.classification.load(folder, size=(512, 512), resize=ResizeMode.FIT, only=DataType.TRAINING, show_btype=True)

print("Found classes: {}".format(classes))

# DEBUG: show the output of the generator
for img, label, ds in gen:
  cv2.putText(img, label, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
  # show the result
  cv2.imshow("Storage Output", img)
  cv2.waitKey(0)
```

### Kitti & Beard

The loading of kitti and beard data is quite similar (i.e. kitti uses the beard loader internally). Both function should have similar signatures. The only difference are:

* `beard_style` parameter for `kitti.load()`, which switches between classic kitti format and beard generator style output.
* `classes` parameter for `kitti.load()`, which allows to provide classes that deviate from default kitti classes (for beard, these are stored in the config file)

Therefore we will only look at beard loading here:

```python
import storage
from storage.utils import ResizeMode, PadMode, DataType
# DEBUG: using cv2 for debug output
import cv2

# load the generator
config, gen = storage.beard.load(folder, only=DataType.DEVELOPMENT, size=(512, 512), resize=ResizeMode.PAD_COLOR, pad_color=(255, 255, 255), pad_mode=PadMode, show_btype=False)

# DEBUG: show the output of the generator
colors = storage.utils.get_spaced_colors(len(classes) + 1)[1:]
for img, gdata, mdata in gen:
  # note: gdata contains global image information (empty in kitti) and mdata hold classes and locations of objects
  # go through all elements
  for item in mdata:
    # highlight bbs
    if storage.utils.const.ITEM_BBOX in item:
      # retrieve the boxes
      coords = item[storage.utils.const.ITEM_BBOX]
      cls = item[storage.utils.const.ITEM_CLASS]

      # draw the bounding box
      cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), colors[cls], 2)
      cv2.putText(img, cls,(coords[0], coords[3] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[cls], 1, cv2.LINE_AA)

  # show the result
  cv2.imshow("Storage Output", img)
  cv2.waitKey(0)
```

For additional insights take a look at the `scripts` folder.

**NOTE:** In the default case the class attribute stored in `item` for kitti data is named `type` and not `class` (as stored in `storage.utils.const.ITEM_CLASS`)

## Dataset structures

[Beard](beard-definition) and [Kitti](kitti-definition) structures are described in separate documents. Classification expects a simple structure. Like in beard data is split into multiple folders for the datatype (`train`, `val`, `dev`). Each folder contains a subfolder for each class that should be classified (e.g. `cat` and `dog`). These subfolders then contain the actual images.

## Dependencies

* `lycon` or `cv2` - for fast loading of images and resizing (`pip install lycon`, however there seems not to be real windows support at the moment) [NOTE: you can also use cv2 instead, the library will adapt automatically]
* default python stack (`numpy`, `pandas`, etc.)

## Performance

One of the performance bottlenecks appears to be the numpy `pad` functions. However they are currently rewritten (see [here](https://github.com/numpy/numpy/pull/11358)) and might improve performance in future versions of numpy.

## Known Issues

* Augmentation only works with absolute coordinates on x-y ordering! (otherwise might produce wrong results, use `test_input` to verify!)

## License

Published under MIT License.
