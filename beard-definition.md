# **B**rainPlug **E**xtended **AR**ea **D**atasets (BEARD)

This is the definition of the dataset format used by BrainPlug for object detections. It works as an extension to the kitti format.

#### Table of contents

* [Folder structure](#folder-structure)
* [Label format](#label-format)
* [Custom class mappings](#custom-class-mappings)

## Folder structure

You should have one folder containing images, and another folder containing labels.

* Image filenames are formatted like `IDENTIFIER.EXTENSION` (e.g. `000001.png` or `foo.jpg`).
* Label filenames are formatted like `IDENTIFIER.txt` (e.g. `000001.txt` or `foo.txt`).

These identifiers need to match.
So, if you have a `foo.png` in your image directory, there must to be a corresponding `foo.txt` in your labels directory.

If you want to include validation data, then you need separate folders for validation images and validation labels.
A typical folder layout would look something like this:
```
train/
├── images/
│   └── 000001.png
└── labels/
    └── 000001.txt
val/
├── images/
│   └── 000002.png
└── labels/
    └── 000002.txt
config.json
```

## Label format

The format of the labels if defined through the config file, which allows an easy extension of the format. In general there are two categories of elements: required and optional.

**Required:**

These elements should be in every definition.

* *Bounding Boxes* - Array that contains the coordinates of the bounding boxes
  * specified as an `array` of length 4
  * contains special attribute `bb_type`, which can be either `absolute` for `[TOP, LEFT, BOTTOM, RIGHT]` or `relative` for `[TOP, LEFT, HEIGHT, WIDTH]`
* *Class* - Class of the object as `enum`. Should always be at position `0`

**Optional**:

Those are elements that can be added to definitions. As the format is variable, there are unlimited more possibilities. However to ensure compatibility between different models and the labeling pipeline, the format for the most common additional elements are outlined here.

* Metaitem
* `complexity` - Defines the curriculum learning complexity of the element
* confidence

### JSON-Structure

The json file is divided into two sub-categories. `global` for all image wide elements are written in the first line of each txt file. `boxes` are the order of all following lines that contain the properties of each box.

#### Types of elements

There exists the following `types` of elements:

* `enum`
  * `values` - contains possible values of the enum in the correct order (for indexing)
  * `dtype` - specifies if the values are stored as Integers (`int`) or as strings (`str`)
* `value`
  * `dtype` - the data type of the element. If not given just assume `float`.
* `array`
  * `length` - defines how many elements this array has
  * `dtype` - the data type of the element. If not given just assume `float`.
* All Types
  * `optional` - boolean that defines if this element can be skipped

#### Attributes

Each element should contain the following attributes:

* `type` - one of the types defined above see above
* `name` - name of the elements (not relevant for training, but for labeling pipeline)
* `pos` - defines the position of the element in the txt
  * *NOTE*: if position is not given (or does not match - here parser warning!) the order in the outer json array is used

Objects might contain the following attributes:

* `dependency` - defines if this value is dependent on a certain class - should be ignored if class not given
* `_comment` - for explanation of the current elements (only used for the labeling pipeline and human understanding)

#### List of typical Global Elements

* `confidence` - Float that defines the confidence of the labels in the entire image
* `complexity` - Float that defines the complexity of the entire image (used for curriculum learning)

#### List of typical Boxes Elements

* `class` - string or int of the class of the single box
* `bbox` - array of the coordinates of the single box. Usually length 4.
  * `bb_type` - Additional attribute that can be either `relative` or `absolute` to define if the coordinates are `[top, left, height, width]` or `[top, left, bottom, right]`
  * `order` - Defines if data is stored as `'x-y'` (cv2 default) or `'y-x'` (tensorflow and numpy default). Recommended: `y-x`
  > NOTE: These values are currently not enforced and are purely informational for the loading side

* `blurred` - bool that defines if the box is blurred

### Examples

**Example of config.json**:
```json
{
  "global": [
    {
      "type": "value",
      "name": "complexity",
      "dtype": "float",
      "pos": 0,
      "_comment": "defines the curriculum learning complexity of the element"
    }
  ],
  "boxes": [
    {
      "type": "enum",
      "pos": 0,
      "name": "type",
      "dtype": "int",
      "values": ["DONTCARE", "BAG", "FACE"]
    },
    {
      "type": "box-array",
      "length": 4,
      "name": "bbox",
      "bb_type": "relative",
      "order": "x-y",
      "dtype": "int",
      "pos": 1
    },
    {
      "type": "value",
      "name": "side-face",
      "pos": 5,
      "dtype": "float",
      "dependency": ["FACE"]
    }
  ]
}
```

**Example of *.txt**:
```
0.9
BAG 120 254 20 20 0
FACE 45 23 40 40 1
```

This will result in a `BAG` item at coordinates `120, 254` with width and height set to `20`. The next position is ignored, as it only relates to the `FACE` class.

The format for KITTI labels is explained in the `readme.txt` from the "Object development kit".
Here is the relevant portion:

---

Data Format Description
-----------------------

The data for training and testing can be found in the corresponding folders.
The sub-folders are structured as follows:

  - image_02/ contains the left color camera images (png)
  - label_02/ contains the left color camera label files (plain text files)
  - calib/ contains the calibration for all four cameras (plain text file)

The label files contain the following information, which can be read and
written using the matlab tools (readLabels.m, writeLabels.m) provided within
this devkit. All values (numerical or strings) are separated via spaces,
each row corresponds to one object. The 15 columns represent:

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

Here, 'DontCare' labels denote regions in which objects have not been labeled,
for example because they have been too far away from the laser scanner. To
prevent such objects from being counted as false positives our evaluation
script will ignore objects detected in don't care regions of the test set.
You can use the don't care labels in the training set to avoid that your object
detector is harvesting hard negatives from those areas, in case you consider
non-object regions from the training images as negative examples.

The coordinates in the camera coordinate system can be projected in the image
by using the 3x4 projection matrix in the calib folder, where for the left
color camera for which the images are provided, P2 must be used. The
difference between rotation_y and alpha is, that rotation_y is directly
given in camera coordinates, while alpha also considers the vector from the
camera center to the object center, to compute the relative orientation of
the object with respect to the camera. For example, a car which is facing
along the X-axis of the camera coordinate system corresponds to rotation_y=0,
no matter where it is located in the X/Z plane (bird's eye view), while
alpha is zero only, when this object is located along the Z-axis of the
camera. When moving the car away from the Z-axis, the observation angle
will change.

To project a point from Velodyne coordinates into the left color image,
you can use this formula: x = P2 * R0_rect * Tr_velo_to_cam * y
For the right color image: x = P3 * R0_rect * Tr_velo_to_cam * y

Note: All matrices are stored row-major, i.e., the first values correspond
to the first row. R0_rect contains a 3x3 matrix which you need to extend to
a 4x4 matrix by adding a 1 as the bottom-right element and 0's elsewhere.
Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix
in the same way!

Note, that while all this information is available for the training data,
only the data which is actually needed for the particular benchmark must
be provided to the evaluation server. However, all 15 values must be provided
at all times, with the unused ones set to their default values (=invalid) as
specified in writeLabels.m. Additionally a 16'th value must be provided
with a floating value of the score for a particular detection, where higher
indicates higher confidence in the detection. The range of your scores will
be automatically determined by our evaluation server, you don't have to
normalize it, but it should be roughly linear. If you use writeLabels.m for
writing your results, this function will take care of storing all required
data correctly.

---

## Class Mappings

In general classes are used as plain-text in the `*.txt` files. However one might also use numbers to reduce disk-space. Then the

### The `dontcare` Class
**NOTE:** Class 0 is treated as a special case.

See "Label format" above for a detailed description.
All classes which don't exist in the provided mapping are implicitly mapped to 0.

`Dontcare` elements are in general ignored (however the BrainPlug labeling pipeline should be able to export them with a grayed overlay).
However most networks will use Class 0 as a background class to declare negative examples.


## Store Beard Data

The `beard.py` class comes with a `store` function. This function requires multiple parameters:

* `gen` - Python Generator that yields single data items as arrays: (Image as Numpy Array, Global_data, Meta_data, `BeardDataType` for the current image)
  * `global_data` - Python Dict, where key is the name of the global element (as defined in the config) and value is the actual item value (e.g. `{ 'complexity': 0.5, 'augmented': False }`)
  * `meta_data` - Array that contains python Dicts for each labeled element inside the image. The dicitionary is defined analog to `global_data` with the elements as defined in the `boxes` section of the config.
* `config` - Python Dict that contains the config as in the Json (see in folder `beard_test` for an example)
* `folder` - the folder to store the dataset into
* `clean` - defines if all data already present in `folder` should be deleted
* `debug` - enables console output
* `start_id` - id used for the naming of the first image (used to extend datasets)
