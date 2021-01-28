import argparse
import os
import re
import nibabel as nib
import numpy as np

import esmraldi.imzmlio as io
import esmraldi.imageutils as utils

import matplotlib.pyplot as plt
import SimpleITK as sitk


def bounding_size(images):
    max_size = images[0].shape
    for im in images:
        shape = im.shape
        max_size = tuple(max(max_size[i], shape[i]) for i in range(len(shape)))
    return max_size




parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input 2D image directories containing NifTI images (.nii or .tif)")
parser.add_argument("-o", "--output", help="Output peak selected imzML")
parser.add_argument("-r", "--recursive", help="Traverse input directory recursively", action="store_true" )
parser.add_argument("-p", "--pattern", help="Selects images fitting this regexp pattern (default=*)", default="*")

args = parser.parse_args()

inputname = args.input
outname = args.output
is_recursive = args.recursive
pattern = args.pattern

re_pattern = re.compile(pattern)

list_image_names = []

for root, dirs, files in os.walk(inputname):
    first_level = (root.count(os.sep) - inputname.count(os.sep) == 0)
    if (not is_recursive and first_level) or is_recursive:
        for f in files:
            if (f.endswith(".nii") or f.endswith(".tif")) and re_pattern.match(f):
                list_image_names.append(os.path.join(root, f))

list_image = []
for im_name in list_image_names:
    if im_name.endswith(".nii"):
        im = nib.load(im_name)
        im_array = im.get_fdata()
    else:
        image_itk = sitk.ReadImage(im_name, sitk.sitkFloat32)
        im_array = sitk.GetArrayFromImage(image_itk)
    list_image.append(im_array)

max_size = bounding_size(list_image)
image3D = utils.center_images(list_image, max_size)

if outname.endswith(".nii"):
    nibimg = nib.Nifti1Image(image3D, np.eye(4))
    nibimg.to_filename(outname)
else:
    sitk.WriteImage(sitk.Cast(sitk.GetImageFromArray(image3D.T), sitk.sitkFloat32), outname)
