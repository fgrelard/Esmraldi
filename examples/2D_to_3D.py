import argparse
import os
import re
import nibabel as nib
import numpy as np

import esmraldi.imzmlio as io

import matplotlib.pyplot as plt



def bounding_size(images):
    max_size = images[0].shape
    for im in images:
        shape = im.shape
        max_size = tuple(max(max_size[i], shape[i]) for i in range(len(shape)))
    return max_size

def center_images(images, size):
    shape_3D = size + (len(images),)
    image_3D = np.zeros(shape_3D)
    for i in range(len(images)):
        im = images[i]
        shape = im.shape
        start = tuple((size[i] - shape[i])//2 for i in range(len(size)))
        end = tuple(start[i] + shape[i] for i in range(len(shape)))
        index = tuple(slice(start[i], end[i]) for i in range(len(start)))
        index += (i,)
        image_3D[index] = im
    return image_3D



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input 2D image directories containing NifTI images (.nii)")
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
            if f.endswith(".nii") and re_pattern.match(f):
                list_image_names.append(os.path.join(root, f))

list_image = []
for im_name in list_image_names:
    im = nib.load(im_name)
    im_array = im.get_fdata()
    list_image.append(im_array)

max_size = bounding_size(list_image)
image3D = center_images(list_image, max_size)

nibimg = nib.Nifti1Image(image3D, np.eye(4))
nibimg.to_filename(outname)
