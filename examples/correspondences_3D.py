import esmraldi.segmentation as seg
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from dtw import *

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Input 3D fixed image")
parser.add_argument("-m", "--moving", help="Input 3D moving image")
parser.add_argument("-o", "--output", help="Output 3D moving image with correspondences found on the z-axis wrt to fixed image")
parser.add_argument("-s", "--sigma", help="Standard deviation for gaussian smoothing", default=0.01)
parser.add_argument("-c", "--continuity", help="Enforce continuity in the sequence of slices", action="store_true")
parser.add_argument("-r", "--reverse", help="Reverse moving image order along the z-axis", action="store_true")


args = parser.parse_args()

fixed_name = args.fixed
moving_name = args.moving
output_name = args.output
sigma = float(args.sigma)
is_continuity = bool(args.continuity)
is_reversed = bool(args.reverse)

fixed = nib.load(fixed_name).get_fdata()
moving = nib.load(moving_name).get_fdata()

bin_fixed = np.where(fixed > 0, 255, 0)
bin_moving = np.where(moving > 0, 255, 0)

correspondences = seg.slice_correspondences(bin_fixed, bin_moving, sigma, is_reversed, is_continuity)

corresponding_images = fixed[..., correspondences]

correspondence_image = nib.Nifti1Image(corresponding_images, np.eye(4))
correspondence_image.to_filename(output_name)
