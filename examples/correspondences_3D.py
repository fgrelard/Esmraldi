import esmraldi.segmentation as seg
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Input 3D fixed image")
parser.add_argument("-m", "--moving", help="Input 3D moving image")
parser.add_argument("-o", "--output", help="Output 3D moving image with correspondences found on the z-axis wrt to fixed image")

args = parser.parse_args()

fixed_name = args.fixed
moving_name = args.moving
output_name = args.output

fixed = nib.load(fixed_name).get_fdata()
moving = nib.load(moving_name).get_fdata()

bin_fixed = np.where(fixed > 0, 255, 0)
bin_moving = np.where(moving > 0, 255, 0)

r,t = seg.slice_correspondences(bin_fixed, bin_moving)

plt.plot(range(0, len(r)*2,2), r, range(0, len(t)), t[::-1])
plt.show()
