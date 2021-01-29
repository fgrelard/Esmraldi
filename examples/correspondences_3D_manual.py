import esmraldi.imageutils as utils
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Input 3D fixed image")
parser.add_argument("-m", "--moving", help="Input 3D moving image")
parser.add_argument("-o", "--output", help="Output 3D moving image with correspondences found on the z-axis wrt to fixed image")
parser.add_argument("--resolution_fixed", help="Resolution for fixed image on the z-axis", default=1)
parser.add_argument("--resolution_moving", help="Resolution for moving image on the z-axis", default=1)
parser.add_argument("--spacing_fixed", nargs="+", type=int, help="Spacing (or slice numbers) on the z-axis for fixed image", default=[1])
parser.add_argument("--spacing_moving", nargs="+", type=int, help="Spacing (or slice numbers) on the z-axis for moving image", default=[1])
parser.add_argument("-r", "--reverse", help="Reverse moving image order along the z-axis", action="store_true")


args = parser.parse_args()

fixed_name = args.fixed
moving_name = args.moving
output_name = args.output
resolution_fixed = float(args.resolution_fixed)
resolution_moving = float(args.resolution_moving)
spacing_fixed = args.spacing_fixed
spacing_moving = args.spacing_moving
is_reversed = bool(args.reverse)

fixed = nib.load(fixed_name).get_fdata()
fixed = fixed.reshape(fixed.shape[:3])
moving = nib.load(moving_name).get_fdata()

print(spacing_fixed)
print(spacing_moving)
if len(spacing_fixed) == 1:
    spacing_fixed = float(spacing_fixed[0])
    slicenumber_fixed = [i*spacing_fixed for i in range(fixed.shape[-1])]
else:
    slicenumber_fixed = list(spacing_fixed)
    print(slicenumber_fixed)

if len(spacing_moving) == 1:
    spacing_moving = float(spacing_moving[0])
    slicenumber_moving = [i*spacing_moving for i in range(moving.shape[-1])]
else:
    slicenumber_moving = list(spacing_moving)

correspondences = utils.slice_correspondences_manual(fixed, moving, resolution_fixed, resolution_moving, slicenumber_fixed, slicenumber_moving, is_reversed)
corresponding_images = fixed[..., correspondences]
print(fixed.shape, moving.shape)
correspondence_image = nib.Nifti1Image(corresponding_images, np.eye(4))
correspondence_image.to_filename(output_name)
