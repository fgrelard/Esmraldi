import esmraldi.registration as reg
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Input 3D fixed image")
parser.add_argument("-m", "--moving", help="Input 3D moving image")
parser.add_argument("-o", "--output", help="Output image")
parser.add_argument("--points_fixed", nargs="+", type=int, help="Landmarks for fixed image (x1, y1, x2, y2...)", default=[1])
parser.add_argument("--points_moving", nargs="+", type=int, help="Landmarks for moving image (x1, y1, x2, y2...)", default=[1])
parser.add_argument("--crop", action="store_true")

args = parser.parse_args()

fixed_name = args.fixed
moving_name = args.moving
out_name = args.output
points_fixed = args.points_fixed
points_moving = args.points_moving
crop = args.crop

fixed = sitk.GetArrayFromImage(sitk.ReadImage(fixed_name))
moving = sitk.GetArrayFromImage(sitk.ReadImage(moving_name))

deformed = reg.registration_landmarks(fixed, moving, points_fixed, points_moving, crop)

deformed_itk = sitk.GetImageFromArray(deformed)
sitk.WriteImage(deformed_itk, out_name)
