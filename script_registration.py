import argparse
import os
import sys
import subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--min_learning_rate", help="Learning rate", default=0)
parser.add_argument("--max_learning_rate", help="Learning rate", default=2)
parser.add_argument("--min_relaxation_factor", help="Relaxation factor", default=0.9)
parser.add_argument("--max_relaxation_factor", help="Relaxation factor", default=0.9)
parser.add_argument("-f", "--fixed", help="Fixed image")
parser.add_argument("-m", "--moving", help="Moving image")
parser.add_argument("-r", "--register", help="Registration image or directory containing images (same number as in moving and fixed)")
parser.add_argument("-o", "--output", help="Output")
parser.add_argument("-b", "--bins", help="number per bins", default=10)
parser.add_argument("-s", "--symmetry", help="best fit with flipped image", action="store_true", default=False)
parser.add_argument("--pattern", help="Pattern to match image if registername is a directory", default=".*")
parser.add_argument("--level", help="Level to find files", default=2)
parser.add_argument("--resize", help="Resize the moving image to match the fixed image size", action="store_true")
parser.add_argument("--best_rotation", help="Initialize registration by finding the best rotation angle between the two images", action="store_true")
parser.add_argument("--sampling_percentage", help="Sampling percentage", default=0.1)
parser.add_argument("--min_step", help="Minimum step for gradient descent", default=0.001)
parser.add_argument("--step_realign", help="Step to realign mzs for 3D volumes", default=0.05)
parser.add_argument("--apply_mask", help="Apply mask from segmentation (0 valued-pixels in the segmentation)", action="store_true")
args = parser.parse_args()

fixedname = args.fixed
movingname = args.moving
outputname = args.output
registername = args.register
bins = int(args.bins)
flipped = args.symmetry
is_resize = bool(args.resize)
is_best_rotation = bool(args.best_rotation)

min_learning_rate = float(args.min_learning_rate)
max_learning_rate = float(args.max_learning_rate)
min_relaxation_factor = float(args.min_relaxation_factor)
max_relaxation_factor = float(args.max_relaxation_factor)
sampling_percentage = float(args.sampling_percentage)
min_step = float(args.min_step)
pattern = args.pattern
level = int(args.level)
step_realign = float(args.step_realign)
apply_mask = args.apply_mask

step = 0.1
step_lr = 0.3
for relax in np.arange(min_relaxation_factor, max_relaxation_factor+step, step):
    for lr in np.arange(min_learning_rate, max_learning_rate+step_lr, step_lr):
        print(relax, lr)
        cmd = "python3 -m examples.registration -f " + fixedname + " --moving " + movingname + " -r " + registername + " --relaxation_factor " + str(relax) + " --learning_rate " + str(lr) + " -s --min_step 0.00001 -o " + outputname + " --resize --best_rotation"
        subprocess.call(cmd, shell=True)
