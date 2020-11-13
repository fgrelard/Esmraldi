"""
Registration quantification
Computes precision, recall, f-measure
Mutual information,
Superimposes fixed and moving image
"""
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import argparse
import os
import re
import math
import matplotlib.colors as mcolors
import scipy.ndimage
import esmraldi.segmentation as seg
import esmraldi.imageutils as utils

from sklearn import metrics
from esmraldi.registration import *


def tryint(s):
    """
    Casts to integer

    Parameters
    ----------
    s: type
        any variable

    Returns
    ----------
    int
        s variable to int

    """
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]

    Parameters
    ----------
    s: str
        a string

    Returns
    ----------
    list
        split string

    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """
    Sort the given list according to alphanum_key

    Parameters
    ----------
    l: list
        input list

    Returns
    ----------
    list
        sorted list
    """
    l.sort(key=alphanum_key)


def plot_similarity(imRef, imRegistered):
    """
    Plot histogram side-by-side between two images
    Displays correlation coefficient

    Parameters
    ----------
    imRef: np.ndarray
        reference image
    imRegistered: np.ndarray
        moving image

    """
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(imRef.ravel(), bins=20, normed=True)
    axes[0].set_title('Histogramme image IRM')
    axes[1].hist(imRegistered.ravel(), bins=20, normed=True)
    axes[1].set_title('Histogramme image MALDI')
    # axes[2].hist2d(imRef.ravel(), imRegistered.ravel(), bins=20, norm=mcolors.PowerNorm(0.5), normed=True)
    plt.show()
    # Plotting the signal in the T1 slice against the signal in the T2 slice:

    plt.plot(imRef.ravel(), imRegistered.ravel(), '.')
    plt.xlabel('Intensités IRM')
    plt.ylabel('Intensités MALDI')
    #plt.title('MALDI vs IRM')
    plt.show()
    print(np.corrcoef(imRef.ravel(), imRegistered.ravel())[0, 1])


def quality_registration_size_bin(fixed, registered_dir):
    """
    Evaluates registration quality
    through precision on several images

    Parameters
    ----------
    fixed: np.ndarray
        fixed image
    registered_dir: str
        path to registered images

    """
    precision = {}
    l = os.listdir(registered_dir)
    sort_nicely(l)
    for registered in l:
        number = alphanum_key(registered)[1]
        registered = sitk.ReadImage(registered_dir+registered, sitk.sitkFloat32)
        p, r = quality_registration(fixed, registered)
        precision[number] = p
    return zip(*sorted(precision.items()))


def determine_scale(imRef, image):
    imRef_array = sitk.GetArrayFromImage(imRef)

    number_nzs = np.count_nonzero(imRef)
    max_diff = 2**32
    best_size = (imRef_array.shape[1], imRef_array.shape[0])
    for s in range(100):
        size = (imRef_array.shape[1], imRef_array.shape[0])
        size = tuple([round((1+s*0.01)*x) for x in size])
        imScaled = utils.resize(image, size)
        imScaled_array = sitk.GetArrayFromImage(imScaled)
        number_nzs_scaled =  np.count_nonzero(imScaled_array)
        diff = abs(number_nzs - number_nzs_scaled)
        if diff < max_diff:
            max_diff = diff
            best_size = size
    return best_size

def crop_nonzero(image, size):
    x, y = np.nonzero(image)
    xl,xr = x.min(),x.max()
    yl,yr = y.min(),y.max()
    im = image[xl:xr+1, yl:yr+1]
    pad_w, pad_h = (size[0] - im.shape[0])/2, (size[1] - im.shape[1])/2
    print(pad_w, pad_h)
    im = np.pad(im, ((int(pad_w), math.ceil(pad_w)), (int(pad_h), math.ceil(pad_h))), mode='constant')
    return im

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Fixed image")
parser.add_argument("-r", "--registered", help="Moving image")
parser.add_argument("-o", "--original", help="Original before registration image")
parser.add_argument("-b", "--bins", help="number of bins", default=20)
parser.add_argument("-t", "--threshold", help="Threshold for binary image to compute precision and recall (-1 uses Otsu)", default=-1)

args = parser.parse_args()
fixedname = args.fixed
registeredname = args.registered
originalname = args.original
threshold = int(args.threshold)
bins = int(args.bins)

fixed = sitk.ReadImage(fixedname, sitk.sitkFloat32)
original = sitk.ReadImage(originalname, sitk.sitkFloat32)

# precision = quality_registration_size_bin(fixed, registeredname)
# plt.plot(*precision, ".b-")
# plt.xlabel("Nombre de classes k")
# plt.ylabel("Precision")
# plt.show()

registered = sitk.ReadImage(registeredname, sitk.sitkFloat32)

simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(registered), sitk.sitkUInt8)

cimg = sitk.Compose(simg1, simg2, simg1//3.+simg2//1.5)

if cimg.GetDimension() == 2:
    plt.imshow(sitk.GetArrayFromImage(cimg))
    plt.axis('off')
    plt.show()
else:
    plt.imshow(sitk.GetArrayFromImage(cimg[:,:,0]))
    plt.axis('off')
    plt.show()

p, r = quality_registration(fixed, registered, threshold)
f = (2*p*r)/(p+r)
print("Precision=", p, " recall=", r, " fmeasure=", f)

print("Average precision=", np.mean(p), " recall=", np.mean(r), " fmeasure=", np.mean(f))
fixed_array = sitk.GetArrayFromImage(fixed)
registered_array = sitk.GetArrayFromImage(registered)
original_array = sitk.GetArrayFromImage(original)
size = original.GetSize()
scale = determine_scale(original, registered)
scaled_registered = utils.resize(registered, scale)

scaled_registered_array = sitk.GetArrayFromImage(scaled_registered)
scaled_registered_array = crop_nonzero(scaled_registered_array, size)
scaled_registered = sitk.GetImageFromArray(scaled_registered_array)

print(scaled_registered_array.shape, original_array.shape)
print("Non zeros count=",np.count_nonzero(scaled_registered_array), np.count_nonzero(original_array))
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(original_array)
# ax[1].imshow(scaled_registered_array.T)
# plt.show()

# The one-dimensional histograms of the example slices:

#plot_similarity(fixed_array, registered_array)

hist_2d, x_edges, y_edges = np.histogram2d(
    original_array.ravel(),
    scaled_registered_array.ravel(),
    bins=bins)

print("Mutual information = ", mutual_information(original, scaled_registered, bins))
print("Correlation coefficient = ", np.corrcoef(original_array.ravel(), scaled_registered_array.ravel())[0, 1])
