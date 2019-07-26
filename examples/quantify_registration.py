import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import argparse
from sklearn import metrics
import os
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def precision(im1, im2):
    tp = np.count_nonzero((im2 + im1) == 2)
    allp = np.count_nonzero(im2 == 1)
    return tp * 1.0 / allp

def recall(im1, im2):
    tp = np.count_nonzero((im2 + im1) == 2)
    allr = np.count_nonzero(im1 == 1)
    return tp * 1.0 / allr

def quality_registration(imRef, imRegistered):
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    imRef_bin = otsu_filter.Execute(imRef)
    imRegistered_bin = otsu_filter.Execute(imRegistered)
    p = precision(imRef_bin, imRegistered_bin)
    r = recall(imRef_bin, imRegistered_bin)
    return p, r

def plot_similarity(imRef, imRegistered):
    axes[0].hist(imRef.ravel(), bins=20)
    axes[0].set_title('Histogramme image IRM')
    axes[1].hist(imRegistered.ravel(), bins=20)
    axes[1].set_title('Histogramme image MALDI')
    plt.show()
    # Plotting the signal in the T1 slice against the signal in the T2 slice:

    plt.plot(imRef.ravel(), imRegistered.ravel(), '.')
    plt.xlabel('Intensités IRM')
    plt.ylabel('Intensités MALDI')
    #plt.title('MALDI vs IRM')
    plt.show()
    print(np.corrcoef(imRef.ravel(), imRegistered.ravel())[0, 1])


def quality_registration_size_bin(fixed, registered_dir):
    precision = {}
    l = os.listdir(registered_dir)
    sort_nicely(l)
    for registered in l:
        number = alphanum_key(registered)[1]
        registered = sitk.ReadImage(registered_dir+registered, sitk.sitkFloat32)
        p, r = quality_registration(fixed, registered)
        precision[number] = p
    return zip(*sorted(precision.items()))

def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Fixed image")
parser.add_argument("-r", "--registered", help="Moving image")
parser.add_argument("-b", "--bins", help="number of bins", default=5)

args = parser.parse_args()
fixedname = args.fixed
registeredname = args.registered

args = parser.parse_args()
fixed = sitk.ReadImage(fixedname, sitk.sitkFloat32)

# precision = quality_registration_size_bin(fixed, registeredname)
# plt.plot(*precision, ".b-")
# plt.xlabel("Nombre de classes k")
# plt.ylabel("Precision")
# plt.show()

registered = sitk.ReadImage(registeredname, sitk.sitkFloat32)

simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(registered), sitk.sitkUInt8)
p, r = quality_registration(fixed, registered)
print("Precision=", p, " recall=", r)

fixed_array = sitk.GetArrayFromImage(fixed)
registered_array = sitk.GetArrayFromImage(registered)

# The one-dimensional histograms of the example slices:

fig, axes = plt.subplots(1, 2)


hist_2d, x_edges, y_edges = np.histogram2d(
    fixed_array.ravel(),
    registered_array.ravel(),
    bins=20)



print(mutual_information(hist_2d))
