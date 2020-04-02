import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import argparse
from sklearn import metrics
import os
import re
from src.registration import *
import matplotlib.colors as mcolors

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


def plot_similarity(imRef, imRegistered):
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
    precision = {}
    l = os.listdir(registered_dir)
    sort_nicely(l)
    for registered in l:
        number = alphanum_key(registered)[1]
        registered = sitk.ReadImage(registered_dir+registered, sitk.sitkFloat32)
        p, r = quality_registration(fixed, registered)
        precision[number] = p
    return zip(*sorted(precision.items()))




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

cimg = sitk.Compose(simg1, simg2, simg1//3.+simg2//1.5)
plt.imshow(sitk.GetArrayFromImage(cimg))
plt.axis('off')
plt.show()
p, r = quality_registration(fixed, registered)
f = (2*p*r)/(p+r)
print("Precision=", p, " recall=", r, " fmeasure=", f)

fixed_array = sitk.GetArrayFromImage(fixed)
registered_array = sitk.GetArrayFromImage(registered)

# The one-dimensional histograms of the example slices:

#plot_similarity(fixed_array, registered_array)

hist_2d, x_edges, y_edges = np.histogram2d(
    fixed_array.ravel(),
    registered_array.ravel(),
    bins=20)



print("Mutual information = ", mutual_information(fixed, registered))
