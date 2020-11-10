"""
Compare quality of segmentation with
curvature estimation
"""
import numpy as np
# import similaritymeasures
import scipy.ndimage
import esmraldi.spectraprocessing as sp
import scipy.signal as signal
import argparse
import csv
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats.stats import spearmanr
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
sns.set(style="darkgrid")

# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def distance_two_distributions(dis1, dis2, delta):
    """
    Distance between two distributions
    as average pointwise vertical distance

    Parameters
    ----------
    dis1: np.ndarray
        first distribution
    dis2: np.ndarray
        second distribution

    Returns
    ----------
    float
        average distance
    """
    sum = 0
    n1 = len(dis1)
    for i in range(delta,n1-delta):
        val1 = dis1[i]
        val2 = dis2[i-delta:i+delta]
        distance = np.amin(np.abs(val2 - val1))
        sum += distance
    return sum / n1



def find_peaks(data, prominence, w):
    """
    Find peaks in distribution
    according to prominence

    Parameters
    ----------
    data: np.ndarray
        data
    prominence: int
        threshold on prominence
    w: int
        size of window

    Returns
    ----------
    list
        peak list

    """
    peaks, _ = signal.find_peaks(tuple(data),
                                 height=prominence,
                                 wlen=w,
                                 distance=1)
    return peaks

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def h_distance(mri, maldi):
    t = np.array(mri)
    current_values = np.array(maldi)
    closest = current_values[(np.abs(t[:, None] - current_values).argmin(axis=1))]
    diff = np.mean(np.abs(t - closest))
    return diff

def h_distance_tol(mri, maldi, tol):
    t = np.array(mri)
    o = np.array(maldi)
    cap_o = o[(np.abs(t[:, None] - o) < tol).any(0)]
    closest = t[(np.abs(cap_o[:, None] - t).argmin(axis=1))]
    diff = np.mean(np.abs(closest - cap_o))
    return diff

def fhmeasure(mri, maldi, tol):
    t = np.array(mri)
    o = np.array(maldi)
    common_o = o[(np.abs(t[:, None] - o) < tol).any(0)]
    common_t = t[(np.abs(o[:, None] - t)< tol).any(0)]
    p = len(common_t) * 1.0 / len(t)
    r = len(common_o) * 1.0 / len(t)
    # print(common_o, o, common_t, t)
    # print(len(common_o))
    if p+r == 0:
        return 0
    # print(p, r)
    f = 2 * p * r / (p+r)
    return f

def fhmeasure_aligned(image_curvature, mri, image_sdp_name, threshold=0.2, sigma=2):
    X_image, Y_image = coordinates_from_sdp(image_sdp_name)
    translation_image = np.array(Y_image).argmax()

    image_curvature = scipy.ndimage.gaussian_filter1d(np.copy(image_curvature), sigma)
    indices_image = (find_peaks(image_curvature, threshold, 50)).tolist()
    mri_curvature = scipy.ndimage.zoom(mri, len(image_curvature)/len(mri), order=3)
    mri_curvature = scipy.ndimage.gaussian_filter1d(np.copy(mri_curvature), sigma)
    indices_mri = (find_peaks(mri_curvature, threshold, 50)).tolist()
    h_d, trans = best_hdistance(indices_mri, indices_image, len(image_curvature))
    image_curvature = np.roll(image_curvature, -trans)
    indices_image = (find_peaks(image_curvature, threshold, 50)).tolist()
    h_d = spearmanr(mri_curvature, image_curvature)[0]
    # h_d = fhmeasure(indices_mri, indices_image, 35)
    return h_d

def h_distances(image_names, mri):
    values = []
    for image_name in image_names:
        image_curvature = np.loadtxt(image_name)
        image_sdp_name = image_name.split("_curvature.txt")[0] + "_ensured.sdp"
        X_image, Y_image = coordinates_from_sdp(image_sdp_name)
        translation_image = np.array(Y_image).argmax()
        if image_name in image_names[-2:]:
            translation_image-=50
        # translation_image = np.array(X_image).argmin()
        image_curvature = np.roll(image_curvature, -translation_image)
        if (X_image[translation_image] < X_image[translation_image+1]):
            image_curvature[1:] = np.flip(image_curvature[1:])

        image_curvature = scipy.ndimage.gaussian_filter1d(np.copy(image_curvature), 1)
        indices_image = (find_peaks(image_curvature, 0.12, 50)).tolist()
        mri_curvature = scipy.ndimage.zoom(mri, len(image_curvature)/len(mri), order=1)
        mri_curvature = scipy.ndimage.gaussian_filter1d(np.copy(mri_curvature), 2)
        indices_mri = (find_peaks(mri_curvature, 0.2, 50)).tolist()
        h_d = h_distance(indices_mri, indices_image)
        # h_d, _ = best_hdistance(indices_mri, indices_image, len(image_curvature))
        values.append(h_d)
    return values

def best_hdistance(mri, maldi, length):
    t = np.array(mri)
    min_diff = 2**32
    for delta in range(length):
        current_values = np.array([(length+peak-delta)%length for peak in maldi])
        diff = h_distance(t, current_values)
        if diff < min_diff:
            best_delta = delta
            min_diff = diff
    return min_diff, best_delta

def coordinates_from_sdp(name):
    with open(name) as f:
        reader = csv.reader(f, delimiter=" ")
        X, Y = [], []
        for row in reader:
            try:
                X.append(int(row[0]))
                Y.append(int(row[1]))
            except Exception as e:
                continue
    return X, Y


def display_starting_point(data, X, Y, translation):
    upx,upy = np.amax(X, axis=0), np.amax(Y, axis=0)
    M = np.zeros(shape=(upx+1, upy+1))
    values = [(len(data)+i-translation)%len(data) for i in range(len(data))]
    M[X, Y] = values
    plt.imshow(M)
    plt.show()

def best_spearman(image_curvature, mri_curvature,sigma=1.5):
    image_curvature = scipy.ndimage.gaussian_filter1d(np.copy(image_curvature), sigma)
    mri_curvature = scipy.ndimage.zoom(mri_curvature, len(image_curvature)/len(mri_curvature), order=2)
    mri_curvature = scipy.ndimage.gaussian_filter1d(np.copy(mri_curvature), sigma)
    best_s = 0
    for i in range(len(image_curvature)):
        tmp = np.roll(image_curvature, i)
        s = spearmanr(mri_curvature, tmp)[0]
        if s > best_s:
            best_s = s
    return best_s


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Fixed curvature")
parser.add_argument("-m", "--moving", help="Moving curvature directory")
args = parser.parse_args()

fixed_name = args.fixed
moving_name = args.moving
fixed_sdp_name = fixed_name.split("_curvature.txt")[0] + "_ensured.sdp"

# moving_sdp_name = moving_name.split("_curvature.txt")[0] + "_ensured.sdp"

mri_curvature = np.loadtxt(fixed_name)
# maldi_curvature = np.loadtxt(moving_name)

image_names = []
for file in sorted_alphanumeric(os.listdir(moving_name)):
    if file.endswith(".txt"):
        print(file)
        image_names.append(moving_name+file)

X_fixed, Y_fixed = coordinates_from_sdp(fixed_sdp_name)
# X_moving, Y_moving = coordinates_from_sdp(moving_sdp_name)

translation_fixed = 240
translation_fixed = np.array(X_fixed).argmin()
# display_starting_point(mri_curvature, X_fixed, Y_fixed, translation_fixed)



#Shift distribution
mri_curvature = np.roll(mri_curvature, -translation_fixed)

fhs, lengths = [], []
for image_name in image_names:
    image_curvature = np.loadtxt(image_name)
    image_sdp_name = image_name.split("_curvature.txt")[0] + "_ensured.sdp"
    fh = fhmeasure_aligned(image_curvature, mri_curvature, image_sdp_name, threshold=0.13, sigma=1)
    length = len(image_curvature)
    fhs.append(fh)
    lengths.append(length)

fig,ax1 = plt.subplots()
ax1.set_xlabel("Image number")
ax1.set_ylabel("Length")
ln1 = ax1.plot(lengths, "tab:blue")
ax2 = ax1.twinx()
ln2 = ax2.plot(fhs, "tab:orange", lw=2, alpha=0.7)
ax2.set_ylabel("Curvature F-measure")
lns = ln1 + ln2
labs = ["Length", "Curvature F-measure"]
ax1.legend(lns, labs, loc=0)
ax2.grid(False)
# ax2.figure.legend()
plt.show()
