import numpy as np
import SimpleITK as sitk
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage import data, color
from skimage.draw import circle
import matplotlib.pyplot as plt
import math

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


def fmeasure(precision, recall):
    return 2 * precision * recall / (precision + recall)


def mutual_information(imRef, imRegistered):
    """
    Mutual information for joint histogram
    """

    fixed_array = sitk.GetArrayFromImage(imRef)
    registered_array = sitk.GetArrayFromImage(imRegistered)
    hgram, x_edges, y_edges = np.histogram2d(fixed_array.ravel(),
                                             registered_array.ravel(),
                                             bins=20)
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def detect_circle(image, threshold, min_radius, max_radius):
    cond = np.where(image < threshold)
    image_copy = np.copy(image)
    image_copy[cond] = 0
    edges = canny(image_copy, sigma=3, low_threshold=10, high_threshold=40)

    # Detect two radii
    hough_radii = np.arange(min_radius, max_radius, 10)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)
    if len(cx) > 0:
        return cx[0], cy[0], radii[0]
    return -1, -1, -1



def detect_tube(image, threshold=150, min_radius=10, max_radius=50):
    cy, cx, radii = [], [], []
    for i in range(image.shape[0]):
        center_x, center_y, radius = detect_circle(image[i, :,:], threshold, min_radius, max_radius)
        if center_y >= 0:
            cy.append(center_y)
            cx.append(center_x)
            radii.append(radius)
    center_y = np.median(cy)
    center_x = np.median(cx)
    radius = np.median(radii)
    return center_x, center_y, radius

def fill_circle(center_x, center_y, radius, image, color=0):
    image2 = np.copy(image)
    dim = len(image2.shape)
    rr, cc = circle(int(center_y), int(center_x), int(radius), image2.shape[dim-2:])
    if dim == 2:
        image2[rr, cc] = 0
    if dim == 3:
        image2[:, rr,cc] = 0
    return image2


def best_fit(fixed, array_moving, numberOfBins, samplingPercentage):
    width = fixed.GetWidth()
    height = fixed.GetHeight()
    f_max = 0
    index = -1
    best_resampler = None
    for i in range(array_moving.shape[0]):
        moving = sitk.GetImageFromArray(array_moving[i, ...])
        moving = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkFloat32)
        moving.SetSpacing(fixed.GetSpacing())
        try:
            resampler = register(fixed, moving, numberOfBins, samplingPercentage)
            out = resampler.Execute(moving)
        except Exception as e:
            print(e)
        else:
            simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
            simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
            mut = mutual_information(simg1, simg2)
            if (mut > f_max):
                f_max = mut
                index = i
                best_resampler = resampler
    return best_resampler


def resize(image, size):
    dim = len(image.GetSize())
    new_dims = [size for i in range(2)]
    spacing = [image.GetSize()[0]/size for i in range(2)]
    if dim == 3:
        new_dims.append(image.GetSize()[2])
        spacing.append(1)
    resampled_img = sitk.Resample(image,
                                  new_dims,
                                  sitk.Transform(),
                                  sitk.sitkNearestNeighbor,
                                  image.GetOrigin(),
                                  spacing,
                                  image.GetDirection(),
                                  0.0,
                                  image.GetPixelID())
    return resampled_img

def register(fixed, moving, numberOfBins, samplingPercentage):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfBins)
    R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
    #R.SetOptimizerAsRegularStepGradientDescent(0.01,.001,2000)
    R.SetOptimizerAsOnePlusOneEvolutionary(10000)
    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(tx)

    try:
        outTx = R.Execute(fixed, moving)
    except Exception as e:
        print(e)
    else:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(outTx)
        return resampler
    return None
