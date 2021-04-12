"""
Module for the registration of two images
"""

import math
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import esmraldi.segmentation as seg
import esmraldi.imageutils as utils
from scipy.ndimage.morphology import distance_transform_edt

import scipy.optimize as optimizer

def precision(im1, im2):
    """
    Precision between two images
    defined as card(im1 \cap im2)/card(im2)

    Parameters
    ----------
    im1: np.ndarray
        first binary image
    im2: np.ndarray
        second binary image

    Returns
    ----------
    float
        precision value

    """
    tp = np.count_nonzero((im2 + im1) == 2)
    allp = np.count_nonzero(im2 == 1)
    return tp * 1.0 / allp

def recall(im1, im2):
    """
    Recall between two images
    defined as card(im1 \cap im2)/card(im1)

    Parameters
    ----------
    im1: np.ndarray
        first binary image
    im2: np.ndarray
        second binary image

    Returns
    ----------
    float
        recall value
    """
    tp = np.count_nonzero((im2 + im1) == 2)
    allr = np.count_nonzero(im1 == 1)
    return tp * 1.0 / allr

def quality_registration(imRef, imRegistered, threshold=-1):
    """
    Evaluates registration quality.

    Binarizes images,
    then computes recall and precision.

    Parameters
    ----------
    imRef: np.ndarray
        reference (fixed) image
    imRegistered: np.ndarray
        deformable (moving) image - after registration
    threshold: int
        threshold to get binary images. A value
        of -1 means using Otsu thresholding scheme.

    Returns
    ----------
    tuple
        precision and recall values
    """
    if threshold == -1:
        threshold_filter = sitk.OtsuThresholdImageFilter()
    else:
        threshold_filter = sitk.BinaryThresholdImageFilter()
        threshold_filter.SetLowerThreshold(0)
        threshold_filter.SetUpperThreshold(threshold)

    threshold_filter.SetInsideValue(0)
    threshold_filter.SetOutsideValue(1)
    imRef_bin = threshold_filter.Execute(imRef)
    imRegistered_bin = threshold_filter.Execute(imRegistered)

    p, r = [], []
    if imRef_bin.GetDimension() > 2:
        for i in range(imRef_bin.GetSize()[-1]):
            p.append(precision(imRef_bin[:,:,i], imRegistered_bin[:,:,i]))
            r.append(recall(imRef_bin[:,:,i], imRegistered_bin[:,:,i]))
    else:
        p.append(precision(imRef_bin, imRegistered_bin))
        r.append(recall(imRef_bin, imRegistered_bin))
    return np.array(p), np.array(r)


def fmeasure(precision, recall):
    """
    Computes the F-Measure, or F1-score,
    that is the harmonic mean
    of the precision and recall.

    Parameters
    ----------
    precision: float
        precision value
    recall: float
        recall value

    Returns
    ----------
    float
        fmeasure

    """
    return 2 * precision * recall / (precision + recall)


def mutual_information(imRef, imRegistered, bins=20):
    """
    Mutual information for joint histogram
    based on entropy computation

    Parameters
    ----------
    imRef: np.ndarray
        reference (fixed) image
    imRegistered: np.ndarray
        deformable (moving)image
    bins: int
        number of bins for joint histogram

    Returns
    ----------
    float
        mutual information

    """

    fixed_array = sitk.GetArrayFromImage(imRef)
    registered_array = sitk.GetArrayFromImage(imRegistered)
    hgram, x_edges, y_edges = np.histogram2d(fixed_array.ravel(),
                                             registered_array.ravel(),
                                             bins=bins)
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def best_fit(fixed, array_moving, number_of_bins, sampling_percentage, find_best_rotation=False, learning_rate=1.1, min_step=0.001, relaxation_factor=0.8):
    """
    Finds the best fit between variations of the same image
    according to mutual information measure.

    Different variations of the image (e.g. symmetry)
    are stored in the
    first dimension of the array.

    Parameters
    ----------
    fixed: np.ndarray
        reference (fixed) image
    array_moving: np.ndarray
        3D deformable (moving) image
    number_of_bins: int
        number of bins for sampling
    sampling_percentage: float
        proportion of points to consider in sampling

    Returns
    ----------
    sitk.ImageRegistrationMethod
        registration object
    """
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
            resampler = register(fixed, moving, number_of_bins, sampling_percentage, find_best_rotation=find_best_rotation, learning_rate=learning_rate, min_step=min_step, relaxation_factor=relaxation_factor)
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
    return best_resampler, index


def dt_mse(fixed, moving):
    moving_dt = utils.compute_DT(moving)
    fixed_array = sitk.GetArrayFromImage(fixed)
    moving_dt_array = sitk.GetArrayFromImage(moving_dt)
    diff_sq = (moving_dt_array - fixed_array) ** 2
    return np.mean(diff_sq)


def find_best_transformation(scale_and_rotation, initial_transform, fixed, moving):
    #Transform with current scale and rotation parameters
    tx = sitk.Transform(initial_transform)
    scale = scale_and_rotation[0]
    rotation = scale_and_rotation[1]
    parameters = list(tx.GetParameters())
    parameters[0] = scale
    parameters[1] = rotation
    tx.SetParameters(parameters)

    #Apply transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)
    deformed = resampler.Execute(moving)

    #Compute metric
    metric = dt_mse(fixed, deformed)
    return metric


def register(fixed, moving, number_of_bins, sampling_percentage, find_best_rotation=False, seed=sitk.sitkWallClock, learning_rate=1.1, min_step=0.001, relaxation_factor=0.8):
    """
    Registration between reference (fixed)
    and deformable (moving) images.

    The transform is initialized with moments

    Metric: mutual information

    Optimization: gradient descent

    Interpolation: nearest neighbor

    Parameters
    ----------
    fixed: np.ndarray
        reference (fixed) image
    moving: np.ndarray
        deformable (moving) image
    number_of_bins: int
        number of bins for sampling
    sampling_percentage: float
        proportion of points to consider in sampling
    seed: int
        seed for metric sampling
    learning_rate: float
        learning rate for gradient descent optimizer
    min_step: float
        minimum step: stop criterion for the optimizer
    relaxation_factor: float
        relaxation factor for the parameters of the transform between each step of the optimizer

    Returns
    ----------
    sitk.ImageRegistrationMethod
        registration object

    """
    R = sitk.ImageRegistrationMethod()
    transform = sitk.Similarity2DTransform()

    if fixed.GetDimension()==3:
        transform = sitk.Similarity3DTransform()

    if find_best_rotation:
        fixed_DT = utils.compute_DT(fixed)
        moving_DT = utils.compute_DT(moving)

        R.SetMetricAsMeanSquares()

        R.SetOptimizerAsExhaustive(numberOfSteps=[9,32,0,0,0,0], stepLength=0.1)

        tx = sitk.CenteredTransformInitializer(fixed_DT, moving_DT, transform, sitk.CenteredTransformInitializerFilter.MOMENTS)

        x = [0, 0]
        ranges = (slice(0.1, 2.0, 0.1), slice(-3.2, 3.2, 0.1))
        x0 = optimizer.brute(lambda x=x: find_best_transformation(x, tx, fixed_DT, moving_DT), ranges=ranges, finish=None)


        parameters = list(tx.GetParameters())
        parameters[0] = x0[0]
        parameters[1] = x0[1]
        tx.SetParameters(parameters)
        transform = sitk.Similarity2DTransform(tx)

    R.SetMetricAsMattesMutualInformation(number_of_bins)
    R.SetMetricSamplingPercentage(sampling_percentage, seed)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=100,
        relaxationFactor=relaxation_factor,
        gradientMagnitudeTolerance = 1e-5,
        maximumStepSizeInPhysicalUnits = 0.0)

    if not find_best_rotation:
        transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            transform,
            sitk.CenteredTransformInitializerFilter.MOMENTS)

    R.SetInitialTransform(transform)

    try:
        outTx = R.Execute(fixed, moving)
    except Exception as e:
        print(e)
    else:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(outTx)
        return resampler
    return None
