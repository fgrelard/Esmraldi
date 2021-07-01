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

from esmraldi.sliceviewer import SliceViewer
import esmraldi.imzmlio as imzmlio

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
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(sitk.GetArrayFromImage(imRef_bin))
    ax[1].imshow(sitk.GetArrayFromImage(imRegistered_bin))
    plt.show()

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


def dt_mutual_information(imRef, imRegistered, bins=20):
    imRegistered_DT = utils.compute_DT(imRegistered)
    return mutual_information(imRef, imRegistered_DT)

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

def initialize_resampler(fixed, tx):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)
    return resampler

def find_best_transformation(scale_and_rotation, initial_transform, fixed, moving, update_DT=True):
    #Transform with current scale and rotation parameters
    tx = sitk.Transform(initial_transform)
    scale = scale_and_rotation[0]
    rotation = scale_and_rotation[1]
    parameters = list(tx.GetParameters())
    parameters[0] = scale
    parameters[1] = rotation
    tx.SetParameters(parameters)

    #Apply transform
    resampler = initialize_resampler(fixed, tx)
    deformed = resampler.Execute(moving)

    #Compute metric
    if update_DT:
        # metric = dt_mutual_information(fixed, deformed)
        metric = utils.dt_mse(fixed, deformed)
    else:
        # metric = mutual_information(fixed, deformed)
        metric = utils.mse(fixed, deformed)

    return metric

def find_best_translation(translation_vector, initial_transform, fixed, moving):
    #Transform with current scale and rotation parameters
    transform = sitk.Transform(initial_transform)
    parameters = list(transform.GetParameters())
    for i, value in enumerate(translation_vector):
        parameters[i] = value
    transform.SetParameters(parameters)

    #Apply transform
    resampler = initialize_resampler(fixed, transform)
    deformed = resampler.Execute(moving)

    #Compute metric
    metric = utils.mse(fixed, deformed)
    return metric

def register_component_images(fixed_array, moving_array, component_images_array, translation_range=1):
    fixed_itk = sitk.GetImageFromArray(fixed_array)
    dim = fixed_itk.GetDimension()
    translated_component_images = component_images_array.copy()
    translation_array = np.zeros(component_images_array.shape[:-1] + (dim,))

    for i in range(component_images_array.shape[-1]):
        transform = sitk.TranslationTransform(dim)
        component_image = component_images_array[..., i]
        component_image_itk = sitk.GetImageFromArray(component_image)
        x = [0] * dim
        ranges = (slice(0, 1.0, 1.0),) * (dim-2) + (slice(-translation_range, translation_range+1, 1.0),) * 2
        x0 = optimizer.brute(lambda x=x: find_best_translation(x, transform, fixed_itk, component_image_itk), ranges=ranges, finish=None)
        parameters = list(transform.GetParameters())
        parameters[0] = x0[0]
        parameters[1] = x0[1]
        transform.SetParameters(parameters)

        print(parameters)
        resampler = initialize_resampler(fixed_itk, transform)
        deformed_itk = resampler.Execute(component_image_itk)
        deformed_array = sitk.GetArrayFromImage(deformed_itk)
        translated_component_images[..., i] = deformed_array

        threshold_filter = sitk.OtsuThresholdImageFilter()
        threshold_filter.SetInsideValue(0)
        threshold_filter.SetOutsideValue(1)
        deformed_thresholded = threshold_filter.Execute(component_image_itk)
        deformed_thresholded_array = sitk.GetArrayFromImage(deformed_thresholded)
        xy = np.argwhere(deformed_thresholded_array > 0)
        new_xy = np.array(parameters)
        current_xy = translation_array[xy[:, 0], xy[:, 1]]
        translation_array[xy[:, 0], xy[:, 1]] = np.sign(new_xy) * np.maximum(current_xy, np.abs(new_xy))

    translated_moving_array = moving_array.copy()
    sorted_indices = np.argsort(np.linalg.norm(translation_array, axis=-1), axis=None)[::-1]
    for i, ind in enumerate(sorted_indices):
        xy = np.unravel_index(ind, component_images_array.shape[:-1])
        t = translation_array[xy][::-1]
        if not t.any():
            break
        upper_bound = np.array(component_images_array.shape[:-1])-1
        old_xy = (np.minimum(np.maximum(xy + t, [0, 0]), upper_bound)).astype(np.int)
        new_xy = (np.minimum(np.maximum(xy - t, [0, 0]), upper_bound)).astype(np.int)
        x, y = xy
        translated_moving_array[new_xy[0], new_xy[1], ...] = moving_array[x, y, ...]
        translated_moving_array[x, y, ...] = moving_array[old_xy[0], old_xy[1], ...]



    return translated_component_images, translated_moving_array



def register(fixed, moving, number_of_bins, sampling_percentage, find_best_rotation=False, use_DT=True, update_DT=False, normalize_DT=False, seed=sitk.sitkWallClock, learning_rate=1.1, min_step=0.001, relaxation_factor=0.8):
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
        fixed_DT = fixed
        moving_DT = moving

        if use_DT:
            fixed_DT = utils.compute_DT(fixed)
            moving_DT = utils.compute_DT(moving)

        if normalize_DT:
            fixed_DT = utils.normalized_dt(fixed)
            moving_DT = utils.normalized_dt(moving)


        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(sitk.GetArrayFromImage(fixed_DT))
        # ax[1].imshow(sitk.GetArrayFromImage(moving_DT))
        # plt.show()
        tx = sitk.CenteredTransformInitializer(fixed_DT, moving_DT, transform, sitk.CenteredTransformInitializerFilter.MOMENTS)

        tx2 = sitk.CenteredTransformInitializer(fixed_DT, moving_DT, transform, sitk.CenteredTransformInitializerFilter.GEOMETRY)

        x = [0, 0]
        ranges = (slice(0.1, 2.0, 0.1), slice(-3.2, 3.2, 0.05))
        x1, metric1, _, _ = optimizer.brute(lambda x=x: find_best_transformation(x, tx, fixed_DT, moving_DT, update_DT), ranges=ranges, finish=None, full_output=True)

        x2, metric2, _, _ = optimizer.brute(lambda x=x: find_best_transformation(x, tx2, fixed_DT, moving_DT, update_DT), ranges=ranges, finish=None, full_output=True)
        print(metric1, metric2)
        x0 = x1
        if metric2 < metric1:
            x0 = x2
            tx = tx2

        parameters = list(tx.GetParameters())
        parameters[0] = x0[0]
        parameters[1] = x0[1]
        tx.SetParameters(parameters)
        transform = sitk.Similarity2DTransform(tx)
        print(parameters)


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
        resampler = initialize_resampler(fixed, outTx)
        return resampler
    return None
