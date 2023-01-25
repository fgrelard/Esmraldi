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

def quality_registration(imRef, imRegistered, threshold=-1, display=False):
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
    if display:
        if imRef_bin.GetDimension() == 2:
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(sitk.GetArrayFromImage(imRef_bin))
            ax[1].imshow(sitk.GetArrayFromImage(imRegistered_bin))
            plt.show()
        else:
            fig, ax = plt.subplots(1,2)
            tracker = SliceViewer(ax, sitk.GetArrayFromImage(imRef_bin), sitk.GetArrayFromImage(imRegistered_bin))
            fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
            plt.show()

    p, r = [], []
    if imRef_bin.GetDimension() > 3:
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
    """
    Computes the mutual information on DT

    Parameters
    ----------
    imRef: sitk.Image
        reference image
    imRegistered: sitk.Image
        target image
    bins: int
        number of bins for mutual information

    Returns
    ----------
    float
        mutual information between DT of images
    """
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
    find_best_rotation: bool
        whether to find the best rotation
    learning_rate: float
        learning rate for gradient descent optimization
    min_step: float
        min step for optimizer
    relaxation_factor: float
        relaxation factor for optimizer


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
    """
    Utility function to setup the resampler
    i.e. the object which can apply a transform
    onto an image.

    Initialized with nearest neighbor interpolation.


    Parameters
    ----------
    fixed: sitk.Image
        reference image
    tx: sitk.Transform
        the transform

    Returns
    ----------
    sitk.ResampleImageFilter
        the resampler

    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)
    return resampler

def find_best_transformation(scale_and_rotation, initial_transform, fixed, moving, update_DT=True):
    """
    Function returning the similarity metric value
    between the reference image and the deformed target image.

    The current transformation parameters (scale and rotation)
    are applied to the target image before estimating the metric,
    i.e. either regular mean squared error, or mean squared error on
    the distance transformed images.

    Parameters
    ----------
    scale_and_rotation: list
        current scale and rotation parameters
    initial_transform: sitk.Transform
        initial transform
    fixed: sitk.Image
        reference image
    moving: sitk.Image
        target image
    update_DT: bool
        whether to re-compute the DT at each step (DT-MSE)


    Returns
    ----------
    float
        similarity metric value

    """
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

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(sitk.GetArrayFromImage(fixed).T)
    # ax[1].imshow(sitk.GetArrayFromImage(deformed).T)
    # plt.show()

    #Compute metric
    if update_DT:
        # metric = dt_mutual_information(fixed, deformed)
        metric = utils.dt_mse(fixed, deformed)
    else:
        # metric = mutual_information(fixed, deformed)
        metric = utils.mse(fixed, deformed)

    return metric


def find_best_translation_scale(scale_and_translation, initial_transform, fixed, moving):
    tx = sitk.Transform(initial_transform)
    scale = scale_and_translation[0]
    translation = scale_and_translation[2:]
    parameters = list(tx.GetParameters())
    parameters[0] = scale
    parameters[1] = 0
    parameters[2] = translation[0]
    parameters[3] = translation[1]

    tx.SetParameters(parameters)

    #Apply transform
    resampler = initialize_resampler(fixed, tx)
    deformed = resampler.Execute(moving)

    fixed_array = sitk.GetArrayFromImage(fixed)
    deformed_array = sitk.GetArrayFromImage(deformed)
    fixed_array[deformed_array==0] = 0

    fixed = sitk.GetImageFromArray(fixed_array)


    metric = mutual_information(fixed, deformed)
    return -metric


def find_best_translation(translation_vector, initial_transform, fixed, moving):
    """
    Function returning the similarity metric value
    between the reference image and the deformed target image.
    This function is used in the context of NMF component image registration.

    The current transformation parameters (only translation)
    are applied to the target image before estimating the metric,
    i.e. mean squared error.

    Parameters
    ----------
    translation_vector: list
        translation vector in nD
    initial_transform: sitk.Transform
        the initial transform
    fixed: sitk.Image
        reference image
    moving: sitk.Image
        moving image

    Returns
    ----------
    float
        similarity metric value
    """
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
    """
    Function used for the registration of NMF component images.

    Translation of component images to match the reference.

    Parameters
    ----------
    fixed_array: np.ndarray
        reference image
    moving_array: np.ndarray
        target image
    component_images_array: np.ndarray
        NMF component images: shape (w, h, number_of_components)
    translation_range: int
        translation search between [-t, t], default 1

    Returns
    ----------
    tuple(np.ndarray, np.ndarray)
        The translated NMF component images, the translation map

    """
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



def register(fixed, moving, number_of_bins, sampling_percentage, find_best_rotation=False, use_DT=True, update_DT=True, normalize_DT=False, seed=1, learning_rate=1.1, min_step=0.001, relaxation_factor=0.8):
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
    find_best_rotation: bool
        whether to use exhaustive search to find the best scaling and rotation parameters
    use_DT: bool
        whether to use the distance transformations of images during exhaustive optimization (if find_best_rotation==True)
    update_DT: bool
        whether to update the distance transformations during exhaustive optimization (if find_best_rotation==True)
    normalize_DT: bool
        whether to normalize the distance transformations during exhaustive optimization ((if find_best_rotation==True)
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

    if use_DT:
        fixed_DT = utils.compute_DT(fixed)
        moving_DT = utils.compute_DT(moving)

    if find_best_rotation:
        fixed_DT = fixed
        moving_DT = moving

        if normalize_DT:
            fixed_DT = utils.normalized_dt(fixed)
            moving_DT = utils.normalized_dt(moving)

        tx = sitk.CenteredTransformInitializer(fixed_DT, moving_DT, transform, sitk.CenteredTransformInitializerFilter.MOMENTS)

        tx2 = sitk.CenteredTransformInitializer(fixed_DT, moving_DT, transform, sitk.CenteredTransformInitializerFilter.GEOMETRY)

        x = [0, 0]
        ranges = (slice(0.1, 2.0, 0.1), slice(-3.2, 3.2, 0.05))
        ranges = (slice(0.6, 0.7, 0.01), slice(-3.2, 3.2, 0.05))
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
    if use_DT:
        R.SetMetricAsMeanSquares()

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=100,
        relaxationFactor=relaxation_factor,
        gradientMagnitudeTolerance = 1e-5,
        maximumStepSizeInPhysicalUnits = 0.0)

    # R.SetOptimizerAsOnePlusOneEvolutionary()

    if not find_best_rotation:
        transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            transform,
            sitk.CenteredTransformInitializerFilter.MOMENTS)

    R.SetInitialTransform(transform)

    try:
        if use_DT:
            outTx = R.Execute(fixed_DT, moving_DT)
        else:
            outTx = R.Execute(fixed, moving)
    except Exception as e:
        outTx = transform
    print(outTx.GetParameters())
    resampler = initialize_resampler(fixed, outTx)
    return resampler
    return None


def preprocess_image(image):
    is_ms_image = hasattr(image, "image")
    processed_image = image
    shape = ((2,) if image.ndim == 3 else ()) + (0, 1)
    if is_ms_image:
        processed_image = processed_image.image
    else:
        processed_image = np.transpose(image, shape)
    return processed_image, is_ms_image

def crop_image(image, points):
    lower = round(np.amin(points[::2])), round(np.amin(points[1::2]))
    upper = round(np.amax(points[::2])+1), round(np.amax(points[1::2])+1)
    if image.ndim == 2:
        image = image[lower[1]:upper[1], lower[0]:upper[0]]
    else:
        image = image[:, lower[1]:upper[1], lower[0]:upper[0]]
    new_points = [int(p - lower[i%2]) for i, p in enumerate(points)]
    return image, new_points

def apply_registration(fixed, register, landmark_transform):
    fixed_dim = fixed.ndim
    dim = register.ndim
    size = np.array(register.shape)[::-1]

    if fixed_dim == 2:
        fixed_itk = sitk.GetImageFromArray(fixed)
        resampler = initialize_resampler(fixed_itk, landmark_transform)
    if fixed_dim == 3:
        fixed_itk = sitk.GetImageFromArray(fixed[0, ...])
        resampler = initialize_resampler(fixed_itk, landmark_transform)

    if dim == 2:
        register_itk = sitk.GetImageFromArray(register)
        deformed_itk = resampler.Execute(register_itk)
    elif dim == 3:
        slices = []
        for i in range(size[2]):
            img_slice = sitk.GetImageFromArray(register[i, ...])
            img_slice.SetSpacing([1, 1])
            out_slice = resampler.Execute(img_slice)
            out_slice = sitk.JoinSeries(out_slice)
            slices.append(out_slice)
        stackmaker = sitk.TileImageFilter()
        stackmaker.SetLayout([1, 1, 0])
        deformed_itk = stackmaker.Execute(slices)

    deformed = sitk.GetArrayFromImage(deformed_itk)
    return deformed

def registration_landmarks(fixed, moving, points_fixed, points_moving, to_crop=False):
    fixed, is_ms_fixed = preprocess_image(fixed)
    moving, is_ms_moving = preprocess_image(moving)
    if to_crop:
        fixed, points_fixed = crop_image(fixed, points_fixed)

    landmark_transform = sitk.LandmarkBasedTransformInitializer(sitk.AffineTransform(2), points_fixed, points_moving)

    deformed = apply_registration(fixed, moving, landmark_transform)
    return deformed
