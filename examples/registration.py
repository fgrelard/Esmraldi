"""
Registration example

Rigid registration
Transform=rigid, centers initialized with moments
Metric=Mattes mutual information
Optimization=Gradient descent
"""
import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import numpy as np
import esmraldi.imzmlio as imzmlio
import os
import re
import esmraldi.registration as reg
import esmraldi.segmentation as segmentation
import esmraldi.spectraprocessing as sp
import esmraldi.imageutils as utils
import esmraldi.utils as ut

from esmraldi.sliceviewer import SliceViewer

def command_iteration(method) :
    """
    Callback called after each registration iteration

    Parameters
    ----------
    method: sitk.ImageRegistrationMethod
        the registration method

    """
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))



def resample(image, transform):
    """
    Applies a transform to an image


    Parameters
    ----------
    image: sitk.Image
        input image
    transform: sitk.Transform
        a transform

    Returns
    ----------
    sitk.Image
        transformed image

    """
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)




def registration_number_bins(outputname, min_bins, max_bins):
    """
    Registration over a range of bins
    for the sampling strategy

    Parameters
    ----------
    outputname: str
        output image name
    min_bins: int
        minimum number of bins
    max_bins: int
        maximum number of bins

    """
    for i in range(min_bins, max_bins):
        outname = os.path.splitext(outputname)[0] + "_bins" + str(i) +".png"
        try:
            resampler = reg.register(fixed, moving, i)
            out = resampler.Execute(moving)
            out = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
            sitk.WriteImage(out, outname)
        except Exception as e:
            pass

def extract_slice_itk(image, index):
    extract = sitk.ExtractImageFilter()
    size = list(image.GetSize())
    size[2] = 0
    extract.SetSize(size)
    extract.SetIndex([0, 0, index])
    return extract.Execute(image)


def register2D(fixed, moving, numberOfBins, is_best_rotation=False, array_moving=None, flipped=False, sampling_percentage=0.1, learning_rate=1.1, min_step=0.001, relaxation_factor=0.8):
    to_flip = False
    print("register 2D")
    # Flip axis and choose best fit during registration
    if flipped:
        # Construct a 3D image
        # 2 slices = original + flipped
        flipped_moving = sitk.Flip(moving, (True, False))
        moving_and_flipped = np.zeros((2, moving.GetSize()[1], moving.GetSize()[0]), dtype=np.float32)
        moving_and_flipped[0, ...] = sitk.GetArrayFromImage(moving)
        moving_and_flipped[1, ...] = sitk.GetArrayFromImage(flipped_moving)
        array_flipped = moving_and_flipped
        #determines if symmetry of image is a better match
        best_resampler, index = reg.best_fit(fixed, array_flipped, numberOfBins, sampling_percentage, find_best_rotation=is_best_rotation, learning_rate=learning_rate, min_step=min_step, relaxation_factor=relaxation_factor)
        if index == 1:
            to_flip = True
    else:
        #normal registration
        best_resampler = reg.register(fixed, moving, numberOfBins, sampling_percentage, find_best_rotation=is_best_rotation, learning_rate=learning_rate, min_step=min_step, relaxation_factor=relaxation_factor)
    return best_resampler, to_flip


def apply_registration(image, best_resampler, to_flip):
    try:
        if to_flip:
            image = sitk.Flip(image, (True, False))
            image = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
        out = best_resampler.Execute(image)
    except Exception as e:
        print("Problem with best_resampler")
        print(e)
        return None
    return out

def apply_registration_imzml(image, best_resampler, to_flip, fixed_size):
    size = image.GetSize()
    pixel_type = image.GetPixelID()

    outRegister = sitk.Image(fixed_size[0], fixed_size[1], size[2], pixel_type )
    for i in range(size[2]):
        slice = image[:,:,i]
        slice.SetSpacing([1, 1])

        outSlice = apply_registration(slice, best_resampler, to_flip)
        outSlice = sitk.JoinSeries(outSlice)
        outRegister = sitk.Paste(outRegister, outSlice, outSlice.GetSize(), destinationIndex=[0,0,i])
    return outRegister

def read_image_to_register(registername, is_imzml, to_flip, moving):
    mz = None
    if is_imzml:
        imzml = imzmlio.open_imzml(registername)
        current_s = imzmlio.get_full_spectra(imzml)
        max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
        max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
        max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
        array = imzmlio.get_images_from_spectra(current_s, (max_x, max_y, max_z)).T
        register = sitk.GetImageFromArray(array)
        mz, _ = imzml.getspectrum(0)
    else:
        register = sitk.ReadImage(registername, sitk.sitkFloat32)
    if to_flip:
        dim = register.GetDimension()
        flip = (True, False) + ((False,) if dim > 2 else ())
        register = sitk.Flip(register, flip)

    dim = register.GetDimension()
    register.SetSpacing([1 for i in range(dim)])

    identity = np.identity(dim).tolist()
    flat_list = [item for sublist in identity for item in sublist]
    direction = tuple(flat_list)
    register.SetDirection(flat_list)

    return register, mz



def extract_image_from_directories(path, pattern, level=2):
    list_image_names = []
    re_pattern = re.compile(pattern)
    for root, dirs, files in ut.walklevel(path, level):
        for f in files:
            if re_pattern.match(f):
                list_image_names.append(os.path.join(root, f))
    return list_image_names

def get_resampler(is_different_resampler, best_resamplers, flips, index):
    to_flip = False
    if is_different_resampler:
        best_resampler = best_resamplers[index]
        to_flip = flips[index]
    else:
        best_resampler = best_resamplers[0]
        to_flip = flips[0]
    return best_resampler, to_flip


def realign_image(image, reference):
    image_centered_itk = image
    dim = image.GetDimension()
    if dim == 3 and \
       (image.GetSize()[0] != reference.GetSize()[0] or \
        image.GetSize()[1] != reference.GetSize()[1]):
        image_centered = utils.center_images(np.transpose(sitk.GetArrayFromImage(image), (0, 2, 1)), (reference.GetSize()[0], reference.GetSize()[1]))
        image_centered_itk = sitk.GetImageFromArray(image_centered.T)

    image_centered_itk.SetSpacing([1 for i in range(image.GetDimension())])
    return image_centered_itk

def registration_imzml(register, fixed, best_resampler, to_flip, mz, index):
    global intensities, coordinates, mzs, spectra
    size = fixed.GetSize()
    pixel_type = register.GetPixelID()
    outRegister = apply_registration_imzml(register, best_resampler, to_flip, fixed.GetSize())
    k = outRegister.GetSize()[2]
    outResized = sitk.Image(size[0], size[1], k, pixel_type)
    for i in range(k):
        out2D = sitk.JoinSeries(outRegister[:,:,i])
        outResized = sitk.Paste(outResized, out2D, out2D.GetSize(), destinationIndex=[0,0,i])
    I, coords = imzmlio.get_spectra_from_images(sitk.GetArrayFromImage(outResized).T)
    coords = [(elem[0], elem[1], index+1) for elem in coords]
    max_x = max(coords, key=lambda item:item[0])[0]
    max_y = max(coords, key=lambda item:item[1])[1]
    if (max_x != size[0] or max_y != size[1]):
        I += [[0 for i in range(len(mz))]]
        coords += [(size[0], size[1], index+1)]
    max_x = max(coords, key=lambda item:item[0])[0]
    max_y = max(coords, key=lambda item:item[1])[1]
    intensities += I
    coordinates += coords
    current_mzs = [mz] * len(coords)
    mzs += current_mzs
    current_spectra = np.stack((current_mzs, I), axis=1)
    spectra = spectra + current_spectra.tolist()


def registration_itk(image, fixed, best_resamplers, flips):
    dim_image = image.GetDimension()
    if dim_image == 2:
        outImage = apply_registration(image, best_resamplers[0], flips[0])

    elif dim_image == 3:
        size = fixed.GetSize()
        pixel_type = fixed.GetPixelID()
        outImage = sitk.Image(size[0], size[1], size[2], pixel_type)
        for i in range(size[2]):
            print("Slice ", i)
            diff_resampler = (size[2] == image.GetSize()[2])
            best_resampler, flip = get_resampler(diff_resampler, best_resamplers, flips, i)
            out2D = apply_registration(image[:,:,i], best_resampler, to_flip)
            out2D = sitk.JoinSeries(out2D)
            outImage = sitk.Paste(outImage, out2D, out2D.GetSize(), destinationIndex=[0,0,i])
    pixel_type_reg = image.GetPixelID()
    if pixel_type_reg >= sitk.sitkFloat32:
        outImage = sitk.Cast(outImage, sitk.sitkFloat32)
    return outImage



parser = argparse.ArgumentParser()
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
parser.add_argument("--learning_rate", help="Learning rate", default=1.1)
parser.add_argument("--relaxation_factor", help="Relaxation factor", default=0.9)
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

learning_rate = float(args.learning_rate)
relaxation_factor = float(args.relaxation_factor)
sampling_percentage = float(args.sampling_percentage)
min_step = float(args.min_step)
pattern = args.pattern
level = int(args.level)
step_realign = float(args.step_realign)
apply_mask = args.apply_mask

fixed = sitk.ReadImage(fixedname, sitk.sitkFloat32)
moving = sitk.ReadImage(movingname, sitk.sitkFloat32)



#Resizing
dim_moving = moving.GetDimension()
moving_before_resize = sitk.GetImageFromArray(sitk.GetArrayFromImage(moving))
if is_resize:
    fixed_size = fixed.GetSize()
    moving_size = (fixed_size[0], int(moving.GetSize()[1]*fixed_size[0]/moving.GetSize()[0]))
    if dim_moving == 3:
        moving_size += (fixed_size[2], )
    print(moving_size)
    moving = utils.resize(moving, moving_size)
    moving = sitk.Cast(moving, sitk.sitkFloat32)


fixed.SetSpacing([1 for i in range(fixed.GetDimension())])
moving.SetSpacing([1 for i in range(dim_moving)])

array_moving = sitk.GetArrayFromImage(moving)

width = fixed.GetWidth()
height = fixed.GetHeight()
numberOfBins = int(math.sqrt(height * width / bins))


best_resamplers = []
flips = []
if dim_moving == 2:
    best_resampler, to_flip = register2D(fixed, moving, numberOfBins, is_best_rotation, array_moving, flipped, sampling_percentage, learning_rate, min_step, relaxation_factor)
    print(to_flip)
    out = apply_registration(moving, best_resampler, to_flip)
    # p,r =reg.quality_registration(sitk.Cast(fixed, sitk.sitkUInt8), sitk.Cast(out, sitk.sitkUInt8), 40)
    # print(reg.fmeasure(np.mean(p), np.mean(r)))
    best_resamplers.append(best_resampler)
    flips.append(to_flip)


if dim_moving == 3:
    size = fixed.GetSize()
    pixel_type = fixed.GetPixelID()
    out = sitk.Image(size[0], size[1], size[2], pixel_type)
    for i in range(array_moving.shape[0]):
        print("Slice ", i)
        best_resampler, to_flip = register2D(fixed[:,:,i], moving[:,:,i], numberOfBins, is_best_rotation, array_moving, flipped, sampling_percentage, learning_rate, min_step, relaxation_factor)

        out2D = apply_registration(moving[:,:,i], best_resampler, to_flip)
        out2D = sitk.JoinSeries(out2D)
        out = sitk.Paste(out, out2D, out2D.GetSize(), destinationIndex=[0,0,i])
        best_resamplers.append(best_resampler)
        flips.append(to_flip)

#Display result
if out != None:
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)

    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1//3.+simg2//1.5)

    if dim_moving == 2:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(sitk.GetArrayFromImage(fixed))
        ax[1].imshow(sitk.GetArrayFromImage(moving))
        ax[2].imshow(sitk.GetArrayFromImage(cimg))
    elif dim_moving == 3:
        fig, ax = plt.subplots(1, 1)
        tracker = SliceViewer(ax, sitk.GetArrayFromImage(cimg))
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

#Apply transformation to registration image
if registername:
    register_image_names = [registername]
    if os.path.isdir(registername):
        register_image_names = extract_image_from_directories(registername, pattern, level)

    is_different_resampler = False
    if len(register_image_names) == len(best_resamplers):
        is_different_resampler = True

    print("Different resampler" , is_different_resampler)

    intensities, coordinates, mzs = [], [], []
    spectra = []
    i = 0

    outImages = []
    for register_name in register_image_names:
        best_resampler, flip = get_resampler(is_different_resampler, best_resamplers, flips, i)
        is_imzml = register_name.lower().endswith(".imzml")
        print("flip", flip)
        register, mz = read_image_to_register(register_name, is_imzml, to_flip=False, moving=moving)
        register = realign_image(register, moving_before_resize)

        if is_resize:
            new_size = (moving.GetSize()[0], moving.GetSize()[1]) + ((register.GetSize()[2],) if register.GetDimension() > 2 else ())
            register = utils.resize(register, new_size)

        if apply_mask:
            array_register = sitk.GetArrayFromImage(register)
            if is_imzml:
                mask = array_moving[i, ...]
                array_register[:, mask == 0] = 0
            else:
                array_register[array_moving == 0] = 0
            register = sitk.GetImageFromArray(array_register)

        if is_imzml:
            registration_imzml(register, fixed, best_resampler, flip, mz, i)
        else:
            outImage = registration_itk(register, fixed, best_resamplers, flips)
            outImages.append(outImage)
        i += 1

    if is_imzml:
        if len(register_image_names) > 1:
            spectra = np.array(spectra, dtype=object)
            realigned_spectra = sp.realign_mzs(spectra, mzs, step=step_realign)
            mzs = realigned_spectra[:, 0]
            intensities = realigned_spectra[:, 1]
        imzmlio.write_imzml(mzs, intensities, coordinates, outputname)
    else:
        if len(outImages) > 1:
            size_out_image = fixed.GetSize()
            pixel_type = fixed.GetPixelID()
            outImage = sitk.Image(size[0], size[1], len(outImages), pixel_type)
            for i in range(len(outImages)):
                out2D = outImages[i]
                out2D = sitk.JoinSeries(out2D)
                outImage = sitk.Paste(outImage, out2D, out2D.GetSize(), destinationIndex=[0,0,i])
            sitk.WriteImage(outImage, outputname)
        elif len(outImages) > 0:
            sitk.WriteImage(outImages[0], outputname)
