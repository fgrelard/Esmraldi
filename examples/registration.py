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
import esmraldi.registration as reg
import esmraldi.segmentation as segmentation

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


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Fixed image")
parser.add_argument("-m", "--moving", help="Moving image")
parser.add_argument("-r", "--register", help="Registration image")
parser.add_argument("-o", "--output", help="Output")
parser.add_argument("-b", "--bins", help="number per bins", default=10)
parser.add_argument("-s", "--symmetry", help="best fit with flipped image", action="store_true", default=False)
args = parser.parse_args()

fixedname = args.fixed
movingname = args.moving
outputname = args.output
registername = args.register
bins = int(args.bins)
flipped = args.symmetry

fixed = sitk.ReadImage(fixedname, sitk.sitkFloat32)
moving = sitk.ReadImage(movingname, sitk.sitkFloat32)

#Resizing
dim_moving = moving.GetDimension()
fixed_size = fixed.GetSize()
moving_size = (fixed_size[0], int(moving.GetSize()[1]*fixed_size[0]/moving.GetSize()[0]))
print(moving_size)
moving = segmentation.resize(moving, moving_size)
sx = fixed.GetSpacing()
spacing = tuple([sx[i] for i in range(dim_moving)])
moving.SetSpacing(spacing)


moving = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkUInt8)
moving = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkFloat32)
array_moving = sitk.GetArrayFromImage(moving)

# Flip axis and choose best fit during registration
if dim_moving == 2 and flipped:
    # Construct a 3D image
    # 2 slices = original + flipped
    dim_moving = 3
    flipped_moving = sitk.Flip(moving, (True, False))
    moving_and_flipped = np.zeros((2, array_moving.shape[-2], array_moving.shape[-1]), dtype=np.float32)
    moving_and_flipped[0, ...] = sitk.GetArrayFromImage(moving)
    moving_and_flipped[1, ...] = sitk.GetArrayFromImage(flipped_moving)
    array_moving = moving_and_flipped


width = fixed.GetWidth()
height = fixed.GetHeight()
numberOfBins = int(math.sqrt(height * width / bins))
samplingPercentage = 0.1

to_flip = False

if dim_moving == 2:
    best_resampler = reg.register(fixed, moving, numberOfBins, samplingPercentage)

if dim_moving == 3:
    print(array_moving.shape)
    best_resampler, index = reg.best_fit(fixed, array_moving, numberOfBins, samplingPercentage)
    if flipped and index == 1:
        to_flip = True

simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
try:
    if to_flip:
        print("Flipped!")
        moving = sitk.Flip(moving, (True, False))
        moving = sitk.GetImageFromArray(sitk.GetArrayFromImage(moving))
    out = best_resampler.Execute(moving)
except Exception as e:
    print("Problem with best_resampler")
    print(e)
else:
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1//3.+simg2//1.5)

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(sitk.GetArrayFromImage(moving))
    ax[1].imshow(sitk.GetArrayFromImage(cimg))
    plt.show()

# print("-------")
# print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
# print(" Iteration: {0}".format(R.GetOptimizerIteration()))
# print(" Metric value: {0}".format(R.GetMetricValue()))

if registername:
    is_imzml = registername.lower().endswith(".imzml")
    if is_imzml:
        imzml = imzmlio.open_imzml(registername)
        array = imzmlio.to_image_array(imzml).T
        register = sitk.GetImageFromArray(array)
    else:
        register = sitk.ReadImage(registername, sitk.sitkFloat32)
    if to_flip:
        dim = register.GetDimension()
        flip = (True, False) + ((False,) if dim > 2 else ())
        register = sitk.Flip(register, (True, False, False))
        register = sitk.GetImageFromArray(sitk.GetArrayFromImage(register))

    dim = register.GetDimension()
    new_size = moving_size + ((register.GetSize()[2],) if dim > 2 else ())
    register = segmentation.resize(register, new_size)

    identity = np.identity(dim).tolist()
    flat_list = [item for sublist in identity for item in sublist]
    direction = tuple(flat_list)
    register.SetDirection(flat_list)

    size = register.GetSize()
    pixel_type = register.GetPixelID()

    if len(size) == 2:
        outRegister = sitk.Image(fixed_size[0], fixed_size[1], pixel_type )

    if len(size) == 3:
        outRegister = sitk.Image(fixed_size[0], fixed_size[1], size[2], pixel_type )

    sx = fixed.GetSpacing()
    spacing = tuple([sx[0] for i in range(dim)])
    register.SetSpacing(spacing)

    if len(size) == 2:
        outRegister = best_resampler.Execute(register)
        # if not is_imzml:
        #     outRegister = sitk.Cast(sitk.RescaleIntensity(outRegister), pixel_type)

    if len(size) == 3:
        for i in range(size[2]):
            slice = register[:,:,i]
            slice.SetSpacing(fixed.GetSpacing())

            outSlice = best_resampler.Execute(slice)
            # if not is_imzml:
            #     outSlice = sitk.Cast(sitk.RescaleIntensity(outSlice), pixel_type)

            outSlice = sitk.JoinSeries(outSlice)
            outRegister = sitk.Paste(outRegister, outSlice, outSlice.GetSize(), destinationIndex=[0,0,i])

    if is_imzml:
        mz, y = imzml.getspectrum(0)
        intensities, coordinates = imzmlio.get_spectra_from_images(sitk.GetArrayFromImage(outRegister).T)
        mzs = [mz] * len(coordinates)
        imzmlio.write_imzml(mzs, intensities, coordinates, outputname)
    else:
        outRegister = sitk.Cast(outRegister, sitk.sitkUInt8)
        sitk.WriteImage(outRegister, outputname)
