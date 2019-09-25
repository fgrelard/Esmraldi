import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import numpy as np
import src.imzmlio as imzmlio
import os

import src.registration as reg

def command_iteration(method) :
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))



def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def register(fixed, moving, numberOfBins):
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

def registration_number_bins(outputname, min_bins, max_bins):
    for i in range(min_bins, max_bins):
        outname = os.path.splitext(outputname)[0] + "_bins" + str(i) +".png"
        try:
            resampler = register(fixed, moving, i)
            out = resampler.Execute(moving)
            out = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
            sitk.WriteImage(out, outname)
        except Exception as e:
            pass


def resize(image, size):
    dim = len(image.GetSize())
    new_dims = [size for i in range(2)]
    spacing = [image.GetSize()[0]/size for i in range(2)]
    if dim == 3:
        new_dims.append(image.GetSize()[2])
        spacing.append(1)
    print(new_dims)
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




parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Fixed image")
parser.add_argument("-m", "--moving", help="Moving image")
parser.add_argument("-r", "--register", help="Registration image")
parser.add_argument("-o", "--output", help="Output")
parser.add_argument("-b", "--bins", help="number per bins", default=10)

args = parser.parse_args()

fixedname = args.fixed
movingname = args.moving
outputname = args.output
registername = args.register
bins = int(args.bins)

fixed = sitk.ReadImage(fixedname, sitk.sitkFloat32)
moving = sitk.ReadImage(movingname, sitk.sitkFloat32)

#Resizing
moving = resize(moving, fixed.GetSize()[0])

moving = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkUInt8)
array_moving = sitk.GetArrayFromImage(moving)

#Removing tube
minr = 15
maxr = 30

dim_moving = len(moving.GetSize())
if dim_moving == 2:
    center_x, center_y, radius = reg.detect_circle(array_moving, threshold=150, min_radius=minr, max_radius=maxr)
    array_moving = reg.fill_circle(center_x, center_y, maxr, array_moving)

if dim_moving == 3:
    center_x, center_y, radius = reg.detect_tube(array_moving, min_radius=minr, max_radius=maxr)
    array_moving = reg.fill_circle(center_x, center_y, maxr, array_moving)

moving = sitk.GetImageFromArray(array_moving)
moving = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkFloat32)

plt.imshow(array_moving)
plt.show()

width = fixed.GetWidth()
height = fixed.GetHeight()
numberOfBins = int(math.sqrt(height * width / bins))
samplingPercentage = 0.1

if dim_moving == 2:
    best_resampler = register(fixed, moving, numberOfBins)

if dim_moving == 3:
    best_resampler = best_fit(fixed, array_moving)


simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
try:
    out = best_resampler.Execute(moving)
except Exception as e:
    print(e)
else:
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1//3.+simg2//1.5)

plt.imshow(sitk.GetArrayFromImage(cimg))
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
    dim = register.GetDimension()
    identity = np.identity(dim).tolist()
    flat_list = [item for sublist in identity for item in sublist]
    direction = tuple(flat_list)
    register.SetDirection(flat_list)

    size = register.GetSize()
    pixel_type = register.GetPixelID()

    if len(size) == 2:
        outRegister = sitk.Image(width, height, pixel_type )

    if len(size) == 3:
        outRegister = sitk.Image(width, height, size[2], pixel_type )
    sx = fixed.GetSpacing()[0]
    spacing = tuple([sx for i in range(dim)])
    outRegister.SetSpacing(spacing)

    if len(size) == 2:
        outRegister = best_resampler.Execute(register)
        if not is_imzml:
            outRegister = sitk.Cast(sitk.RescaleIntensity(outRegister), pixel_type)

    if len(size) == 3:
        for i in range(size[2]):
            slice = register[:,:,i]
            slice.SetSpacing(fixed.GetSpacing())
            outSlice = best_resampler.Execute(slice)
            if not is_imzml:
                outSlice = sitk.Cast(sitk.RescaleIntensity(outSlice), pixel_type)
            outSlice = sitk.JoinSeries(outSlice)
            outRegister = sitk.Paste(outRegister, outSlice, outSlice.GetSize(), destinationIndex=[0,0,i])

    if is_imzml:
        mz, y = imzml.getspectrum(0)
        intensities, coordinates = imzmlio.get_spectra_from_images(sitk.GetArrayFromImage(outRegister).T)
        mzs = [mz] * len(coordinates)
        imzmlio.write_imzml(mzs, intensities, coordinates, outputname)

    else:
        sitk.WriteImage(outRegister, outputname)
