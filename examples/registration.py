import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import numpy as np
import src.imzmlio as imzmlio
import os

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
    # R.SetOptimizerAsRegularStepGradientDescent(1.0,.001,2000)
    R.SetOptimizerAsOnePlusOneEvolutionary(5000)
    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(tx)

    try:
        outTx = R.Execute(fixed, moving)
    except Exception as e:
        pass
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


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Fixed image")
parser.add_argument("-m", "--moving", help="Moving image")
parser.add_argument("-r", "--register", help="Registration image")
parser.add_argument("-o", "--output", help="Output")
parser.add_argument("-b", "--bins", help="number of bins", default=10)

args = parser.parse_args()

fixedname = args.fixed
movingname = args.moving
outputname = args.output
registername = args.register
bins = int(args.bins)

fixed = sitk.ReadImage(fixedname, sitk.sitkFloat32)
moving = sitk.ReadImage(movingname, sitk.sitkFloat32)



gradient_filter = sitk.GradientMagnitudeImageFilter()
gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
gaussian.SetSigma(1.0)
#moving = gaussian.Execute(moving)

width = fixed.GetWidth()
height = fixed.GetHeight()
numberOfBins = int(math.sqrt(height * width / bins))
samplingPercentage = 0.1
moving.SetSpacing(fixed.GetSpacing())


numberOfBins = int(math.sqrt(height * width / bins))

resampler = register(fixed, moving, numberOfBins)
out = resampler.Execute(moving)
simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1//3.+simg2//1.5)


plt.imshow(sitk.GetArrayFromImage(cimg))
plt.show()


print("-------")
print(outTx)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))

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


    if len(size) == 2:
        outRegister = sitk.Image(width, height, sitk.sitkUInt8 )

    if len(size) == 3:
        outRegister = sitk.Image(width, height, size[2], sitk.sitkUInt8 )
    sx = fixed.GetSpacing()[0]
    spacing = tuple([sx for i in range(dim)])
    outRegister.SetSpacing(spacing)

    if len(size) == 2:
        outRegister = resampler.Execute(register)
        outRegister = sitk.Cast(sitk.RescaleIntensity(outRegister), sitk.sitkUInt8)

    if len(size) == 3:
        for i in range(size[2]):
            slice = register[:,:,i]
            slice.SetSpacing(fixed.GetSpacing())
            outSlice = resampler.Execute(slice)
            outSlice = sitk.Cast(sitk.RescaleIntensity(outSlice), sitk.sitkUInt8)
            outSlice = sitk.JoinSeries(outSlice)
            outRegister = sitk.Paste(outRegister, outSlice, outSlice.GetSize(), destinationIndex=[0,0,i])

    if is_imzml:
        mz, y = imzml.getspectrum(0)
        intensities, coordinates = imzmlio.get_spectra_from_images(sitk.GetArrayFromImage(outRegister).T)
        mzs = [mz] * len(coordinates)
        imzmlio.write_imzml(mzs, intensities, coordinates, outputname)

    else:
        sitk.WriteImage(outRegister, outputname)
