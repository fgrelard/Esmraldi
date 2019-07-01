import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import numpy as np
import src.imzmlio as imzmlio

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

def precision(im1, im2):
    tp = np.count_nonzero((im2 + im1) == 2)
    allp = np.count_nonzero(im2 == 1)
    return tp * 1.0 / allp
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(sitk.GetArrayFromImage(tp))
    # ax[1].imshow(sitk.GetArrayFromImage(im1))
    # ax[2].imshow(sitk.GetArrayFromImage(im2))
    # plt.show()
    # print(tp)

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

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Fixed image")
parser.add_argument("-m", "--moving", help="Moving image")
parser.add_argument("-r", "--register", help="Registration image")
parser.add_argument("-o", "--output", help="Output")
parser.add_argument("-b", "--bins", help="number of bins", default=5)

args = parser.parse_args()

fixedname = args.fixed
movingname = args.moving
outputname = args.output
registername = args.register
bins = int(args.bins)

fixed =  sitk.ReadImage(fixedname, sitk.sitkFloat32)
moving = sitk.ReadImage(movingname, sitk.sitkFloat32)



gradient_filter = sitk.GradientMagnitudeImageFilter()
gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
gaussian.SetSigma ( 1.0 )
#moving = gaussian.Execute(moving)

width = fixed.GetWidth()
height = fixed.GetHeight()
numberOfBins = int(math.sqrt(height * width / bins))
samplingPercentage = 0.1
moving.SetSpacing(fixed.GetSpacing())


numberOfBins = int(math.sqrt(height * width / bins))
R = sitk.ImageRegistrationMethod()
R.SetMetricAsMattesMutualInformation(numberOfBins)
R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
# R.SetOptimizerAsRegularStepGradientDescent(1.0,.001,2000)
R.SetOptimizerAsOnePlusOneEvolutionary(2000)
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
    out = resampler.Execute(moving)


    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1//3.+simg2//1.5)

    p, r = quality_registration(simg1, simg2)


plt.imshow(sitk.GetArrayFromImage(cimg))
plt.show()


print("-------")
print(outTx)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))

if registername:
    register = sitk.ReadImage(registername, sitk.sitkFloat32)
    register.SetDirection( (1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0))
    size = register.GetSize()
    outRegister = sitk.Image(width, height, size[2], sitk.sitkUInt8 )
    sx = fixed.GetSpacing()[0]
    outRegister.SetSpacing((sx, sx, sx))

    for i in range(size[2]):
        slice = register[:,:,i]
        slice.SetSpacing(fixed.GetSpacing())
        outSlice = resampler.Execute(slice)
        outSlice = sitk.Cast(sitk.RescaleIntensity(outSlice), sitk.sitkUInt8)
        outSlice = sitk.JoinSeries(outSlice)
        outRegister = sitk.Paste(outRegister, outSlice, outSlice.GetSize(), destinationIndex=[0,0,i])

    sitk.WriteImage(outRegister, outputname)
