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

if registername:
    register = sitk.ReadImage(registername, sitk.sitkFloat32)

gradient_filter = sitk.GradientMagnitudeImageFilter()
gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
gaussian.SetSigma ( 1.0 )
moving = gaussian.Execute(moving)

width = fixed.GetWidth()
height = fixed.GetHeight()
numberOfBins = int(math.sqrt(height * width / bins))
print(numberOfBins)
samplingPercentage = 0.1
moving.SetSpacing(fixed.GetSpacing())




R = sitk.ImageRegistrationMethod()
R.SetMetricAsMattesMutualInformation(numberOfBins)
R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
#R.SetOptimizerAsGradientDescent(1.0, 2000)
R.SetOptimizerAsRegularStepGradientDescent(1.0,.001,2000)
tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
R.SetInitialTransform(tx)

outTx = R.Execute(fixed, moving)
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(1)
resampler.SetTransform(outTx)
outTranslation = resampler.Execute(moving)



print("-------")
print(outTx)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed);
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0)
resampler.SetTransform(outTx)
out = resampler.Execute(moving)

register.SetDirection( (1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0))
size = register.GetSize()
outRegister = sitk.Image(width, height, size[2], sitk.sitkUInt8 )
sx = fixed.GetSpacing()[0]
outRegister.SetSpacing((sx, sx, sx))
print(outRegister.GetDirection())

for i in range(size[2]):
    slice = register[:,:,i]
    slice.SetSpacing(fixed.GetSpacing())
    outSlice = resampler.Execute(slice)
    outSlice = outSlice[::,::-1]
    outSlice = sitk.Cast(sitk.RescaleIntensity(outSlice), sitk.sitkUInt8)
    outSlice = sitk.JoinSeries(outSlice)
    outRegister = sitk.Paste(outRegister, outSlice, outSlice.GetSize(), destinationIndex=[0,0,i])

sitk.WriteImage(outRegister, outputname)

simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)


plt.imshow(sitk.GetArrayFromImage(cimg))
plt.show()
