import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt


def command_iteration(method) :
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Fixed image")
parser.add_argument("-m", "--moving", help="Moving image")
parser.add_argument("-o", "--output", help="Output")
args = parser.parse_args()

fixedname = args.fixed
movingname = args.moving
outputname = args.output

fixed =  sitk.ReadImage(fixedname, sitk.sitkFloat32)
moving = sitk.ReadImage(movingname, sitk.sitkFloat32)

gradient_filter = sitk.GradientMagnitudeImageFilter()
# fixed = gradient_filter.Execute(fixed)
# moving = gradient_filter.Execute(moving)


numberOfBins = 24
samplingPercentage = 0.6
moving.SetSpacing(fixed.GetSpacing())

R = sitk.ImageRegistrationMethod()
R.SetMetricAsMattesMutualInformation()
R.SetMetricAsMattesMutualInformation(numberOfBins)
R.SetMetricSamplingPercentage(samplingPercentage,sitk.sitkWallClock)
#R.SetMetricSamplingStrategy(R.RANDOM)
R.SetOptimizerAsRegularStepGradientDescent(1.0,.0001,2000)
tx = sitk.TranslationTransform(fixed.GetDimension())

#R.SetMovingInitialTransform(tx)
R.SetInitialTransform(tx)
R.SetInterpolator(sitk.sitkLinear)

R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

outTx = R.Execute(fixed, moving)
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed);
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(1)
resampler.SetTransform(outTx)
outTranslation = resampler.Execute(moving)


R2 = sitk.ImageRegistrationMethod()
#R2.SetMetricAsMattesMutualInformation()
R2.SetMetricAsMeanSquares()
R2.SetMetricSamplingPercentage(samplingPercentage,sitk.sitkWallClock)
R2.SetOptimizerAsRegularStepGradientDescent(2.0,.001,2000)
tx2 = sitk.AffineTransform(fixed.GetDimension())
tx2 = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
R2.SetInitialTransform(tx2)



outTx2 = R2.Execute(fixed, outTranslation)

compositeTransform = sitk.Transform(2, sitk.sitkComposite)
compositeTransform.AddTransform(outTx)
compositeTransform.AddTransform(outTx2)

print(fixed.GetSpacing())
print(moving.GetSpacing())
print(outTranslation.GetSpacing())
print("-------")
print(outTx)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed);
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(1)
resampler.SetTransform(tx2)
out = resampler.Execute(moving)


simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
#sitk.Show( cimg, "ImageRegistration4 Composition" )


plt.imshow(sitk.GetArrayFromImage(cimg))
plt.show()
