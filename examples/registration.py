import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import numpy as np
import src.imzmlio as imzmlio
import os
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage import data, color
from skimage.draw import circle


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


def detect_circles(image, threshold, min_radius, max_radius):
    cond = np.where(image < threshold)
    image_copy = np.copy(image)
    image_copy[cond] = 0
    edges = canny(image_copy, sigma=3, low_threshold=10, high_threshold=40)

    # Detect two radii
    hough_radii = np.arange(min_radius, max_radius, 10)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)
    if len(cx) > 0:
        return cx[0], cy[0], radii[0]
    return -1, -1, -1



def detect_tube(image, threshold=150, min_radius=10, max_radius=50):
    cy, cx, radii = [], [], []
    for i in range(image.shape[0]):
        center_x, center_y, radius = detect_circles(image[i, :,:], threshold, min_radius, max_radius)
        if center_y >= 0:
            cy.append(center_y)
            cx.append(center_x)
            radii.append(radius)
    center_y = np.median(cy)
    center_x = np.median(cx)
    radius = np.median(radii)
    return center_x, center_y, radius

def fill_circle(center_x, center_y, radius, image, color=0):
    image2 = np.copy(image)
    rr, cc = circle(int(center_y), int(center_x), int(radius), image2.shape[1:])
    image2[:, rr,cc] = 0
    return image2



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

moving = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkUInt8)
moving = resize(moving, fixed.GetSize()[0])
array_moving = sitk.GetArrayFromImage(moving)
center_x, center_y, radius = detect_tube(array_moving, min_radius=30, max_radius=50)
array_moving = fill_circle(center_x, center_y, radius, array_moving)

moving = sitk.GetImageFromArray(array_moving[4, ...])
moving = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkFloat32)

plt.imshow(array_moving[4, ...])
plt.show()

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
        outRegister = resampler.Execute(register)
        if not is_imzml:
            outRegister = sitk.Cast(sitk.RescaleIntensity(outRegister), pixel_type)

    if len(size) == 3:
        for i in range(size[2]):
            slice = register[:,:,i]
            slice.SetSpacing(fixed.GetSpacing())
            outSlice = resampler.Execute(slice)
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
