import argparse
import SimpleITK as sitk
import src.imzmlio as imzmlio
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input image")
parser.add_argument("-t", "--transform", help="Input transform from ITK (.mha)")
parser.add_argument("-o", "--output", help="Output image")
args = parser.parse_args()

inputname = args.input
transformname = args.transform
outputname = args.output

transform = sitk.ReadImage(transformname)
t64 = sitk.Cast(transform, sitk.sitkVectorFloat64)
field = sitk.DisplacementFieldTransform(t64)

array = sitk.GetArrayFromImage(transform)
print(array.shape)

is_imzml = inputname.lower().endswith(".imzml")
if is_imzml:
    imzml = imzmlio.open_imzml(inputname)
    image = imzmlio.to_image_array(imzml)
    image = sitk.GetImageFromArray(image.T)
else:
    image = sitk.ReadImage(inputname)


size = image.GetSize()
if len(size) == 2:
    outRegister = sitk.Resample(image, field, sitk.sitkNearestNeighbor, 0)
elif len(size) == 3:
    pixel_type = image.GetPixelID()
    outRegister = sitk.Image(size[0], size[1], size[2], pixel_type )
    for i in range(size[2]):
        slice = image[:,:,i]
        outSlice = sitk.Resample(slice, field, sitk.sitkNearestNeighbor, 0)
        outSlice = sitk.JoinSeries(outSlice)
        outRegister = sitk.Paste(outRegister, outSlice, outSlice.GetSize(), destinationIndex=[0, 0, i])


if is_imzml:
    mz, y = imzml.getspectrum(0)
    ########DO SOMETHING
    intensities, coordinates = imzmlio.get_spectra_from_images(sitk.GetArrayFromImage(outRegister).T)
    mzs = [mz] * len(coordinates)
    imzmlio.write_imzml(mzs, intensities, coordinates, outputname)
else:
    sitk.WriteImage(outRegister, outputname)
