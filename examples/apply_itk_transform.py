"""
Example file to apply a
non-rigid ITK transform
based on a displacement field
"""

import argparse
import SimpleITK as sitk
import esmraldi.imzmlio as imzmlio
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input image (image before deformation, .imzML or ITK format)")
parser.add_argument("-t", "--transform", help="Input transform from ITK (.mha)")
parser.add_argument("-o", "--output", help="Output image after deformation (.imzML or ITK format)")
args = parser.parse_args()

inputname = args.input
transformname = args.transform
outputname = args.output

transform = sitk.ReadImage(transformname)
t64 = sitk.Cast(transform, sitk.sitkVectorFloat64)
field = sitk.DisplacementFieldTransform(t64)
array = sitk.GetArrayFromImage(transform)


is_imzml = inputname.lower().endswith(".imzml")
if is_imzml:
    imzml = imzmlio.open_imzml(inputname)
    spectra = imzmlio.get_full_spectra(imzml)
    max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
    max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
    max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
    image = imzmlio.get_images_from_spectra(spectra, (max_x, max_y, max_z))
    print(image.shape)
    image = sitk.GetImageFromArray(image.T)
else:
    image = sitk.ReadImage(inputname)

dim = image.GetDimension()
identity = np.identity(dim).tolist()
flat_list = [item for sublist in identity for item in sublist]
direction = tuple(flat_list)
image.SetDirection(flat_list)
size = image.GetSize()
shape = sitk.GetArrayFromImage(image).T.shape
image.SetSpacing(transform.GetSpacing())
print("Image shape=", shape[::-1], "Field shape=", array.shape)
print("Image spacing=", image.GetSpacing(), " transform spacing", transform.GetSpacing())

if not is_imzml:
    outRegister = sitk.Resample(image, field, sitk.sitkNearestNeighbor, 0)
    outRegister_array = sitk.GetArrayFromImage(outRegister).T
elif len(shape) == 3:
    print("3D registration (imzML)")
    pixel_type = image.GetPixelID()
    outRegister = sitk.Image(size[0], size[1], size[2], pixel_type )
    for i in range(size[2]):
        slice = image[:,:,i]
        outSlice = sitk.Resample(slice, field, sitk.sitkNearestNeighbor, 0)
        outSlice = sitk.JoinSeries(outSlice)
        outRegister = sitk.Paste(outRegister, outSlice, outSlice.GetSize(), destinationIndex=[0, 0, i])
    outRegister_array = sitk.GetArrayFromImage(outRegister).T
elif len(shape) == 4:
    print("4D registration (imzML)")
    pixel_type = image.GetPixelID()
    array_image = sitk.GetArrayFromImage(image).T
    outRegister_array = np.zeros_like(array_image)
    for i in range(shape[-1]):
        image_3D = array_image[..., i]
        image_3D_itk = sitk.GetImageFromArray(image_3D.T)
        image_3D_itk.SetSpacing(transform.GetSpacing())
        if i == 0:
            print(image_3D_itk.GetSize())
        register_image_3D = sitk.Resample(image_3D_itk, field, sitk.sitkNearestNeighbor, 0)
        outRegister_array[..., i] = sitk.GetArrayFromImage(register_image_3D).T


fig, ax = plt.subplots(1, 2)

before = sitk.GetArrayFromImage(image).T
index = np.unravel_index(np.argmax(np.mean(before, axis=-1), axis=None), before.shape)
print(index)
after = outRegister_array
print(outRegister_array.shape)
if len(shape) == 3:
    before = before[:,:, index[-1]]
    after = after[:,:, index[-1]]
elif len(shape) == 4:
    before = before[:,:, index[-2], index[-1]]
    after = after[:,:, index[-2], index[-1]]
ax[0].imshow(before.T)
ax[1].imshow(after.T)
plt.show()

if is_imzml:
    mz, y = imzml.getspectrum(0)
    intensities, coordinates = imzmlio.get_spectra_from_images(outRegister_array)
    mzs = [mz] * len(coordinates)
    imzmlio.write_imzml(mzs, intensities, coordinates, outputname)
else:
    outRegister = sitk.Cast(outRegister, sitk.sitkFloat32)
    sitk.WriteImage(outRegister, outputname)
