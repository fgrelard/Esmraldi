import numpy as np
import matplotlib.pyplot as plt
import esmraldi.imzmlio as imzmlio
import esmraldi.segmentation as seg
import esmraldi.fusion as fusion
import SimpleITK as sitk
import scipy.spatial.distance as distance
import argparse

from esmraldi.sliceviewer import SliceViewer

"""
Computes the cosine distance between each MALDI ion image and the MRI image
And sorts the MALDI according to this distance in descending order
"""
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image (imzML or nii)")
parser.add_argument("-m", "--mri", help="Input MRI image (ITK format)")
parser.add_argument("-o", "--output", help="Output image (ITK format)")
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold", default=0)
parser.add_argument("-r", "--ratio", help="Compute ratio images", action="store_true")


args = parser.parse_args()

inputname = args.input
mriname = args.mri
outname = args.output
threshold = int(args.threshold)
is_ratio = args.ratio


if inputname.lower().endswith(".imzml"):
    imzml = imzmlio.open_imzml(inputname)
    spectra = imzmlio.get_full_spectra(imzml)
    max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
    max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
    max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
    image = imzmlio.get_images_from_spectra(spectra, (max_x, max_y, max_z))
    # image = imzmlio.to_image_array(imzml)
    mzs, intensities = imzml.getspectrum(0)
else:
    image = sitk.GetArrayFromImage(sitk.ReadImage(inputname)).T
    mzs = [i for i in range(image.shape[2])]
    mzs = np.asarray(mzs)

print("Mass-to-charge ratio=", mzs)

image = image[..., mzs >= threshold]
image = imzmlio.normalize(image)

mzs = mzs[mzs >= threshold]
mzs = np.around(mzs, decimals=2)
mzs = mzs.astype(str)

image_mri = sitk.GetArrayFromImage(sitk.ReadImage(mriname, sitk.sitkFloat32)).T

if len(image.shape) == 3:
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image[..., 0])
    ax[0].imshow(image_mri)
    plt.show()
elif len(image.shape) == 4:
    fig, ax = plt.subplots(1, 1)
    tracker = SliceViewer(ax, np.transpose(image[..., 0], (2, 1, 0)))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    tracker = SliceViewer(ax, np.transpose(image_mri, (2, 1, 0)))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


if is_ratio:
    ratio_images, ratio_mzs = fusion.extract_ratio_images(image, mzs)
    image = np.concatenate((image, ratio_images), axis=2)
    mzs = np.concatenate((mzs, ratio_mzs))


image_flatten = fusion.flatten(image, is_spectral=True).T
image_mri = imzmlio.normalize(image_mri)
image_mri_flatten = fusion.flatten(image_mri).T

print(image_mri_flatten.shape, image_flatten.shape)
distances = []
for i in range(image_flatten.shape[-1]):
    maldi = image_flatten[..., i]
    d = distance.cosine(maldi, image_mri_flatten[..., 0])
    distances.append(d)

indices = [i for i in range(len(distances))]
indices.sort(key=lambda x:distances[x], reverse=False)

indices_array = np.array(indices)
# print(distances)
# print(indices_array)

similar_images = np.take(image, indices, axis=-1)
similar_mzs = np.take(mzs, indices)

np.savetxt(outname, similar_mzs, delimiter=";", fmt="%s")
print(similar_mzs)

if len(similar_images.shape) == 3:
    plt.imshow(similar_images[..., 0])
    plt.show()
elif len(similar_images.shape) == 4:
    fig, ax = plt.subplots(1, 1)
    tracker = SliceViewer(ax, np.transpose(similar_images[..., 0], (2, 1, 0)))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
