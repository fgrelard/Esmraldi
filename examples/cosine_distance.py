import numpy as np
import matplotlib.pyplot as plt
import esmraldi.imzmlio as imzmlio
import esmraldi.segmentation as seg
import esmraldi.fusion as fusion
import SimpleITK as sitk
import scipy.spatial.distance as distance
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
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
    mzs, intensities = imzml.getspectrum(0)
else:
    image = sitk.GetArrayFromImage(sitk.ReadImage(inputname)).T
    mzs = [i for i in range(image.shape[2])]
    mzs = np.asarray(mzs)

image = image[..., mzs >= threshold]

mzs = mzs[mzs >= threshold]
mzs = np.around(mzs, decimals=2)
mzs = mzs.astype(str)

image_mri = sitk.GetArrayFromImage(sitk.ReadImage(mriname, sitk.sitkFloat32)).T

if len(image.shape) == 3:
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image[..., 0])
    ax[1].imshow(image_mri)
    plt.show()

elif len(image.shape) == 4:
    fig, ax = plt.subplots(1, 2)
    display_maldi = np.transpose(image[..., 0], (2, 1, 0))
    display_mri = np.transpose(image_mri, (2, 1, 0))
    tracker = SliceViewer(ax, display_maldi, display_mri)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


image = imzmlio.normalize(image)
image_flatten = fusion.flatten(image, is_spectral=True)
image_mri = imzmlio.normalize(image_mri)
image_mri_flatten = fusion.flatten(image_mri)

if is_ratio:
    ratio_images, ratio_mzs = fusion.extract_ratio_images(image, mzs)
    image = np.concatenate((image, ratio_images), axis=2)
    mzs = np.concatenate((mzs, ratio_mzs))


indices_closest = fusion.get_closest_indices(image_flatten[0], image_mri_flatten[0])
values = np.array([255-int((i*255)/len(indices_closest)) for i in range(len(indices_closest))])
maldi_closest = image_flatten[0]
maldi_closest[indices_closest] = values
maldi_closest = np.reshape(maldi_closest, image.shape[:-1])

print(image_mri_flatten.shape, image_flatten.shape[:1])
cosines = cosine_similarity(image_flatten,image_mri_flatten)


indices = [i for i in range(len(cosines))]
indices.sort(key=lambda x:cosines[x][0], reverse=True)


indices_array = np.array(indices)

similar_images = np.take(image, indices, axis=-1)
similar_mzs = np.take(mzs, indices)



np.savetxt(outname, similar_mzs, delimiter=";", fmt="%s")
print(similar_mzs)

if len(similar_images.shape) == 3:
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(similar_images[..., 0])
    ax[1].imshow(image_mri)
    ax[2].imshow(maldi_closest)
    plt.show()
elif len(similar_images.shape) == 4:
    fig, ax = plt.subplots(1, 3)
    tracker = SliceViewer(ax, np.transpose(similar_images[..., 0], (2, 1, 0)),  np.transpose(image_mri, (2, 1, 0)), np.transpose(maldi_closest, (2, 1, 0)), vmin=0, vmax=255)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
