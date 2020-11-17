import numpy as np
import matplotlib.pyplot as plt
import esmraldi.segmentation as seg
import esmraldi.imzmlio as imzmlio
import esmraldi.fusion as fusion
import argparse
import scipy.ndimage as ndimage
import SimpleITK as sitk

from esmraldi.fastmap import FastMap
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

def mapping_neighbors(image, radius, weights):
    r = radius
    size = 2*r+1
    img_padded = np.pad(image, (r,r), 'constant')
    mapping_matrix = np.zeros(shape=(image.shape[0], image.shape[1], size, size, image.shape[-1]))
    for index in np.ndindex(image.shape[:-1]):
        i, j = index
        neighbors = image[i-r:i+r+1, j-r:j+r+1]
        if neighbors.shape[0] != size or neighbors.shape[1] != size:
            continue
        mapping_matrix[index] = neighbors * weights[..., None]
    return mapping_matrix

def gaussian_weights(radius):
    size = 2*radius+1
    sigma = size/4
    return np.array([[np.exp((-i**2-j**2)/(2*sigma**2)) for i in range(-radius,radius+1)] for j in range(-radius,radius+1)])

def spatially_aware_clustering(image, k, n, radius):
    weights = gaussian_weights(radius)
    mapping_matrix = mapping_neighbors(image, radius, weights)
    old_shape = mapping_matrix.shape
    new_shape = (np.prod(old_shape[:-3]), np.prod(old_shape[-3:]))
    fastmap_matrix = mapping_matrix.reshape(new_shape)

    if n < new_shape[-1]:
        fastmap = FastMap(fastmap_matrix, n)
        proj = fastmap.compute_projections()
        pd_X = pairwise_distances(fastmap_matrix)**2
        pd_proj = pairwise_distances(proj)**2
        print("Sum abs. diff=", np.sum(np.abs(pd_X - pd_proj)))
    else:
        proj = fastmap_matrix

    kmeans = KMeans(k, random_state=0).fit(proj)
    labels = kmeans.labels_
    image_labels = labels.reshape(old_shape[:-3])
    plt.imshow(image_labels)
    plt.show()
    return image_labels



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image (imzML or nii)")
parser.add_argument("-o", "--output", help="Output image (ITK format)")
parser.add_argument("-n", "--number", help="Number of dimensions after dimension reduction (fastmap)", default=50)
parser.add_argument("-k", "--classes", help="Number of clusters for kmeans", default=7)
parser.add_argument("-r", "--radius", help="Radius for spatial features", default=1)
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold (optional)", default=0)

args = parser.parse_args()

inputname = args.input
outname = args.output
radius = int(args.radius)
n = int(args.number)
k = int(args.classes)
threshold = int(args.threshold)


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
print("Number of peaks=",image.shape[-1])
flatten_image = fusion.flatten(image, is_spectral=True)

image_labels = spatially_aware_clustering(image, n, k, radius)
image_labels_itk = sitk.GetImageFromArray(image_labels.astype(np.uint8))
sitk.WriteImage(image_labels_itk, outname)
