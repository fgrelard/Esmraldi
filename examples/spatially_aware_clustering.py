import numpy as np
import matplotlib.pyplot as plt
import esmraldi.segmentation as seg
import esmraldi.imzmlio as imzmlio
import esmraldi.fusion as fusion
import argparse
import scipy.ndimage as ndimage

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

def spatially_aware_clustering(image, k, radius):
    weights = gaussian_weights(radius)
    mapping_matrix = mapping_neighbors(image, radius, weights)
    print(mapping_matrix.shape)
    distance_spectra(mapping_matrix[50,50], mapping_matrix[50,51])

def distance_spectra(s1, s2):
    D = (s1 - s2)**2
    distance = np.sum(D)
    return distance




parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image (imzML or nii)")
parser.add_argument("-o", "--output", help="Output image (ITK format)")
parser.add_argument("-n", "--number", help="Number of components for dimension reduction", default=5)
parser.add_argument("-r", "--radius", help="Radius for spatial features", default=1)
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold (optional)", default=0)

args = parser.parse_args()

inputname = args.input
outname = args.output
radius = int(args.radius)
n = int(args.number)
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

print("Mass-to-charge ratio=", mzs)

image = image[..., mzs >= threshold]

mzs = mzs[mzs >= threshold]
mzs = np.around(mzs, decimals=2)
mzs = mzs.astype(str)

flatten_image = fusion.flatten(image, is_spectral=True)

spatially_aware_clustering(image, n, radius)
