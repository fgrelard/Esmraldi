import esmraldi.segmentation as seg
import esmraldi.registration as reg
import esmraldi.fusion as fusion
import esmraldi.imzmlio as io
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.signal as signal
import SimpleITK as sitk

from scipy.stats.stats import spearmanr
from skimage.filters import threshold_otsu, rank
from skimage import measure
from skimage.morphology import binary_erosion, opening, disk


def segmentation_msi(images):
    ## Find spatially coherent images in the image
    ## Images are binarized using the set of thresholds in quantiles
    ## Images that are spatially coherent are defined as images where the area
    ## of the largest connected component is greater than
    ## percentage*max_{im \in images} area(im)
    relevant_set = seg.find_similar_images_spatial_coherence_percentage(images, percentage=0.95, quantiles=[60, 70, 80, 90])

    ## Region growing
    ## First determine list of seeds from average image
    mean_image = np.uint8(cv.normalize(np.average(relevant_set, axis=2), None, 0, 255, cv.NORM_MINMAX))
    threshold = 50
    otsu = threshold_otsu(mean_image)
    labels = measure.label(mean_image > otsu, background=0)
    regionprop = seg.properties_largest_area_cc(labels)
    largest_cc = seg.region_property_to_cc(labels, regionprop)
    seeds = set(((int(coord[0]), int(coord[1])) for coord in regionprop.coords))

    ## Then perform region growing with these seeds
    list_end, evolution_segmentation = seg.region_growing(relevant_set, seeds, threshold)

    ## Extract mask from segmentation, and apply mathematical morphology
    ## opening with se_radius=2 to fill holes
    x = [elem[0] for elem in list_end]
    y = [elem[1] for elem in list_end]
    mask = np.ones_like(mean_image)
    mask[x, y] = 0
    selem = disk(2)
    mask = opening(mask, selem)
    masked_mean_image = np.ma.array(mean_image, mask=mask)
    masked_mean_image = masked_mean_image.filled(0)
    return masked_mean_image.T

def evaluation_segmentation(curvature_msi, curvature_mri):
    def h_distance(mri, maldi):
        t = np.array(mri)
        current_values = np.array(maldi)
        closest = current_values[(np.abs(t[:, None] - current_values).argmin(axis=1))]
        diff = np.mean(np.abs(t - closest))
        return diff

    def find_peaks(data, prominence, w):
        peaks, _ = signal.find_peaks(tuple(data),
                                     height=prominence,
                                     wlen=w,
                                     distance=1)
        return peaks

    def best_hdistance(mri, maldi, length):
        t = np.array(mri)
        min_diff = 2**32
        for delta in range(length):
            current_values = np.array([(length+peak-delta)%length for peak in maldi])
            diff = h_distance(t, current_values)
            if diff < min_diff:
                best_delta = delta
                min_diff = diff
        return min_diff, best_delta

    def fhmeasure_aligned(image_curvature, mri, threshold=0.2, sigma=2):
        image_curvature = scipy.ndimage.gaussian_filter1d(np.copy(image_curvature), sigma)
        indices_image = (find_peaks(image_curvature, threshold, 50)).tolist()
        mri_curvature = scipy.ndimage.zoom(mri, len(image_curvature)/len(mri), order=3)
        mri_curvature = scipy.ndimage.gaussian_filter1d(np.copy(mri_curvature), sigma)
        indices_mri = (find_peaks(mri_curvature, threshold, 50)).tolist()
        h_d, trans = best_hdistance(indices_mri, indices_image, len(image_curvature))
        image_curvature = np.roll(image_curvature, -trans)
        indices_image = (find_peaks(image_curvature, threshold, 50)).tolist()
        h_d = spearmanr(mri_curvature, image_curvature)[0]
        return h_d

    spearman_msi = fhmeasure_aligned(curvature_msi, curvature_mri, threshold=0.12, sigma=np.pi/2)
    return spearman_msi

def display_curvature(image, points, curvature, padding=0):
    image_copy = np.pad(image, (padding,padding), 'constant')
    X = np.zeros_like(image_copy, dtype=np.float)
    x, y = points[:,0], points[:,1]
    for i in range(len(curvature)):
        X[y[i],x[i]] = curvature[i]
    X = np.ma.masked_where(X == 0, X)
    plt.clf()
    plt.imshow(image_copy, cmap="gray")
    plt.imshow(X,cmap="jet", vmin=0, vmax=0.25)
    plt.show()

## Open imzML
## This is a MS image where peaks are selected
imzml = io.open_imzml("/mnt/d/INRAE/MALDI/MSI_20190419_01/00/peaksel_prominence75.imzML")
spectra = io.get_full_spectra(imzml)
max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
msi_images = io.get_images_from_spectra(spectra, shape=(max_x,max_y))

## Open MRI image
mri_itk_image = sitk.ReadImage("/mnt/d/INRAE/Registration/BLE/250/50/mri_slice11_processed.png")
mri_image = sitk.GetArrayFromImage(mri_itk_image)

## Curvature values obtained by the VCM estimator
## of Lachaud et al. available in DGtal:
## More at https://dgtal.org/doc/stable/moduleVCM.html
## here, parameters r=4, R=5 for VCM computation
## The actual script used is available at https://github.com/fgrelard/MyDGtalContrib/blob/master/curvature.sh
## Requires DGtal library compilation
curvature_mri = np.loadtxt("/mnt/d/INRAE/Segmentation/mri/results/mri_slice11_contour_wopericarp_curvature.txt")
curvature_msi = np.loadtxt("/mnt/d/INRAE/Segmentation/coherence/results/peaksel_prominence75_spatialcoherence_evolution_contour0030_curvature.txt")
points_mri = np.loadtxt("/mnt/d/INRAE/Segmentation/mri/results/mri_slice11_contour_wopericarp_ensured.sdp",skiprows=1, dtype=int)
points_msi = np.loadtxt("/mnt/d/INRAE/Segmentation/coherence/results/peaksel_prominence75_spatialcoherence_evolution_contour0030_ensured.sdp", skiprows=1, dtype=int)

display_curvature(mri_image, points_mri, curvature_mri)

segmented_msi = segmentation_msi(msi_images)
print(msi_images.shape, segmented_msi.shape)
display_curvature(segmented_msi, points_msi, curvature_msi, padding=3)
spearman_msi = evaluation_segmentation(curvature_msi, curvature_mri)

print(spearman_msi)
