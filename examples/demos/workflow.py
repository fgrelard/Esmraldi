import esmraldi.segmentation as seg
import esmraldi.registration as reg
import esmraldi.fusion as fusion
import esmraldi.imzmlio as io
import esmraldi.imageutils as utils
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


def segmentation_msi(images, spatial_coherence=True):
    ## Find spatially coherent images in the image
    ## Images are binarized using the set of thresholds in quantiles
    ## Images that are spatially coherent are defined as images where the area
    ## of the largest connected component is greater than
    ## percentage*max_{im \in images} area(im)
    if spatial_coherence:
        relevant_set = seg.find_similar_images_spatial_coherence_percentage(images, percentage=0.95, quantiles=[60, 70, 80, 90])
    else:
        # Apply spatial chaos measure (Alexandrov et al., 2013)
        relevant_set = seg.find_similar_images_spatial_chaos(images, threshold=1.0022, quantiles=[60, 70, 80, 90])

    print("Segmentation method = " + ("coherence" if spatial_coherence else "chaos"))
    print("Number of relevant ion images =", relevant_set.shape[-1])
    ## Region growing
    ## First determine list of seeds from average image
    mean_image = np.uint8(cv.normalize(np.average(relevant_set, axis=2), None, 0, 255, cv.NORM_MINMAX))
    threshold = 53
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

def evaluation_segmentation(curvature_msi, curvature_mri, sigma=1.4):
    def best_spearman(image_curvature, mri_curvature):
        ## best spearman coefficient over the set of translations
        image_curvature = scipy.ndimage.gaussian_filter1d(np.copy(image_curvature), sigma)
        mri_curvature = scipy.ndimage.zoom(mri_curvature, len(image_curvature)/len(mri_curvature), order=2)
        mri_curvature = scipy.ndimage.gaussian_filter1d(np.copy(mri_curvature), sigma)
        best_s = 0
        for i in range(len(image_curvature)):
            tmp = np.roll(image_curvature, i)
            s = spearmanr(mri_curvature, tmp)[0]
            if s > best_s:
                best_s = s
        return best_s

    spearman_msi = best_spearman(curvature_msi, curvature_mri)
    return spearman_msi

def export_curvature_figure(filename, image, points, curvature, padding=0):
    image_copy = np.pad(image, (padding,padding), 'constant')
    X = np.zeros_like(image_copy, dtype=np.float)
    x, y = points[:,0], points[:,1]
    for i in range(len(curvature)):
        X[y[i],x[i]] = curvature[i]
    X = np.ma.masked_where(X == 0, X)
    plt.axis('off')
    plt.imshow(image_copy, cmap="gray")
    plt.imshow(X, cmap="jet", vmin=0.0, vmax=0.25)
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()


def register2D(fixed, moving, numberOfBins, is_best_rotation=False, array_moving=None, flipped=False, sampling_percentage=0.1, learning_rate=1.1, min_step=0.001, relaxation_factor=0.8):
    best_resampler = reg.register(fixed, moving, numberOfBins, sampling_percentage, find_best_rotation=is_best_rotation, learning_rate=learning_rate, min_step=min_step, relaxation_factor=relaxation_factor)
    return best_resampler

def apply_registration(image, best_resampler):
    try:
        out = best_resampler.Execute(image)
    except Exception as e:
        print("Problem with best_resampler")
        print(e)
        return None
    return out


def evaluation_registration(fixed, moving, registered):
    p, r = reg.quality_registration(fixed, registered, threshold=30)
    precision, recall = p[0], r[0]
    fmeasure = reg.fmeasure(precision, recall)
    size = moving.GetSize()
    scaled_registered = utils.resize(registered, (size[1], size[0]))
    mi = reg.mutual_information(moving, scaled_registered,50)
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(sitk.GetArrayFromImage(scaled_registered))
    # ax[1].imshow(sitk.GetArrayFromImage(moving))
    # plt.show()
    return precision, recall, fmeasure, mi

def export_registration_figure(fixed, registered):
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(registered), sitk.sitkUInt8)

    cimg = sitk.Compose(simg1, simg2, simg1//3.+simg2//1.5)

    plt.clf()
    plt.imshow(sitk.GetArrayFromImage(cimg))
    plt.axis('off')
    plt.show()


def apply_transform_itk(transformname, image):
    transform = sitk.ReadImage(transformname)
    t64 = sitk.Cast(transform, sitk.sitkVectorFloat64)
    field = sitk.DisplacementFieldTransform(t64)
    array = sitk.GetArrayFromImage(transform)
    outRegister = sitk.Resample(image, field, sitk.sitkNearestNeighbor, 0)
    outRegister = sitk.Cast(outRegister, sitk.sitkFloat32)
    return outRegister




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
curvature_msi_coherence = np.loadtxt("/mnt/d/INRAE/Segmentation/coherence/results/peaksel_prominence75_spatialcoherence_evolution_contour0030_curvature.txt")
curvature_msi_chaos = np.loadtxt("/mnt/d/INRAE/Segmentation/chaos2/results/peaksel_prominence75_spatialchaos2_evolution_contour0007_curvature.txt")
points_mri = np.loadtxt("/mnt/d/INRAE/Segmentation/mri/results/mri_slice11_contour_wopericarp_ensured.sdp",skiprows=1, dtype=int)
points_msi_coherence = np.loadtxt("/mnt/d/INRAE/Segmentation/coherence/results/peaksel_prominence75_spatialcoherence_evolution_contour0030_ensured.sdp", skiprows=1, dtype=int)
points_msi_chaos = np.loadtxt("/mnt/d/INRAE/Segmentation/chaos2/results/peaksel_prominence75_spatialchaos2_evolution_contour0007_ensured.sdp", skiprows=1, dtype=int)

## Transform variational registration
## Obtained through C++ ITK
## Module VariationalRegistration
## More info at https://github.com/InsightSoftwareConsortium/ITKVariationalRegistration
transformname = "/mnt/d/INRAE/Registration/BLE/250/50/results/itk/VariationalRegistration/maldi_slice11_2.mha"

segmented_msi_coherence = segmentation_msi(msi_images)
# segmented_msi_chaos = segmentation_msi(msi_images, spatial_coherence=False)

spearman_msi_coherence = evaluation_segmentation(curvature_msi_coherence, curvature_mri)
# spearman_msi_chaos = evaluation_segmentation(curvature_msi_chaos, curvature_mri)

export_curvature_figure("out/mri_curvature.png", mri_image, points_mri, curvature_mri)
export_curvature_figure("out/msi_curvature_coherence.png",segmented_msi_coherence, points_msi_coherence, curvature_msi_coherence, padding=3)
# export_curvature_figure("out/msi_curvature_chaos.png", segmented_msi_chaos, points_msi_chaos, curvature_msi_chaos, padding=3)


print("Spearman coherence=", spearman_msi_coherence)
# print("Spearman chaos=", spearman_msi_chaos)

segmented_msi_itk = sitk.Cast(sitk.GetImageFromArray(segmented_msi_coherence), sitk.sitkFloat32)
mri_itk_reg = sitk.Cast(sitk.GetImageFromArray(mri_image), sitk.sitkFloat32)
resampler = register2D(mri_itk_reg, segmented_msi_itk, 50)
registered_itk = apply_registration(segmented_msi_itk, resampler)
registered = sitk.GetArrayFromImage(registered_itk)
registered_variational_itk = apply_transform_itk(transformname, registered_itk)

p_affine, r_affine, f_affine, mi_affine = evaluation_registration(mri_itk_reg, segmented_msi_itk, registered_itk)
p_variational, r_variational, f_variational, mi_variational = evaluation_registration(mri_itk_reg, segmented_msi_itk, registered_variational_itk)

print("Affine registration")
print("p=", p_affine, "r=", r_affine, "fmeasure=", f_affine, "mi=", mi_affine)

print("Variational registration")
print("p=", p_variational, "r=", r_variational, "fmeasure=", f_variational, "mi=", mi_variational)

export_registration_figure(mri_itk_reg, registered_itk)
export_registration_figure(mri_itk_reg, registered_variational_itk)
