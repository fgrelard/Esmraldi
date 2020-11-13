import esmraldi.segmentation as seg
import esmraldi.registration as reg
import esmraldi.fusion as fusion
import esmraldi.imzmlio as io
import numpy as np
import cv2 as cv
import os

import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.signal as signal
import SimpleITK as sitk

from scipy.stats.stats import spearmanr
from skimage.filters import threshold_otsu, rank
from skimage import measure
from skimage.morphology import binary_erosion, opening, disk
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


def segmentation_msi(images, spatial_coherence=True):
    ## Find spatially coherent images in the image
    ## Images are binarized using the set of thresholds in quantiles
    ## Images that are spatially coherent are defined as images where the area
    ## of the largest connected component is greater than
    ## a threshold
    threshold = 50
    padding = 3
    images = np.pad(images, (padding,padding), 'constant')
    if spatial_coherence:
        relevant_set = seg.find_similar_images_spatial_coherence(images, factor=2100, quantiles=[60, 70, 80, 90])
    else:
        # Apply spatial chaos measure (Alexandrov et al., 2013)
        relevant_set = seg.find_similar_images_spatial_chaos(images, threshold=1.003, quantiles=[60, 70, 80, 90])

    relevant_set = seg.sort_size_ascending(relevant_set, threshold)

    print("Segmentation method = " + ("coherence" if spatial_coherence else "chaos"))
    print("Number of relevant ion images =", relevant_set.shape[-1], "\n")
    ## Region growing
    ## First determine list of seeds from average image
    mean_image = np.uint8(cv.normalize(np.average(relevant_set, axis=2), None, 0, 255, cv.NORM_MINMAX))
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
    selem = disk(1)
    mask = opening(mask, selem)
    masked_mean_image = np.ma.array(mean_image, mask=mask)
    masked_mean_image = masked_mean_image.filled(0)
    masked_mean_image = masked_mean_image[padding:-padding, padding:-padding]
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

def resize_register(fixed, moving):
    #resizing

    fimg = fixed
    mimg = moving

    dim_mimg = mimg.GetDimension()
    fimg_size = fimg.GetSize()
    mimg_size = (fimg_size[0], int(mimg.GetSize()[1]*fimg_size[0]/mimg.GetSize()[0])) + ((mimg.GetSize()[2],) if mimg.GetDimension() > 2 else ())

    mimg = seg.resize(mimg, mimg_size)
    sx = fimg.GetSpacing()
    spacing = tuple([sx[0] for i in range(dim_mimg)])
    mimg.SetSpacing(spacing)

    mimg = sitk.Cast(sitk.RescaleIntensity(mimg), sitk.sitkUInt8)
    mimg = sitk.Cast(sitk.RescaleIntensity(mimg), sitk.sitkFloat32)
    return mimg

def register2D(fixed, moving, numberOfBins, sampling_percentage=0.1, learning_rate=1.1, min_step=0.001, relaxation_factor=0.8):

    best_resampler = reg.register(fixed, moving, numberOfBins, sampling_percentage)
    return best_resampler

def apply_registration(image, fixed, best_resampler):
    dim_image = image.GetDimension()
    if dim_image == 2:
        try:
            out = best_resampler.Execute(image)
        except Exception as e:
            print(e)
            return None
    else:
        size = fixed.GetSize()
        image = resize_register(fixed, image)
        pixel_type = image.GetPixelID()
        out = sitk.Image(size[0], size[1], image.GetSize()[2], pixel_type)
        for i in range(image.GetSize()[2]):
            out2D = best_resampler.Execute(image[:,:,i])
            out2D = sitk.JoinSeries(out2D)
            out = sitk.Paste(out, out2D, out2D.GetSize(), destinationIndex=[0,0,i])
        pixel_type_reg = image.GetPixelID()
        if pixel_type_reg >= sitk.sitkFloat32:
            out = sitk.Cast(out, sitk.sitkFloat32)
    return out


def apply_transform_itk(transformname, image):
    transform = sitk.ReadImage(transformname)
    t64 = sitk.Cast(transform, sitk.sitkVectorFloat64)
    field = sitk.DisplacementFieldTransform(t64)
    array = sitk.GetArrayFromImage(transform)
    dim = image.GetDimension()
    identity = np.identity(dim).tolist()
    flat_list = [item for sublist in identity for item in sublist]
    direction = tuple(flat_list)
    image.SetDirection(flat_list)
    size = image.GetSize()
    if dim == 2:
        outRegister = sitk.Resample(image, field, 2, 0)
        outRegister = sitk.Cast(outRegister, sitk.sitkFloat32)
    else:
        pixel_type = image.GetPixelID()
        outRegister = sitk.Image(size[0], size[1], size[2], pixel_type )
        for i in range(size[2]):
            slice = image[:,:,i]
            outSlice = sitk.Resample(slice, field, sitk.sitkNearestNeighbor, 0)
            outSlice = sitk.JoinSeries(outSlice)
            outRegister = sitk.Paste(outRegister, outSlice, outSlice.GetSize(), destinationIndex=[0, 0, i])
    return outRegister


def evaluation_registration(fixed, moving, registered, affine):
    p, r = reg.quality_registration(fixed, registered, threshold=40)
    precision, recall = p, r
    fmeasure = reg.fmeasure(precision, recall)
    size = moving.GetSize()
    scaled_registered = seg.resize(registered, (size[1], size[0]))
    mi = reg.mutual_information(affine, registered)
    return precision, recall, fmeasure, mi

def export_registration_figure(name, fixed, registered):
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(registered), sitk.sitkUInt8)

    cimg = sitk.Compose(simg1, simg2, simg1//3.+simg2//1.5)

    plt.imshow(sitk.GetArrayFromImage(cimg))
    plt.axis('off')
    plt.savefig(name, transparent=True, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()




def export_nmf_components(name, image_eigenvectors):
    for i in range(image_eigenvectors.shape[-1]):
        n, ext = os.path.splitext(name)
        current_name = n + "_" + str(i) + ext
        plt.imshow(image_eigenvectors[..., i], cmap="gray")
        plt.axis("off")
        plt.savefig(current_name, transparent=True, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

def export_reconstructed_nmf_image(name, mri_reconstructed):
    plt.imshow(mri_reconstructed, cmap="gray")
    plt.axis('off')
    plt.savefig(name, transparent=True, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()


def export_top3(name, image):
    for i in range(image.shape[-1]):
        n, ext = os.path.splitext(name)
        current_name = n + "_" + str(i+1) + ext
        plt.imshow(image[..., i], cmap="gray")
        plt.axis("off")
        plt.savefig(current_name, transparent=True, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()


## Open imzML
## This is a MS image where peaks are selected
imzml = io.open_imzml("data/wheat/msi.imzML")
spectra = io.get_full_spectra(imzml)
max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
msi_images = io.get_images_from_spectra(spectra, shape=(max_x,max_y))

## Open MRI image
mri_itk_image = sitk.ReadImage("data/wheat/mri.png")
mri_image = sitk.GetArrayFromImage(mri_itk_image)

## Curvature values obtained by the VCM estimator
## of Lachaud et al. available in DGtal:
## More at https://dgtal.org/doc/stable/moduleVCM.html
## here, parameters r=4, R=5 for VCM computation
## The actual script used is available at https://github.com/fgrelard/MyDGtalContrib/blob/master/curvature.sh
## Requires DGtal library compilation
curvature_mri = np.loadtxt("data/wheat/mri_curvature.txt")
curvature_msi_coherence = np.loadtxt("data/wheat/msi_coherence_curvature.txt")
curvature_msi_chaos = np.loadtxt("data/wheat/msi_chaos_curvature.txt")
points_mri = np.loadtxt("data/wheat/mri_curvature_points.sdp",skiprows=1, dtype=int)
points_msi_coherence = np.loadtxt("data/wheat/msi_coherence_curvature_points.sdp", skiprows=1, dtype=int)
points_msi_chaos = np.loadtxt("data/wheat/msi_chaos_curvature_points.sdp", skiprows=1, dtype=int)

## Transform variational registration
## Obtained through C++ ITK
## Module VariationalRegistration
## More info at https://github.com/InsightSoftwareConsortium/ITKVariationalRegistration
## Parameters:
### tau=7.5e-4
### levels=5
### nb_iterations=200
### mu=0.8, lambda=0.9
### regularization=elastic
### forces=ssd
### image domain=fixed image forces
### search space=diffeomorphic
transformname = "data/wheat/transform_variational.mha"

## Registered images FFD1 and FFD2 obtained through ITK's DeformableRegistration4
## https://itk.org/Doxygen/html/Examples_2RegistrationITKv4_2DeformableRegistration4_8cxx-example.html
## FFD 1, good shape, bad intensities
### parameters : control points=25, order=2
## FFD 2, good intensities, bad shape
### parameters : control points=9, order=2
ffd_1_name = "data/wheat/ffd_1.tif"
ffd_2_name = "data/wheat/ffd_2.tif"

print("Segmentation")
print("------------")
segmented_msi_coherence = segmentation_msi(msi_images)
segmented_msi_chaos = segmentation_msi(msi_images, spatial_coherence=False)

spearman_msi_coherence = evaluation_segmentation(curvature_msi_coherence, curvature_mri)
spearman_msi_chaos = evaluation_segmentation(curvature_msi_chaos, curvature_mri)

export_curvature_figure("out/mri_curvature.png", mri_image, points_mri, curvature_mri)
export_curvature_figure("out/msi_curvature_coherence.png",segmented_msi_coherence, points_msi_coherence, curvature_msi_coherence, padding=3)
export_curvature_figure("out/msi_curvature_chaos.png", segmented_msi_chaos, points_msi_chaos, curvature_msi_chaos, padding=3)


print("Spearman coherence=", "{:.3f}".format(spearman_msi_coherence))
print("Spearman chaos=", "{:.3f}".format(spearman_msi_chaos))
print()

print("Registration")
print("------------")
segmented_msi_itk = sitk.Cast(sitk.GetImageFromArray(segmented_msi_coherence), sitk.sitkFloat32)
mri_itk_reg = sitk.Cast(sitk.GetImageFromArray(mri_image), sitk.sitkFloat32)
segmented_msi_itk = resize_register(mri_itk_reg, segmented_msi_itk)
ffd_1 = sitk.ReadImage(ffd_1_name)
ffd_2 = sitk.ReadImage(ffd_2_name)

resampler = register2D(mri_itk_reg, segmented_msi_itk, 27, 1.0)
registered_itk = apply_registration(segmented_msi_itk, mri_itk_reg, resampler)
registered = sitk.GetArrayFromImage(registered_itk)
registered_variational_itk = apply_transform_itk(transformname, registered_itk)

segmented_msi_itk = sitk.Cast(segmented_msi_itk, sitk.sitkUInt8)
p_affine, r_affine, f_affine, mi_affine = evaluation_registration(mri_itk_reg, segmented_msi_itk, registered_itk, registered_itk)
p_variational, r_variational, f_variational, mi_variational = evaluation_registration(mri_itk_reg, segmented_msi_itk, registered_variational_itk, registered_itk)

p_ffd_1, r_ffd_1, f_ffd_1, mi_ffd_1 = evaluation_registration(mri_itk_reg, segmented_msi_itk, ffd_1, registered_itk)
p_ffd_2, r_ffd_2, f_ffd_2, mi_ffd_2 = evaluation_registration(mri_itk_reg, segmented_msi_itk, ffd_2, registered_itk)


print("Affine registration")
print("p=", "{:.3f}".format(p_affine), "\tr=", "{:.3f}".format(r_affine), "\tfmeasure=", "{:.3f}".format(f_affine), "\tm_i=", "{:.3f}".format(mi_affine), "\n")

print("Variational registration")
print("p=", "{:.3f}".format(p_variational), "\tr=", "{:.3f}".format(r_variational), "\tfmeasure=", "{:.3f}".format(f_variational), "\tm_i=", "{:.3f}".format(mi_variational), "\n")

print("FFD_1 registration")
print("p=", "{:.3f}".format(p_ffd_1), "\tr=", "{:.3f}".format(r_ffd_1), "\tfmeasure=", "{:.3f}".format(f_ffd_1), "\tm_i=", "{:.3f}".format(mi_ffd_1), "\n")

print("FFD_2 registration")
print("p=", "{:.3f}".format(p_ffd_2), "\tr=", "{:.3f}".format(r_ffd_2), "\tfmeasure=", "{:.3f}".format(f_ffd_2), "\tm_i=", "{:.3f}".format(mi_ffd_2), "\n")

export_registration_figure("out/registered_affine.png", mri_itk_reg, registered_itk)
export_registration_figure("out/registered_variational.png", mri_itk_reg, registered_variational_itk)
export_registration_figure("out/registered_ffd1.png", mri_itk_reg, ffd_1)
export_registration_figure("out/registered_ffd2.png", mri_itk_reg, ffd_2)

msi_images_itk = sitk.GetImageFromArray(msi_images.T)

## Affine followed by variational on the full MSI image
msi_registered_itk = apply_registration(msi_images_itk, mri_itk_reg, resampler)
msi_registered_itk = apply_transform_itk(transformname, msi_registered_itk)


## Joint analysis
print("Joint statistical analysis")
print("--------------------------")
mzs, intensities = imzml.getspectrum(0)
threshold_mz = 569
image = np.transpose(sitk.GetArrayFromImage(msi_registered_itk), (1, 2, 0))
image = image[..., mzs >= threshold_mz]
mzs = mzs[mzs >= threshold_mz]
mzs = np.around(mzs, decimals=2)
mzs = mzs.astype(str)

image = io.normalize(image)
image_norm = fusion.flatten(image)
mri_norm = io.normalize(mri_image)
mri_norm = fusion.flatten(mri_norm)


n = 7
nmf = NMF(n_components=n, init='nndsvda', solver='cd', random_state=0)
fit_red = nmf.fit(image_norm)
point = fit_red.transform(mri_norm)
X_r = fit_red.transform(image_norm)
image_eigenvectors = fit_red.components_
centers = X_r
point_mri = point
image_eigenvectors = image_eigenvectors.T
new_shape = image.shape[:-1] + (image_eigenvectors.shape[-1],)
image_eigenvectors = image_eigenvectors.reshape(new_shape)

weights = [1 for i in range(centers.shape[1])]
similar_images, similar_mzs, distances = fusion.select_images(image,point_mri, centers, weights,  mzs, None, None)

w_mri = point / np.sum(point)
mri_reconstructed = np.sum([image_eigenvectors[..., i].T * w_mri.T[i] for i in range(len(w_mri.T))], axis=0)
mri_reconstructed = mri_reconstructed.T
i = np.where((mri_reconstructed>0))
diff_reconstruction = np.mean(np.abs(mri_reconstructed - mri_image))/np.max(mri_image)
print("Average diff NMF reconstruction (percentage)=", "{:.5f}".format(diff_reconstruction))

similar_mzs = similar_mzs.astype(np.float)
cosines = cosine_similarity(image_norm, mri_norm)
indices = [i for i in range(len(cosines))]
indices.sort(key=lambda x:cosines[x][0], reverse=True)
indices_array = np.array(indices)
similar_mzs_cosine = np.take(mzs, indices).astype(np.float)

## M/Z lists of the top 10 ions correlating the most spatially with
## the MRI image
## Comparison with cosine similarity measure
print("Ours: m/z of highest correlations=", similar_mzs[:10])
print("Cosine: m/z of highest correlations=", similar_mzs_cosine[:10])

## Deviation between our method and cosine on the
## rankings found on the top 20 of the most correlated ion images
dists = []
number = 20
for i in range(number):
    elem = float(similar_mzs_cosine[i])
    j, = np.where(similar_mzs == elem)
    dists.append(abs(i - j))

dists = np.median(dists)
print("Difference in rankings w.r.t cosine similarity (ratio)=", "{:.5f}".format(dists/len(similar_mzs_cosine)))

## Export joint analysis figures
export_nmf_components("out/nmf_components.png", image_eigenvectors)
export_reconstructed_nmf_image("out/nmf_reconstructed.png", mri_reconstructed)
export_top3("out/top.png", similar_images[..., 0:3])
