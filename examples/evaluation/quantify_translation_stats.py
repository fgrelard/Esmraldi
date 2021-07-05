import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage.draw as draw
import SimpleITK as sitk
import esmraldi.imageutils as utils
import esmraldi.registration as reg
import esmraldi.fusion as fusion
import skimage.transform as transform
import cv2 as cv
import esmraldi.sliceviewer as slicev
import esmraldi.imzmlio as imzmlio
from sklearn.decomposition import NMF, PCA
from scipy.stats import pearsonr
from scipy.spatial import distance

def create_data(deformation=False):
    if deformation:
        img = np.zeros((100, 100, 100))
    else:
        img = np.zeros((100, 100))

    coords = draw.ellipse(50, 50, 35, 13)
    set_ellipse = set((tuple(i) for i in np.array(coords).T))
    img[coords] = 120

    coords2 = draw.disk((50, 63), 5)
    set_disk = set((tuple(i) for i in np.array(coords2).T))
    set_intersection_middle = set_disk.intersection(set_ellipse)
    inters_middle_xy = tuple(np.array(list(set_intersection_middle)).T)
    img[inters_middle_xy] = 200

    if deformation:
        coords_disktop = draw.disk((34, 54), 4)
    else:
        coords_disktop = draw.disk((30, 55), 4)
    set_disk = set((tuple(i) for i in np.array(coords_disktop).T))
    set_intersection = set_disk.intersection(set_ellipse)
    inters_xy = tuple(np.array(list(set_intersection)).T)
    img[inters_xy] = 200

    coords_line = draw.line(82, 46, 60, 40)
    coords_line = tuple(np.hstack((coords_line, draw.line(82, 47, 60, 41))))
    img[coords_line] = 200

    if deformation:

        coords2 = draw.ellipse(50, 63, 7, 2)
        set_coords2 =  set((tuple(i) for i in np.array(coords2).T))
        set_intersection2 = set_coords2.difference(set_ellipse)
        inters2_xy = tuple(np.array(list(set_intersection2)).T)
        img[inters2_xy] = 250
        img[inters_middle_xy] = 250


        coords3 = draw.ellipse(76, 61, 7, 6)
        img[coords3] = 0

        coords_line = tuple(np.hstack((coords_line, draw.line(82, 48, 60, 42))))
        coords_line = tuple(np.hstack((coords_line, draw.line(82, 49, 60, 43))))
        img[coords_line] = 250

        img[inters_xy] = 250



        n_slice = 7

        img_ref = img.copy()

        for i in range(img.shape[-1]):
            if i % 10 < n_slice and i % 20 != n_slice-1:

                img[coords_line + (i,)] = 120
                img[inters_xy + (i,)] = 120
                img[inters_middle_xy + (i,)] = 120
                img[inters2_xy + (i,)] = 120

            #7=Disk middle
            if i % 10 == n_slice:
                img[coords_line + (i,)] = 120
                img[inters_xy + (i,)] = 120


            #8=disk top
            if i % 10 == n_slice + 1:
                img[coords_line + (i,)] = 120
                img[inters2_xy + (i,)] = 120
                img[inters_middle_xy + (i,)] = 120

            #9=Line
            if i % 10  == n_slice + 2:
                img[inters_xy + (i,)] = 120
                img[inters_middle_xy + (i,)] = 120
                img[inters2_xy + (i,)] = 120

            #6 every 20 slices= Line + disk middle
            if i % 20  == n_slice - 1:
                img[inters_xy + (i,)] = 120



        N = 100
        x = np.linspace(-np.pi,np.pi, N)
        sine1D = 20.0 + (20 * np.sin(x * 5.0))
        sine1D = np.uint8(sine1D)
        sine2D = np.tile(sine1D, (N,1))
        sine2D = sine2D.T
        sine2D = np.where(img[..., 0] == 0, 0, sine2D)

        img += sine2D[..., None]
        img_ref += sine2D[..., None]

        utils.export_figure_matplotlib("stats_target_all.png", img_ref[..., 0])


    return img



mean_noise = 10
stddev_noise = 2

reference = create_data()
target = create_data(deformation=True)


reference_itk = sitk.GetImageFromArray(reference)
target_itk = sitk.GetImageFromArray(target)

reference_norm = imzmlio.normalize(reference)
target_norm = imzmlio.normalize(target)

reference_norm = fusion.flatten(reference_norm)
target_norm = fusion.flatten(target_norm, is_spectral=True)


nmf = NMF(n_components=4, init='nndsvda', solver='cd', random_state=0, max_iter=1000)

fit_red = nmf.fit(target_norm)

W = fit_red.transform(target_norm)
H = fit_red.components_

image_eigenvectors = H.T
new_shape = target.shape[:-1] + (image_eigenvectors.shape[-1],)
image_eigenvectors = image_eigenvectors.reshape(new_shape)



image_eigenvectors_translated = image_eigenvectors

H_translated = H

image_eigenvectors_translated, translated_image = reg.register_component_images(reference, target, image_eigenvectors, 5)

H_translated = image_eigenvectors_translated.reshape(H.T.shape).T


fit_red.components_ = H
point = fit_red.transform(reference_norm)


# We use translated components ONLY for MRI reconstruction
fit_red.components_ = H_translated
point_translated = fit_red.transform(reference_norm)

#Normal components for MS images
fit_red.components_ = H
X_r = fit_red.transform(target_norm)

centers = X_r

labels = None
weights = [1 for i in range(centers.shape[1])]

similar_images, similar_mzs, distances = fusion.select_images(target, point, centers, weights,  np.array([i for i in range(target.shape[-1])]), labels, None)


similar_images_translated, similar_mzs_translated, distances_translated = fusion.select_images(target, point_translated, centers, weights,  np.array([i for i in range(target.shape[-1])]), labels, None)

_, idx = np.unique(similar_mzs % 10, return_index=True)
_, idx_translated = np.unique(similar_mzs_translated % 10, return_index=True)

mzs_order = similar_mzs[np.sort(idx)]
mzs_translated_order = similar_mzs_translated[np.sort(idx_translated)]

print(mzs_order, mzs_translated_order)

w_reference = point / np.sum(point)
reference_reconstructed = fusion.reconstruct_image_from_components(image_eigenvectors, w_reference.T)
reference_reconstructed = reference_reconstructed.T
reference_reconstructed = imzmlio.normalize(reference_reconstructed)

w_reference_translated = point_translated / np.sum(point_translated)
reference_reconstructed_translated = fusion.reconstruct_image_from_components(image_eigenvectors_translated, w_reference_translated.T)
reference_reconstructed_translated = reference_reconstructed_translated.T
reference_reconstructed_translated = imzmlio.normalize(reference_reconstructed_translated)

reference_norm = imzmlio.normalize(reference)
reference_reconstructed_norm = imzmlio.normalize(reference_reconstructed)
reference_reconstructed_norm_translated = imzmlio.normalize(reference_reconstructed_translated)

i = np.where((reference_reconstructed_norm>0))
diff_reconstruction = np.mean(np.abs(reference_reconstructed[i] - reference[i]))/np.max(reference_reconstructed)
diff_reconstruction_translated = np.mean(np.abs(reference_reconstructed_translated[i] - reference[i]))/np.max(reference_reconstructed_translated)

p_r, _ = pearsonr(reference_norm.flatten(), reference_reconstructed_norm.flatten())
p_r_translated, _ = pearsonr(reference_norm.flatten(), reference_reconstructed_norm_translated.flatten())

cosine_r = distance.cosine(reference_norm.flatten(), reference_reconstructed_norm.flatten())
cosine_r_translated = distance.cosine(reference_norm.flatten(), reference_reconstructed_norm_translated.flatten())
print("Average diff NMF reconstruction (percentage)=", "{:.5f}".format(diff_reconstruction),  "{:.5f}".format(diff_reconstruction_translated), "pearson", "{:.5f}".format(p_r),  "{:.5f}".format(p_r_translated), "cosine", "{:.5f}".format(1-cosine_r),  "{:.5f}".format(1-cosine_r_translated))

utils.export_figure_matplotlib("stats_reference.png", reference)
utils.export_figure_matplotlib("stats_target_0.png", target[..., 0])
utils.export_figure_matplotlib("stats_target_6.png", target[..., 6])
utils.export_figure_matplotlib("stats_target_7.png", target[..., 7])
utils.export_figure_matplotlib("stats_target_8.png", target[..., 8])
utils.export_figure_matplotlib("stats_target_9.png", target[..., 9])
utils.export_figure_matplotlib("stats_reconstructed.png", reference_reconstructed)
utils.export_figure_matplotlib("stats_reconstructed_translated.png", reference_reconstructed_translated)
