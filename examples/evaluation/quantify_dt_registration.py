"""
Evaluation of the registration
with distance transformed images
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage.draw as draw
import SimpleITK as sitk
import esmraldi.imageutils as utils
import esmraldi.registration as reg
import skimage.transform as transform
import cv2 as cv

def create_data(deformation=False):
    img = np.zeros((100, 100))

    coords = draw.ellipse(50, 50, 35, 13)
    set_ellipse = set((tuple(i) for i in np.array(coords).T))
    img[coords] = 120

    coords2 = draw.disk((50, 63), 5)
    set_disk = set((tuple(i) for i in np.array(coords2).T))
    set_intersection = set_disk.intersection(set_ellipse)
    inters_xy = tuple(np.array(list(set_intersection)).T)
    img[inters_xy] = 200
    if deformation:
        img[inters_xy] = 200

    coords_disktop = draw.disk((30, 55), 7)
    set_disk = set((tuple(i) for i in np.array(coords_disktop).T))
    set_intersection = set_disk.intersection(set_ellipse)
    inters_xy = tuple(np.array(list(set_intersection)).T)
    img[inters_xy] = 200

    coords_line = draw.line(82, 46, 60, 40)
    img[coords_line] = 200
    coords_line = draw.line(82, 47, 60, 41)
    img[coords_line] = 200

    if deformation:
        coords = draw.ellipse(50, 37, 7, 5)

        img[coords] = 0

        coords2 = draw.ellipse(50, 63, 7, 3)
        set_coords2 =  set((tuple(i) for i in np.array(coords2).T))
        set_intersection2 = set_coords2.difference(set_ellipse)
        inters2_xy = tuple(np.array(list(set_intersection2)).T)
        img[inters2_xy] = 200

        coords3 = draw.ellipse(76, 61, 7, 6)
        img[coords3] = 0

        coords_line = draw.line(82, 48, 60, 42)
        img[coords_line] = 200
        coords_line = draw.line(82, 49, 60, 43)
        img[coords_line] = 200

        set_intersection = set_disk.intersection(set_ellipse)
        inters_xy = tuple(np.array(listg(set_intersection)).T)
        img[inters_xy] = 200

        N = 100
        x = np.linspace(-np.pi,np.pi, N)
        sine1D = 20.0 + (20 * np.sin(x * 5.0))
        sine1D = np.uint8(sine1D)
        sine2D = np.tile(sine1D, (N,1))
        sine2D = sine2D.T
        sine2D = np.where(img == 0, 0, sine2D)
        img += sine2D

    return img

mean_noise = 10
stddev_noise = 5

reference = create_data()
target = create_data(deformation=True)

target_original = target.copy()
target = transform.rotate(target, 37, order=0)
target = transform.resize(target, (50, 50), order=1)
target = np.pad(target, 25)

# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(reference)
# ax[1].imshow(target_original)
# ax[2].imshow(target)
# plt.show()

reference_binary = reference.copy()
target_binary = target.copy()
reference_binary[reference_binary > 0] = 255
target_binary[target_binary > 0] = 255


noise2 = np.random.normal(mean_noise, stddev_noise, reference.shape)
target[target > 0] += noise2[target > 0]

plt.imshow(target)
plt.show()

reference_itk = sitk.GetImageFromArray(reference)
target_original_itk = sitk.GetImageFromArray(target_original)
target_itk = sitk.GetImageFromArray(target)
reference_binary = sitk.GetImageFromArray(reference_binary)
target_binary = sitk.GetImageFromArray(target_binary)

dt_reference = utils.compute_DT(reference_itk)
dt_target = utils.compute_DT(target_itk)
dt_target_original = utils.compute_DT(target_original_itk)


R_intensity = reg.register(reference_itk, target_itk, 10, 0.001, find_best_rotation=True, use_DT=False, update_DT=False, learning_rate=0.00000001)
out_intensity = R_intensity.Execute(target_itk)


R_binary = reg.register(reference_binary, target_binary, 10, 0.001, find_best_rotation=True, use_DT=False, update_DT=False, learning_rate=0.00000001)
out_binary = R_binary.Execute(target_itk)

R_dt = reg.register(reference_itk, target_itk, 10, 0.001, find_best_rotation=True, use_DT=True, update_DT=False, learning_rate=0.00000001)
out_dt = R_dt.Execute(target_itk)

R_shape = reg.register(reference_itk, target_itk, 10, 0.001, find_best_rotation=True, use_DT=True, update_DT=True, learning_rate=0.00000001)
out_shape = R_shape.Execute(target_itk)

sitk.WriteImage(sitk.Cast(out_intensity, sitk.sitkUInt8), "registered_intensity.tif")
sitk.WriteImage(sitk.Cast(out_shape, sitk.sitkUInt8), "registered_shape.tif")
sitk.WriteImage(dt_reference, "dt_reference.tif")
sitk.WriteImage(dt_target, "dt_target.tif")
sitk.WriteImage(dt_target_original, "dt_target_original.tif")
sitk.WriteImage(sitk.Cast(reference_itk, sitk.sitkUInt8), "reference.tif")
sitk.WriteImage(sitk.Cast(target_itk, sitk.sitkUInt8), "target.tif")
sitk.WriteImage(sitk.Cast(target_original_itk, sitk.sitkUInt8), "target_original.tif")



fig, ax = plt.subplots(1, 6)
ax[0].imshow(reference)
ax[1].imshow(target)
ax[2].imshow(sitk.GetArrayFromImage(out_intensity))
ax[3].imshow(sitk.GetArrayFromImage(out_binary))
ax[4].imshow(sitk.GetArrayFromImage(out_dt))
ax[5].imshow(sitk.GetArrayFromImage(out_shape))
plt.show()
