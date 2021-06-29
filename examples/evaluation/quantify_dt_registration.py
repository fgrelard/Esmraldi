import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage.draw as draw
import SimpleITK as sitk
import esmraldi.imageutils as utils
import esmraldi.registration as reg
import skimage.transform as transform
import cv2 as cv

def create_data(mean_noise, stddev_noise):
    img = np.zeros((100, 100))

    coords = draw.ellipse(50, 50, 35, 13)
    set_ellipse = set((tuple(i) for i in np.array(coords).T))
    img[coords] = 120

    coords2 = draw.disk((50, 59), 5)
    set_disk = set((tuple(i) for i in np.array(coords2).T))
    set_intersection = set_disk.intersection(set_ellipse)
    inters_xy = tuple(np.array(list(set_intersection)).T)
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

    noise = np.random.normal(mean_noise, stddev_noise, img.shape)

    img += noise



    # rr, cc = draw.ellipse(50, 66, 35, 7)
    # img[rr, cc] = 0

    return img

mean_noise = 10
stddev_noise = 2

target = create_data(mean_noise, stddev_noise)
plt.imshow(target)
plt.show()

reference = transform.rotate(reference, 180)
reference = transform.resize(reference, (50, 50), order=0)
reference = np.pad(reference, 25)

dt_reference = utils.compute_DT(reference_itk)
dt_target = utils.compute_DT(target_itk)
