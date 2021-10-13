"""
Evaluation of the normalization procedures
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import esmraldi.segmentation as seg
import esmraldi.spectraprocessing as sp
import esmraldi.imzmlio as imzmlio
from esmraldi.spectralviewer import SpectralViewer
import esmraldi.imageutils as utils
import skimage.draw as draw
from scipy.stats import pearsonr

def circle_layers(nb_layers, x, y, r, image, max_intensity):
    current_image = image
    for i in range(nb_layers, 0, -1):
        current_r = i * r / nb_layers
        current_i = i * max_intensity / nb_layers
        current_image = seg.fill_circle(x, y, current_r, current_image, current_i)
    return current_image

def reduce_factor(nb_layers, x, y, r, image):
    new_image = image.copy()
    for i in range(nb_layers, 0, -1):
        current_r = i * r / nb_layers
        current_factor = 1/(nb_layers-i+2)
        rr, cc = draw.disk((y,x), current_r, shape=image.shape[:-1])
        new_image[rr, cc, ...] = image[rr, cc, ...] * current_factor
    return new_image

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--factor", help="Multiplicative factor between sum of peaks and sum of noise", default=1)

args = parser.parse_args()

multiplicative_factor = int(args.factor)

nb_points = 1000
size = 100
image = np.zeros((size, size, nb_points))

mean_noise = 20
std_dev_noise = 15
noise = np.random.normal(mean_noise, std_dev_noise, (100, 100, 1000))


nb_peaks = 10
nb_points_without_peaks = nb_points - nb_peaks

sum_noise = nb_points_without_peaks * mean_noise

intensity_peak = sum_noise / nb_peaks * multiplicative_factor
print(intensity_peak)
spacing_mz_peak = int(nb_points / nb_peaks)

indices_peak = []

for i in range(nb_peaks):
    current_mz = spacing_mz_peak * i
    indices_peak.append(current_mz)

    current_image = image[..., current_mz]
    if i < nb_peaks//2:
        current_image = circle_layers(1, 25, 50, 20, current_image, intensity_peak)
    current_image = circle_layers(1, 75, 50, 20, current_image, intensity_peak)
    image[..., current_mz] = current_image

indices_peak_image = [indices_peak] * size**2

image += noise
theoretical_image = image.copy()

image = reduce_factor(5, 25, 50, 20, image)
image = reduce_factor(5, 75, 50, 20, image)
intensities, _ = imzmlio.get_spectra_from_images(image)
mzs = [[i for i in range(nb_points)]] * size**2

spectra = np.stack([mzs, intensities], axis=1)
print(spectra.shape)

spectra_tic = sp.normalization_tic(spectra)
spectra_sic = sp.normalization_sic(spectra, indices_peak_image, width_peak=2)

image_tic = imzmlio.get_images_from_spectra(spectra_tic, image.shape[:-1]).transpose((1, 0, 2))
image_sic = imzmlio.get_images_from_spectra(spectra_sic, image.shape[:-1]).transpose((1, 0, 2))

theoretical_image_area = theoretical_image[..., indices_peak]

image_tic_area = image_tic[..., indices_peak]
image_sic_area = image_sic[..., indices_peak]
pearson_tic = pearsonr(theoretical_image.flatten(), image_tic.flatten())
pearson_sic = pearsonr(theoretical_image.flatten(), image_sic.flatten())
pearson_tic_area = pearsonr(theoretical_image_area.flatten(), image_tic_area.flatten())
pearson_sic_area = pearsonr(theoretical_image_area.flatten(), image_sic_area.flatten())
print("Pearson TIC=", pearson_tic, ", SIC=", pearson_sic)
print("Pearson AREA TIC=", pearson_tic_area, ", SIC=", pearson_sic_area)

utils.export_figure_matplotlib("image_original.png", image[..., 0], dpi=111)
utils.export_figure_matplotlib("image_tic.png", image_tic[..., 0], dpi=111)
utils.export_figure_matplotlib("image_sic.png", image_sic[..., 0], dpi=111)

fig, ax = plt.subplots(3, 1)
spectral = SpectralViewer(ax, image, spectra, cmap="gray")
fig.canvas.mpl_connect('button_press_event', spectral.onclick)
plt.show()

fig, ax = plt.subplots(3, 1)
spectral_tic = SpectralViewer(ax, image_tic, spectra_tic, cmap="gray")
fig.canvas.mpl_connect('button_press_event', spectral_tic.onclick)

fig2, ax2 = plt.subplots(3, 1)
spectral_sic = SpectralViewer(ax2, image_sic, spectra_sic, cmap="gray")
fig2.canvas.mpl_connect('button_press_event', spectral_sic.onclick)
plt.show()
