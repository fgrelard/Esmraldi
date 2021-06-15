import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import esmraldi.segmentation as seg
import esmraldi.spectraprocessing as sp
import esmraldi.imzmlio as imzmlio
from esmraldi.spectralviewer import SpectralViewer

def circle_layers(nb_layers, x, y, r, image, max_intensity):
    current_image = image
    for i in range(nb_layers, 0, -1):
        current_r = (i+1) * r / nb_layers
        current_i = (i+1) * max_intensity / nb_layers
        current_image = seg.fill_circle(x, y, current_r, current_image, current_i)
    return current_image


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--factor", help="Multiplicative factor between sum of peaks and sum of noise", default=1)

args = parser.parse_args()

multiplicative_factor = int(args.factor)

nb_points = 1000
size = 100
image = np.zeros((size, size, nb_points))

mean_noise = 20
std_dev_noise = 20
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
        current_image = circle_layers(5, 25, 50, 20, current_image, intensity_peak)
    current_image = circle_layers(5, 75, 50, 20, current_image, intensity_peak)
    image[..., current_mz] = current_image

indices_peak = [indices_peak] * size**2

image += noise
intensities, _ = imzmlio.get_spectra_from_images(image)
mzs = [[i for i in range(nb_points)]] * size**2

spectra = np.stack([mzs, intensities], axis=1)

spectra_tic = sp.normalization_tic(spectra)
spectra_sic = sp.normalization_sic(spectra, indices_peak, width_peak=2)

image_tic = imzmlio.get_images_from_spectra(spectra_tic, image.shape[:-1])
image_sic = imzmlio.get_images_from_spectra(spectra_sic, image.shape[:-1])


fig, ax = plt.subplots(3, 1)
spectral = SpectralViewer(ax, image, spectra)
fig.canvas.mpl_connect('button_press_event', spectral.onclick)
plt.show()

fig, ax = plt.subplots(3, 1)
spectral_tic = SpectralViewer(ax, image_tic, spectra_tic)
fig.canvas.mpl_connect('button_press_event', spectral_tic.onclick)

fig2, ax2 = plt.subplots(3, 1)
spectral_sic = SpectralViewer(ax2, image_sic, spectra_sic)
fig2.canvas.mpl_connect('button_press_event', spectral_sic.onclick)
plt.show()
