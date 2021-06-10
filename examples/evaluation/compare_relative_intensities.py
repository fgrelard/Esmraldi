import argparse

import numpy as np
import matplotlib.pyplot as plt

import esmraldi.imzmlio as imzmlio
import esmraldi.imageutils as utils
import SimpleITK as sitk

def read_imzml(inputname):
    imzml = imzmlio.open_imzml(inputname)
    spectra = imzmlio.get_full_spectra(imzml)
    max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
    max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
    max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
    image = imzmlio.get_images_from_spectra(spectra, (max_x, max_y, max_z))
    mzs, intensities = imzml.getspectrum(0)
    return spectra, image


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--reference", help="Reference MALDI image (imzML)")
parser.add_argument("-t", "--target", help="Target MALDI image (imzML)")
args = parser.parse_args()

referencename = args.reference
targetname = args.target

spectra_ref, reference_image = read_imzml(referencename)
spectra_target, target_image = read_imzml(targetname)

mzs_ref = spectra_ref[0, 0, ...]
mzs_target = spectra_target[0, 0, ...]

intensities_ref = spectra_ref[:, 1, ...]
intensities_target = spectra_target[:, 1, ...]

print(mzs_ref.shape, intensities_ref.shape)

reference_image_norm = imzmlio.normalize(reference_image)
target_image_norm = imzmlio.normalize(target_image)

print(reference_image_norm.shape, target_image_norm.shape)

max_diff_ref, max_diff_target = None, None
max_mz = 0
diff = 0
max_diff_index = 0

for i, mz in enumerate(mzs_ref):
    index_target = np.argwhere(mzs_target == mz).flatten()
    if not index_target:
        continue
    current_ref = reference_image_norm[..., i].astype(np.float64)
    current_target = target_image_norm[..., index_target[0]].astype(np.float64)
    mse = utils.mse_numpy(current_ref, current_target)
    if mse > diff:
        max_diff_ref = current_ref
        max_diff_target = current_target
        diff = mse
        max_mz = mz
        max_diff_index = np.argmax(np.abs(current_ref - current_target))

print(np.unravel_index(max_diff_index, reference_image_norm.shape[:-1]))
index_mz_max_diff = np.argmax(intensities_target[max_diff_index])

mz_max_diff_image = reference_image_norm[..., index_mz_max_diff]
print("Mz max diff=", mzs_ref[index_mz_max_diff])
print("Diff=", diff, "Mz=", max_mz)

utils.export_figure_matplotlib("reference.png", max_diff_ref, dpi=120)
utils.export_figure_matplotlib("target.png", max_diff_target, dpi=120)
utils.export_figure_matplotlib("highestpeak.png", mz_max_diff_image, dpi=120)
fig, ax = plt.subplots(1, 3)
ax[0].imshow(max_diff_ref)
ax[1].imshow(max_diff_target)
ax[2].imshow(mz_max_diff_image)
plt.show()
