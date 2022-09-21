import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp
import esmraldi.fusion as fusion
import esmraldi.utils as utils
import SimpleITK as sitk
import scipy.spatial.distance as distance
from sklearn.cross_decomposition import PLSRegression, CCA

def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    if mask.ndim > 2:
        mask = rgb2gray(mask)
    mask = mask.T
    return mask

def read_imzml(input_name, normalization):
    if input_name.lower().endswith(".imzml"):
        imzml = io.open_imzml(input_name)
        spectra = io.get_spectra(imzml)
        print(spectra.shape)
        coordinates = imzml.coordinates
        max_x = max(coordinates, key=lambda item:item[0])[0]
        max_y = max(coordinates, key=lambda item:item[1])[1]
        max_z = max(coordinates, key=lambda item:item[2])[2]
        shape = (max_x, max_y, max_z)
        mzs = np.unique(np.hstack(spectra[:, 0]))
        mzs = mzs[mzs>0]
    return spectra, mzs, shape, imzml.coordinates

def indices_peaks(peaks, other_peaks):
    indices = utils.indices_search_sorted(other_peaks, peaks)
    current_step = 14 * other_peaks / 1e6
    indices_ppm = np.abs(peaks[indices] - other_peaks) < current_step
    indices = indices[indices_ppm]
    return indices

def extract_common_mzs(mzs):
    peaks = mzs[0].copy()
    for i in range(1, len(mzs)):
        other_peaks = mzs[i]
        indices = indices_peaks(peaks, other_peaks)
        peaks = peaks[indices]
    mzs_indices = []
    for i in range(len(mzs)):
        current_peaks = mzs[i]
        indices = indices_peaks(current_peaks, peaks)
        mzs_indices.append(indices)
    return peaks, mzs_indices

def normalize_flatten(spectra, coordinates, shape, normalization=True):
    if normalization:
        print("normalization")
        spectra = sp.normalization_tic(spectra)
    full_spectra = io.get_full_spectra_dense(spectra, coordinates, shape)
    images = io.get_images_from_spectra(full_spectra, shape)
    images = io.normalize(images)
    image_flatten = fusion.flatten(images, is_spectral=True).T
    return image_flatten

def read_regions(region_names):
    regions = []
    print("Read image")
    for region_name in region_names:
        region = read_image(region_name)
        regions.append(region)

    regions = np.transpose(np.array(regions), (1,2,0))
    regions = io.normalize(regions)
    region_flatten = fusion.flatten(regions, is_spectral=True).T
    return region_flatten

def extract_indexslices_regions(super_regions):
    L = []
    start = 0
    for i, s in enumerate(super_regions):
        end = s.shape[0]
        step = 1
        for j in range(s.shape[-1]):
            L.append(slice(start, start+end, step))
        start += end
    return L

def extract_indices_combined_regions(super_regions, indices_combined):
    lengths = [s.shape[-1] for s in super_regions]
    cumlen = np.cumsum(lengths)-1
    super_slices = []
    super_indices = []
    for ind in indices:
        region_indices = np.searchsorted(cumlen, ind)
        super_indices.append(region_indices)
        slices = []
        for reg_index in region_indices:
            sr = super_regions[reg_index]
            start = np.sum([super_regions[i].shape[0] for i in range(reg_index)], dtype=int)
            end = start + sr.shape[0]
            slices.append(slice(start, end, 1))
        super_slices.append(slices)
    return super_slices, super_indices



def indices_category_region(names):
    unique, indices = np.unique(names, return_inverse=True)
    indices = [np.where(indices == i)[0].astype(int) for i in range(len(unique))]
    return unique, indices


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML", nargs="+", type=str)
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str, action="append")
parser.add_argument("-t", "--target", help="Target .imzML")
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", action="store_true")
parser.add_argument("-o", "--output", help="Output files")
args = parser.parse_args()

input_name = args.input
target_name = args.target
region_names = args.regions
normalization = args.normalization
outname = args.output

super_regions = []
for dataset_regions in region_names:
    regions = read_regions(dataset_regions)
    super_regions.append(regions)

shape_super_regions = np.sum([s.shape for s in super_regions], axis=0)

all_names = [os.path.splitext(os.path.basename(name))[0] for r in region_names for name in r]
binders = [n.split("_")[0] for n in all_names]
unique_all, indices_all = indices_category_region(all_names)
unique_binders, indices_binders = indices_category_region(binders)
unique_names = np.concatenate((unique_all, unique_binders))
indices = indices_all + indices_binders
slices = extract_indexslices_regions(super_regions)
unique_slices = np.unique(slices)
slices_combined, indices_combined = extract_indices_combined_regions(super_regions, indices)

shape_super_regions[-1] = len(unique_names)
print(shape_super_regions)

combined_regions = np.zeros(shape_super_regions)
index_combined = 0
# for i, s in enumerate(super_regions):
#     for j in range(s.shape[-1]):
#         r = s[..., j]
#         combined_regions[slices[index_combined], index_combined] = r
#         index_combined += 1

lengths = [s.shape[-1] for s in super_regions]
cumlen = np.cumsum(lengths)
cumlen = np.concatenate([[0], cumlen])
for i, slices in enumerate(slices_combined):
    sr_indices = indices_combined[i]
    reg_indices = indices[i]
    reg_indices -= cumlen[sr_indices]
    for j, sr_index in enumerate(sr_indices):
        reg_index = reg_indices[j]
        reg = super_regions[sr_index][..., reg_index]
        combined_regions[slices[j], index_combined] = reg
    index_combined += 1
print("index combined,", index_combined)

print(combined_regions.shape)

image_flatten = []
super_mzs = []
super_spectra = []
super_shapes = []
super_coordinates = []
for name in input_name:
    spectra, mzs, shape, coords = read_imzml(name, normalization)
    super_spectra.append(spectra)
    super_mzs.append(mzs)
    super_shapes.append(shape)
    super_coordinates.append(coords)

spectra, mzs, shape, coords = read_imzml(target_name, normalization)
super_mzs.append(mzs)

mzs, indices = extract_common_mzs(super_mzs)

image_flatten = []
for i, spectra in enumerate(super_spectra):
    im = normalize_flatten(spectra, super_coordinates[i], super_shapes[i], normalization=normalization)
    im = im[..., indices[i]]
    image_flatten.append(im)

target_im = normalize_flatten(spectra, coords, shape, normalization=normalization)
target_im = target_im[..., indices[i]]

image_flatten = np.concatenate(image_flatten)

print(mzs.shape)
print(image_flatten.shape)
print(target_im.shape)

regression = CCA(n_components=combined_regions.shape[-1], scale=True).fit(image_flatten, combined_regions)

out = regression.predict(image_flatten)
coef = regression.coef_

# names = [os.path.splitext(os.path.basename(r))[0] for r in region_names[0]]
# out_spectra = np.vstack((mzs, coef.T))
# np.savetxt(outname, out_spectra.T, delimiter=",", header="mzs,"+ ",".join(names))

print(out.shape)
for i in range(regression.coef_.shape[-1]):
    print(unique_names[i])
    score = out[..., i]
    current_region = combined_regions[..., i]
    fig, ax = plt.subplots(2, max(len(super_regions), 2))
    for j, r in enumerate(super_regions):
        s = unique_slices[j]
        im_region = np.reshape(current_region[s], super_shapes[j][:-1])
        im_score = np.reshape(score[s], super_shapes[j][:-1])
        ax[0, j].imshow(im_region)
        ax[1, j].imshow(im_score)
    plt.figure()
    plt.plot(mzs, coef[..., i])
    plt.show()
