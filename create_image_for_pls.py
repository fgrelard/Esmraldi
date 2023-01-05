import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import SimpleITK as sitk
from skimage.color import rgb2gray

import esmraldi.spectraprocessing as sp
import esmraldi.imzmlio as io
import esmraldi.fusion as fusion
import skimage.morphology as morphology
import esmraldi.utils as utils
import os

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
        coordinates = imzml.coordinates
        max_x = max(coordinates, key=lambda item:item[0])[0]
        max_y = max(coordinates, key=lambda item:item[1])[1]
        max_z = max(coordinates, key=lambda item:item[2])[2]
        shape = (max_x, max_y, max_z)
        mzs = np.unique(np.hstack(spectra[:, 0]))
        mzs = mzs[mzs>0]
    return spectra, mzs, shape, imzml.coordinates

def indices_peaks(peaks, other_peaks, full=False):
    indices = utils.indices_search_sorted(other_peaks, peaks)
    current_step = 14 * other_peaks / 1e6
    indices_ppm = np.abs(peaks[indices] - other_peaks) < current_step
    if full:
        indices[~indices_ppm] = -1
    else:
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

def extract_all_mzs_nodup(mzs):
    peaks, mzs_indices = extract_common_mzs(mzs)
    for i, indices in enumerate(mzs_indices):
        current_mzs = mzs[i]
        common_mzs = current_mzs[indices]
        other_mzs = np.setdiff1d(current_mzs, common_mzs)
        peaks = np.append(peaks, other_mzs)
    peaks = np.sort(peaks)
    return peaks

def remove_duplicates_ppm(mzs, step_ppm):
    groups = sp.index_groups(np.array(mzs), step=step_ppm, is_ppm=True)
    # print(groups)
    aligned_mzs = sp.peak_reference_indices_median(groups)
    return np.array(aligned_mzs)



def coordinates_from_sampled_regions(super_regions, super_shapes, size, all_names, balanced_names, balanced_counts, erosion_radius=0):
    super_coords = []
    ind = 0
    for i, regions in enumerate(super_regions):
        current_coords = []
        for j in range(regions.shape[-1]):
            current_name = all_names[ind]
            index_balanced = np.where(balanced_names == current_name)
            count = balanced_counts[index_balanced]
            limit = int(size*count)
            region = regions[..., j].copy()
            im_region = np.reshape(region, super_shapes[i])
            footprint = morphology.disk(erosion_radius)
            im_eroded = morphology.erosion(im_region, footprint)
            if im_eroded[im_eroded > 0].size > size:
                im_region = im_eroded
            im_region = im_region.flatten()
            coords = np.argwhere(im_region > 0)
            np.random.shuffle(coords)
            coords = coords[:limit]
            current_coords += coords.flatten().tolist()
            ind += 1
        super_coords.append(current_coords)
    return super_coords

def balance_counts_per_name(all_names, indices):
    all_names = np.array(all_names)
    balanced_names = []
    balanced_counts = []
    for ind in indices:
        current_names = all_names[ind]
        unique, counts = np.unique(current_names, return_counts=True)
        minc = counts.min()
        counts =  minc / counts
        balanced_names += unique.tolist()
        balanced_counts += counts.tolist()
    return np.array(balanced_names), np.array(balanced_counts)


def sample_image(super_regions, super_coords):
    new_regions = []
    for i, regions in enumerate(super_regions):
        coords = super_coords[i]
        current_regions = []
        for j in range(regions.shape[-1]):
            current_region = regions[..., j]
            current_region = current_region[coords]
            current_regions.append(current_region)
        current_regions = np.transpose(current_regions, (1, 0))
        new_regions.append(current_regions)
    return new_regions

def indices_category_region(names):
    unique, indices = np.unique(names, return_inverse=True)
    indices = [np.where(indices == i)[0].astype(int) for i in range(len(unique))]
    return unique, indices

def normalize_flatten(spectra, coordinates, shape, normalization_tic=True, normalization_minmax=True):
    print("normalization TIC=", normalization_tic)
    print("normalization minmax=", normalization_minmax)
    if normalization_tic:
        spectra = sp.normalization_tic(spectra, inplace=True)
    full_spectra = io.get_full_spectra_dense(spectra, coordinates, shape)
    images = io.get_images_from_spectra(full_spectra, shape)
    if normalization_minmax:
        images = io.normalize(images)
    images = images.astype(np.float128) / 255.0
    image_flatten = fusion.flatten(images, is_spectral=True).T
    return image_flatten

def read_regions(region_names):
    regions = []
    region_shapes = []
    names = []
    for region_name in region_names:
        trimmed_name = os.path.splitext(os.path.basename(region_name))[0]
        region = read_image(region_name)
        regions.append(region)
        if "&" in trimmed_name:
            binders = trimmed_name.split("_")[0]
            pigment = trimmed_name.split("_")[1]
            binder1 = binders.split('&')[0]
            binder2 = binders.split('&')[1]
            names.append(binder1 + "_" + pigment)
            names.append(binder2 + "_" + pigment)
            #Add this region twice
            regions.append(region)
        else:
            names.append(trimmed_name)
    regions = np.transpose(np.array(regions), (1,2,0))
    regions = io.normalize(regions)
    region_flatten = fusion.flatten(regions, is_spectral=True).T
    return region_flatten, region.shape, names

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


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML", nargs="+", type=str)
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str, action="append")
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", action="store_true")
parser.add_argument("--sample_size", help="Sample size (in pixels)", default=0)
parser.add_argument("-o", "--output", help="Output files")
args = parser.parse_args()

input_name = args.input
region_names = args.regions
normalization = args.normalization
sample_size = int(args.sample_size)
outname = args.output



super_regions = []
region_shapes = []
all_names = []
for dataset_regions in region_names:
    regions, reg_shape, names = read_regions(dataset_regions)
    region_shapes.append(reg_shape)
    super_regions.append(regions)
    all_names += names

binders = [n.split("_")[0] for n in all_names]
pigments = [n.split("_")[1] for n in all_names]
unique_pigments, indices_pigments = indices_category_region(pigments)
unique_binders, indices_binders = indices_category_region(binders)

if sample_size > 0:
    balanced_names, balanced_counts = balance_counts_per_name(all_names, indices_binders)
    sampled_coords = coordinates_from_sampled_regions(super_regions, region_shapes, sample_size, all_names, balanced_names, balanced_counts, 1)
    # blup = balance_regions(super_coords, binders, pigments)
    super_regions = sample_image(super_regions, sampled_coords)

shape_super_regions = np.sum([s.shape for s in super_regions], axis=0)

unique_names = np.concatenate((unique_pigments, unique_binders))
indices = indices_pigments + indices_binders

# unique_names = unique_binders
# indices = indices_binders
slices = extract_indexslices_regions(super_regions)
unique_slices = np.unique(slices)
slices_combined, indices_combined = extract_indices_combined_regions(super_regions, indices)


shape_super_regions[-1] = len(unique_names)
print(shape_super_regions)

combined_regions = np.zeros(shape_super_regions)
index_combined = 0

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

root = os.path.splitext(outname)[0]

region_dir = os.path.dirname(outname) + os.path.sep + "regions" + os.path.sep
os.makedirs(region_dir, exist_ok=True)
for i, region_name in enumerate(unique_names):
    current_region = combined_regions[..., i][:, np.newaxis].astype(np.float32) / 255.0
    print(current_region.shape)
    sitk.WriteImage(sitk.GetImageFromArray(current_region), region_dir  + region_name + ".tif")

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


mzs, indices = extract_common_mzs(super_mzs)
mzs = extract_all_mzs_nodup(super_mzs)
print(len(mzs))
mzs = remove_duplicates_ppm(mzs, 14)
print(len(mzs))
indices = []
for i, super_mz in enumerate(super_mzs):
    ind = indices_peaks(super_mzs[i], mzs, full=True)
    indices.append(ind)


image_flatten = []
for i, spectra in enumerate(super_spectra):
    im = normalize_flatten(spectra, super_coordinates[i], super_shapes[i], normalization_tic=normalization, normalization_minmax=True)
    if sample_size > 0:
        coords = sampled_coords[i]
        im = im[coords]
    blank_image = np.zeros((im.shape[0], 1))
    target_im = np.hstack((im, blank_image))
    im = im[..., indices[i]]
    image_flatten.append(im)

image_flatten = np.concatenate(image_flatten).astype(np.float32)
image_flatten = image_flatten.T[..., np.newaxis]
print(image_flatten.shape)

sitk.WriteImage(sitk.GetImageFromArray(image_flatten), outname)
io.to_csv(mzs, root + ".csv")
