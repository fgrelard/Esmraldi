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
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
import skimage.morphology as morphology
from sklearn.linear_model import Lasso
import joblib
import gc

def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    if mask.ndim > 2:
        mask = rgb2gray(mask)
    mask = mask.T
    return mask

def indices_false_prediction(prediction, combined_regions, image_flatten):
    indices = []
    for j in range(combined_regions.shape[-1]):
        prediction_sorted = np.argsort(prediction, axis=-1)
        labels = prediction_sorted[:, -1]
        labels_secondmost = prediction_sorted[:, -2]
        idx = np.where((labels != j) & (labels_secondmost != j)
                       & (combined_regions[..., j] > 0))[0]
        combined_regions[idx, j] = 0
        # image_flatten[idx, j] = 0
        indices = np.concatenate((indices, idx))
    return np.unique(indices).astype(int)

def compare_prediction(prediction, combined_regions, sampled_coords, region_shapes, slices):
    for i, current_slice in enumerate(slices):
        for j in range(combined_regions.shape[-1]):
            im = np.zeros(np.prod(region_shapes[i]))
            idx = np.where(combined_regions[current_slice, j] > 0)[0]
            coords = np.array(sampled_coords[i])[idx]
            im[coords] = j+1
            im = np.reshape(im, region_shapes[i])


            labels = np.argsort(prediction[current_slice, :], axis=-1)[:, -1]
            labels_secondmost = np.argsort(prediction[current_slice, :], axis=-1)[:, -2]
            idx = np.where((labels != j) & (labels_secondmost != j) & (combined_regions[current_slice, j] > 0))[0]
            coords_pred = np.array(sampled_coords[i])[idx]
            im_pred = np.zeros(np.prod(region_shapes[i]))
            im_pred[coords_pred] = 1

            idx = np.where((labels == j) | (labels_secondmost == j))[0]
            coords_pred2 = np.array(sampled_coords[i])[idx]
            im_pred[coords_pred2] = 2
            im_pred = np.reshape(im_pred, region_shapes[i])
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(im_pred)
            ax[1].imshow(im)
            plt.show()


def check_convergence(numbers, total, eps):
    previous_diff = 0
    if len(numbers) > 1:
        first = numbers[-2]
        second = numbers[-1]
        diff = first - second
        if diff < total * eps:
            return True
    return False


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str)
parser.add_argument("--lasso", help="Use LASSO", action="store_true")
parser.add_argument("--nb_component", help="Number of components for PLS", type=int, default=10)
parser.add_argument("--alpha", help="Alpha parameter for LASSO", type=float, default=1.0)
parser.add_argument("-o", "--output", help="Output files")
args = parser.parse_args()

input_name = args.input
region_names = args.regions
is_lasso = args.lasso
nb_components = args.nb_component
alpha = args.alpha
outname = args.output

unique_names = [os.path.splitext(os.path.basename(name))[0] for name in region_names]
regions = []
for region_name in region_names:
    region = read_image(region_name)
    regions.append(region)

combined_regions = np.concatenate(regions).T
print(combined_regions.shape)

in_dir = os.path.splitext(input_name)[0]
name_dir = os.path.splitext(outname)[0]

# sampled_coords = np.load(in_dir + "_coords.npy", allow_pickle=True)
# slices = np.load(in_dir + "_slices.npy", allow_pickle=True)
# region_shapes = np.load(in_dir + "_shapes.npy", allow_pickle=True)

if input_name[0].lower().endswith(".imzml"):
    imzml = io.open_imzml(input_name)
    spectra = io.get_spectra(imzml)
    print(spectra.shape)
    coordinates = imzml.coordinates
    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]
    shape = (max_x, max_y, max_z)

    full_spectra = io.get_full_spectra(imzml)
    mzs = np.unique(np.hstack(spectra[:, 0]))
    mzs = mzs[mzs>0]
    print(len(mzs))
    images = io.get_images_from_spectra(full_spectra, shape)
else:
    image_itk = sitk.ReadImage(input_name)
    images = sitk.GetArrayFromImage(image_itk).T
    mzs = np.loadtxt(os.path.splitext(input_name)[0] + ".csv")

image_flatten = images.reshape(images.shape[1:])


print(mzs.shape)
print(image_flatten.shape)


# scaler = StandardScaler()
# image_flatten = scaler.fit_transform(image_flatten)
lengths = []
has_converged = False
while not has_converged:
    if is_lasso:
        regression = Lasso(alpha=alpha, tol=1e-3, selection="random", max_iter=5000).fit(image_flatten, combined_regions)
    else:
        regression = PLSRegression(n_components=nb_components, scale=False, max_iter=5000).fit(image_flatten, combined_regions)
    has_converged = True
    # out = regression.predict(image_flatten)
    # # compare_prediction(out, combined_regions, sampled_coords, region_shapes, slices)
    # indices = indices_false_prediction(out, combined_regions, image_flatten)
    # lengths.append(indices.size)
    # has_converged = check_convergence(lengths, image_flatten.shape[0], 1e-3)

print(lengths)
coef = regression.coef_
# out = regression.predict(image_flatten)

# sitk.WriteImage(sitk.GetImageFromArray(combined_regions.T), name_dir + "_stripped.tif")

#Outputs
prefix_name = name_dir + os.path.sep
os.makedirs(name_dir, exist_ok=True)

joblib.dump(regression, prefix_name + os.path.basename(outname))
np.savetxt(prefix_name + os.path.splitext(os.path.basename(outname))[0] + "_mzs.csv", mzs, delimiter=",")
np.savetxt(prefix_name + os.path.splitext(os.path.basename(outname))[0] + "_names.csv", unique_names, delimiter=",", fmt="%s")
# np.savetxt(prefix_name + os.path.splitext(os.path.basename(outname))[0] + "_train.csv", out, delimiter=",")
np.savetxt(prefix_name + os.path.splitext(os.path.basename(outname))[0] + "_y.csv", combined_regions, delimiter=",")
