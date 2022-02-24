import argparse
import numpy as np
import esmraldi.imzmlio as io
import SimpleITK as sitk
import matplotlib.pyplot as plt
import xlsxwriter

from skimage.color import rgb2gray

def get_norm_image(images, norm, mzs):
    if norm == "tic":
        img_norm = np.sum(images, axis=-1)
    else:
        closest_mz_index = np.abs(mzs - norm).argmin()
        img_norm = images[..., closest_mz_index]
    return img_norm

def process_image(current_img, img_norm):
    return_img = np.zeros_like(current_img)
    np.divide(current_img, img_norm, out=return_img, where=img_norm!=0)
    return return_img

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-m", "--mask", help="Mask image (any ITK format)")
parser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()

input_name = args.input
mask_name = args.mask
output_name = args.output

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
mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_name))
mask = rgb2gray(mask)
mask = mask.T

n = len(np.where(mask>0)[0])

print(n)
indices = np.where(mask > 0)
workbook = xlsxwriter.Workbook(output_name, {'strings_to_urls': False})
header_format = workbook.add_format({'bold': True,
                                     'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D7E4BC',
                                     'border': 1})

left_format = workbook.add_format({'align': 'left'})

worksheet = workbook.add_worksheet("No norm")
worksheet_stats = workbook.add_worksheet("Stats")
worksheet_stats.write_column(0, 0, ["m/z", "Mean", "Stddev", "N"])

worksheet.freeze_panes(1, 0)
worksheet_stats.freeze_panes(1, 1)

worksheets = []
worksheets.append((worksheet, worksheet_stats))

normalization = [828.7287, 812.7540, 790.773, "tic"]
normalization_images = []
for norm in normalization:
    worksheet = workbook.add_worksheet(str(norm))
    worksheet_stats = workbook.add_worksheet("Stats " + str(norm))
    worksheet_stats.write_column(0, 0, ["m/z", "Mean", "Stddev", "N"])

    worksheet.freeze_panes(1, 0)
    worksheet_stats.freeze_panes(1, 1)
    worksheets.append((worksheet, worksheet_stats))

    norm_img = get_norm_image(images, norm, mzs)
    normalization_images.append(norm_img)

closest_mz_indices = [np.abs(mzs - norm).argmin() for norm in normalization[:-1:]]
print(closest_mz_indices)

for i in range(images.shape[-1]):
    mz = mzs[i]
    current_image = images[..., i]

    for j, (worksheet, worksheet_stats) in enumerate(worksheets):
        if j > 0:
            current_image = process_image(current_image, normalization_images[j-1])
            if j-1 < len(closest_mz_indices) and i == closest_mz_indices[j-1]:
                print(mz, normalization[j-1])
                plt.imshow(current_image)
                plt.show()
                current_image_copy = np.zeros_like(current_image)
                current_image_copy[indices] = 1000
                plt.imshow(current_image_copy)
                plt.show()
        sub_region = current_image[indices]

        current_values = sub_region.flatten()

        mean = np.mean(sub_region)
        stddev = np.std(sub_region)

        worksheet.write(0, i, mz, header_format)
        worksheet.write_column(1, i, current_values)

        worksheet_stats.write(0, i+1, mz, header_format)
        worksheet_stats.write_column(1, i+1, [mean, stddev, n])

workbook.close()
