import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
args = parser.parse_args()

input_name = args.input

if input_name.lower().endswith(".imzml"):
    imzml = io.open_imzml(input_name)
    spectra = io.get_spectra(imzml)
    print(spectra.shape)
    coordinates = imzml.coordinates
    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]

    full_spectra = io.get_full_spectra(imzml)
    mzs = np.unique(np.hstack(spectra[:, 0]))
    mzs = mzs[mzs>0]
    print(len(mzs))
    image = io.get_images_from_spectra(full_spectra, (max_x, max_y, max_z))
else:
    image_itk = sitk.ReadImage(input_name)
    image = sitk.GetArrayFromImage(image_itk).T
    mzs = np.loadtxt(os.path.splitext(input_name)[0] + ".csv")
