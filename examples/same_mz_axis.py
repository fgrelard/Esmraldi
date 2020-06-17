import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input spectrum")
args = parser.parse_args()



input_name = args.input

imzml = io.open_imzml(input_name)
spectra = io.get_spectra(imzml)
spectra = sp.same_mz_axis(spectra, 1)

print(spectra.shape)
