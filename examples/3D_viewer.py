import argparse
import os
import functools
import numpy as np
import SimpleITK as sitk
import esmraldi.segmentation as seg
import esmraldi.imzmlio as imzmlio
import esmraldi.spectraprocessing as sp
import esmraldi.viewer3D as viewer3D
import matplotlib.pyplot as plt

import vedo.applications as applications

import vedo

def tracefunc(frame, event, arg, indent=[0]):
      if event == "call":
          indent[0] += 2
          print("-" * indent[0] + "> call function", frame.f_code.co_name)
      elif event == "return":
          print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
          indent[0] -= 2
      return tracefunc

import sys
# sys.setprofile(tracefunc)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input 3D ITK image or imzML file")
parser.add_argument("--memmap", help="Create and read a memmap file", action="store_true")
args = parser.parse_args()

inputname = args.input
is_memmap = args.memmap


if inputname.endswith(".imzML"):
    memmap_dir = os.path.dirname(inputname) + os.path.sep + "mmap" + os.path.sep
    memmap_basename = os.path.splitext(os.path.basename(inputname))[0]
    memmap_image_filename = memmap_dir + memmap_basename + ".npy"
    memmap_spectra_filename = memmap_dir + memmap_basename + "_spectra.npy"
    memmap_files_exist = (os.path.exists(memmap_dir)
                          and os.path.exists(memmap_image_filename)
                          and os.path.exists(memmap_spectra_filename))

    if is_memmap and  memmap_files_exist:
        print("Reading from memmap")
        spectra = np.load(memmap_spectra_filename, mmap_mode="r")
        image = np.load(memmap_image_filename, mmap_mode="r")
    else:
        imzml = imzmlio.open_imzml(inputname)
        mz, I = imzml.getspectrum(0)
        spectra = imzmlio.get_spectra(imzml)
        image = imzmlio.to_image_array_3D(imzml)
        image = np.transpose(image, (2,1,0,3))

        if is_memmap:
            os.makedirs(memmap_dir, exist_ok=True)
            np.save(memmap_image_filename, image)
            np.save(memmap_spectra_filename, spectra)

    vol = vedo.Volume(image[..., 1000])
    mean_spectra = sp.spectra_mean(spectra)
else:
    vol = vedo.load(inputname) # load Volume

vedo.printHistogram(vol, logscale=True)

print(image.shape)
print(vol.dimensions())

sp = vol.spacing()
vol.spacing([sp[0]*1, sp[1]*1, sp[2]*1])
vol.mode(0).color("jet").jittering(True)
vol.interpolation(1)


vp = viewer3D.Slicer(vol, spectra[0,0], mean_spectra, cmaps=('jet', 'gray'),showIcon=False, showHisto=False, useSlider3D=True)

# mean_spectrum_plot.GetPosition2Coordinate().SetValue(0.9, 0.2, 0)
# mean_spectrum_plot.AddObserver("LeftButtonPressEvent", clickfunc)
# mean_spectrum_plot.PickableOn()
# mean_spectrum_plot.PlotPointsOff()


# vedo.closePlotter()
# vedo.show(vol, mean_spectrum_plot,shape="1/1")
