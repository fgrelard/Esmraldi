import argparse
import functools
import numpy as np
import SimpleITK as sitk
import esmraldi.segmentation as seg
import esmraldi.imzmlio as imzmlio
import esmraldi.spectraprocessing as sp

import matplotlib.pyplot as plt

import vedo.applications as applications

from vedo import *

showing_mesh = False

def keyfunc(key):
    global showing_mesh, volclone, vp
    printc('keyfunc called, pressed key:', key)
    if key=='x':
        showing_mesh = not showing_mesh
        if showing_mesh:
            vp.add(vol)
        else:
            vp.remove(vol)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input 3D ITK image or imzML file")
args = parser.parse_args()

inputname = args.input



if inputname.endswith(".imzML"):
    imzml = imzmlio.open_imzml(inputname)
    mz, I = imzml.getspectrum(0)
    spectra = imzmlio.get_spectra(imzml)
    image = imzmlio.to_image_array_3D(imzml)
    image = np.transpose(image, (2,1,0,3))
    vol = Volume(image[..., 0])
    mean_spectra = sp.spectra_mean(spectra)
else:
    vol = load(inputname) # load Volume


printHistogram(vol, logscale=True)
print(image.shape)
print(vol.dimensions())
sp = vol.spacing()
vol.spacing([sp[0]*1, sp[1]*1, sp[2]*10])
vol.mode(0).color("jet").jittering(True)
vol.interpolation(1)


# vp2 = applications.RayCaster(vol)
# vp2.sliders[1][0].Off()
# vp2.sliders[2][0].Off()
# vp2.sliders[3][0].Off()

vp = applications.Slicer(vol, cmaps=('jet', 'gray'),showIcon=False, showHisto=False, useSlider3D=True)

vp.keyPressFunction = keyfunc




vp.remove(vol)


mean_spectrum_plot = pyplot.cornerPlot([mz,mean_spectra],
                                       c=(0.0,0.1,0.3),
                                       bg=(0.3,0.3,0.3),
                                       pos=(0.01, 0.05))

mean_spectrum_plot.GetPosition2Coordinate().SetValue(0.9, 0.2, 0)
vp.add(mean_spectrum_plot)

vp.show()
