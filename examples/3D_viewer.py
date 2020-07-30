import argparse
import numpy as np
import vtk
import SimpleITK as sitk
import esmraldi.segmentation as seg


import matplotlib.pyplot as plt

from vedo import *
import vedo.applications as applications


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input 3D ITK image")
args = parser.parse_args()

inputname = args.input

vol = load(inputname) # load Volume

image = sitk.ReadImage(inputname, sitk.sitkFloat32)
image.SetSpacing([1, 1, 10])
image_array = sitk.GetArrayFromImage(image)
shape = image.GetSize()
new_shape = (shape[0], shape[1], shape[2]*80)
image_resize = seg.resize(image, new_shape)
image_resize_array = sitk.GetArrayFromImage(image_resize)

# vol = Volume(image_array)

printHistogram(vol, logscale=True)


sp = vol.spacing()
vol.spacing([sp[0]*1, sp[1]*1, sp[2]*10])
vol.mode(0).color("jet").jittering(True)
vp = applications.RayCaster(vol)
vp.show(viewup="z", interactive=True)
vp.sliders[0][0].SetEnabled(False)
vp.sliders[1][0].SetEnabled(False)
vp.sliders[2][0].SetEnabled(False)



# # show lego blocks whose value is between vmin and vmax
# lego = vol.legosurface(vmin=60, cmap='viridis')

# # make colormap start at 40
# lego.addScalarBar(horizontal=True, c='k')

# show(vol, __doc__, axes=1, viewup='z')
