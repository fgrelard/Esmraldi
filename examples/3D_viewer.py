import argparse
import functools
import numpy as np
import SimpleITK as sitk
import esmraldi.segmentation as seg


import matplotlib.pyplot as plt

from vedo import *
import vedo.applications as applications

showing_mesh = False

def keyfunc(key):
    global showing_mesh, vol, vp
    printc('keyfunc called, pressed key:', key)
    if key=='x':
        showing_mesh = not showing_mesh
        if showing_mesh:
            vp.add(vol)
        else:
            vp.remove(vol)

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
vol.interpolation(1)
vp = applications.Slicer(vol, cmaps=('jet', 'gray'),showIcon=False, useSlider3D=True)

vp.keyPressFunction = keyfunc

vp2 = applications.RayCaster(vol)
vp2.sliders[0][0].SetEnabled(True)
vp2.sliders[1][0].SetEnabled(True)
vp2.sliders[2][0].SetEnabled(True)
vp.remove(vol)

vp.show()

# vp.show(vol,viewup="z", interactive=True)


# vp2.show(vol, viewup="z", interactive=True)

# # show lego blocks whose value is between vmin and vmax
# lego = vol.legosurface(vmin=60, cmap='viridis')

# # make colormap start at 40
# lego.addScalarBar(horizontal=True, c='k')

# show(vol, __doc__, axes=1, viewup='z')
