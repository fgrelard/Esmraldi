import argparse
import numpy as np
import vtk
import SimpleITK as sitk
import esmraldi.segmentation as seg

from mayavi import mlab

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


from vtk.util.vtkConstants import *

# from vedo import *


def numpy2VTK(img,spacing=[1.0,1.0,1.0]):
    # evolved from code from Stou S.,
    # on http://www.siafoo.net/snippet/314
    importer = vtk.vtkImageImport()

    img_data = img.astype('uint8')
    img_string = img_data.tostring() # type short
    dim = img.shape

    importer.CopyImportVoidPointer(img_string, len(img_string))
    importer.SetDataScalarType(VTK_UNSIGNED_CHAR)
    importer.SetNumberOfScalarComponents(1)

    extent = importer.GetDataExtent()
    importer.SetDataExtent(extent[0], extent[0] + dim[2] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[0] - 1)
    importer.SetWholeExtent(extent[0], extent[0] + dim[2] - 1,
                            extent[2], extent[2] + dim[1] - 1,
                            extent[4], extent[4] + dim[0] - 1)

    importer.SetDataSpacing( spacing[0], spacing[1], spacing[2])
    importer.SetDataOrigin( 0,0,0 )

    return importer

def volumeRender(img, tf=[],spacing=[1.0,1.0,1.0]):
    importer = numpy2VTK(img,spacing)

    # Transfer Functions
    opacity_tf = vtk.vtkPiecewiseFunction()
    color_tf = vtk.vtkColorTransferFunction()

    if len(tf) == 0:
        tf.append([img.min(),0,0,0,0])
        tf.append([img.max(),1,1,1,1])

    for p in tf:
        color_tf.AddRGBPoint(p[0], p[1], p[2], p[3])
        opacity_tf.AddPoint(p[0], p[4])

    # working on the GPU
    # volMapper = vtk.vtkGPUVolumeRayCastMapper()
    # volMapper.SetInputConnection(importer.GetOutputPort())

    # # The property describes how the data will look
    # volProperty =  vtk.vtkVolumeProperty()
    # volProperty.SetColor(color_tf)
    # volProperty.SetScalarOpacity(opacity_tf)
    # volProperty.ShadeOn()
    # volProperty.SetInterpolationTypeToLinear()

    # working on the CPU
    volMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    # compositeFunction = vtk.vtkFixedPointVolumeRayCastCompositeFunction()
    # compositeFunction.SetCompositeMethodToInterpolateFirst()
    # volMapper.SetVolumeRayCastFunction(compositeFunction)
    volMapper.SetInputConnection(importer.GetOutputPort())

    # The property describes how the data will look
    volProperty =  vtk.vtkVolumeProperty()
    volProperty.SetColor(color_tf)
    volProperty.SetScalarOpacity(opacity_tf)
    # volProperty.ShadeOn()
    # volProperty.SetInterpolationTypeToLinear()

    # Do the lines below speed things up?
    # pix_diag = 5.0
    # volMapper.SetSampleDistance(pix_diag / 5.0)
    # volProperty.SetScalarOpacityUnitDistance(pix_diag)


    vol = vtk.vtkVolume()
    vol.SetMapper(volMapper)
    vol.SetProperty(volProperty)

    return [vol]


def vtk_basic( actors ):
    """
    Create a window, renderer, interactor, add the actors and start the thing

    Parameters
    ----------
    actors :  list of vtkActors

    Returns
    -------
    nothing
    """

    # create a rendering window and renderer
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600,600)
    ren.SetBackground(colors.GetColor3d("MistyRose"))
    # ren.SetBackground( 1, 1, 1)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    for a in actors:
        # assign actor to the renderer
        ren.AddActor(a )

    # render
    renWin.Render()

    # enable user interface interactor
    iren.Initialize()
    iren.Start()


def getColorsFromColormap(name, nb=10):
    gradient = cm.get_cmap(name)
    L = []
    for i in range(nb):
        normI = float(i/(nb-1))
        elemColor = [float(i)/nb*255] + [elem for elem in gradient(normI)]
        elemColor[4] = float(i)/nb*0.5
        L.append(elemColor)
    return L


#####


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input 3D ITK image")
args = parser.parse_args()

inputname = args.input


image = sitk.ReadImage(inputname, sitk.sitkFloat32)
image.SetSpacing([1, 1, 80])
image_array = sitk.GetArrayFromImage(image)
shape = image.GetSize()
new_shape = (shape[0], shape[1], shape[2]*80)
image_resize = seg.resize(image, new_shape)
image_resize_array = sitk.GetArrayFromImage(image_resize)

from scipy.stats.mstats import mquantiles
q = mquantiles(image_resize_array.flatten(),[0.7,0.98])
q[0]=max(q[0],1)
q[1] = max(q[1],1)

gray_level=[[0,0,0,0,0],[q[0],0,0,0,0],[q[1],1,1,1,0.5],[image_resize_array.max(),1,1,1,1]]
jet = getColorsFromColormap("jet")

actor_list = volumeRender(image_array, tf=jet, spacing=[1,1,10])

vtk_basic(actor_list)
