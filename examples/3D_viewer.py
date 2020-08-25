import vtk.qt
vtk.qt.QVTKRWIBase ="QGLWidget"

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

import sys
import vtk
from PyQt5 import QtCore, QtGui
from PyQt5 import Qt

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import random

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class MainWindow(Qt.QMainWindow):

    def __init__(self, vol, mz, mean_spectra, parent = None):
        Qt.QMainWindow.__init__(self, parent)

        self.mz = mz
        self.mean_spectra = mean_spectra
        self.frame = Qt.QFrame()
        self.vl = Qt.QGridLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget, 0, 0, 3, 7)

        self.vp = viewer3D.Slicer(vol, self.vtkWidget, cmaps=('jet', 'gray'),useSlider3D=False)
        # self.ren = vtk.vtkRenderer()
        self.ren= self.vp.renderer
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        # self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.vtkWidget.GetRenderWindow().SetAlphaBitPlanes(1)
        self.vtkWidget.GetRenderWindow().SetMultiSamples(0)
        self.ren.SetUseDepthPeeling(1)
        self.ren.SetMaximumNumberOfPeels(100)
        self.ren.SetOcclusionRatio(0.1)

        self.iren = self.vp.interactor

        self.ren.ResetCamera()




        # a figure instance to plot on
        self.figure = plt.figure(figsize=(10,2))

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        # self.canvas.setFixedHeight(150)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.plot()
        self.figure.tight_layout()
        # self.figure.canvas.mpl_connect('button_press_event', self.on_click)



        self.vl.addWidget(self.canvas, 3, 0)
        self.vl.addWidget(self.toolbar, 4, 0)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.vp.show(interactive=0, interactorStyle=0)
        self.show()
        # self.show()
        self.iren.Initialize()
        self.iren.Start()

    def plot(self):
        ''' plot some random stuff '''
        # random data
        data_x = self.mz
        data_y = self.mean_spectra


        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        self.ax = self.figure.add_subplot(111)
        # discards the old graph
        # ax.hold(False) # deprecated, see above

        # plot data
        self.ax.plot(data_x, data_y, "#344E5C")
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        self.ax.yaxis.set_ticks_position('left')
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.set_xlabel("m/z", fontweight="bold")
        self.ax.set_ylabel("I", fontweight="bold")
        self.figure.patch.set_facecolor('#E0E0E0')
        self.figure.patch.set_alpha(0.7)
        self.ax.patch.set_facecolor('#E0E0E0')
        self.ax.patch.set_alpha(0)
        point, = self.ax.plot([],[], marker="o", color="crimson")


        def line_select_callback(event):
            if self.toolbar._actions['zoom'].isChecked() or self.toolbar._actions['pan'].isChecked()or not event.xdata:
                return
            tol = 1
            x1, y1 = event.xdata-tol/2, 0
            x2, y2 = x1+tol, self.ax.get_ylim()[1]
            mask= (data_x > min(x1,x2)) & (data_x < max(x1,x2)) & \
                  (data_y > min(y1,y2)) & (data_y < max(y1,y2))
            xmasked = data_x[mask]
            ymasked = data_y[mask]
            indices = np.argwhere(mask == True)
            print(indices[0][0])
            vol = vedo.Volume(image[..., indices[0,0]])
            self.vp.update(vol)
            if len(xmasked) > 0:
                xmax = xmasked
                ymax = ymasked
                point.set_data([xmax],[ymax])
                self.rect.set_width(x2 - x1)
                self.rect.set_height(y2 - y1)
                self.rect.set_xy((x1, y1))
                self.figure.canvas.draw_idle()


        self.rect = Rectangle((0,0), 0, 0, alpha=0.1, fc='r')
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', line_select_callback)



        # refresh canvas
        self.canvas.draw()





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



sp = vol.spacing()
vol.spacing([sp[0]*1, sp[1]*1, sp[2]*1])
vol.mode(0).color("jet").jittering(True)
vol.interpolation(1)


# vp = viewer3D.Slicer(vol, spectra[0,0], mean_spectra, cmaps=('jet', 'gray'),showIcon=False, showHisto=False, useSlider3D=True)

# vp = applications.Slicer(vol, cmaps=('jet', 'gray'),showIcon=False, showHisto=False, useSlider3D=True)
# vp.show()


app = Qt.QApplication(sys.argv)
window = MainWindow(vol, spectra[0, 0], mean_spectra)
sys.exit(app.exec_())
