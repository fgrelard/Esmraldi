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
import bisect

import vedo

import sys
import vtk

from PyQt5 import Qt
from PyQt5 import QtCore, QtGui, QtWidgets


from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import random

import scipy.spatial.distance as distance

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class MainWindow(Qt.QMainWindow):

    def __init__(self, inputname, vol, mz, mean_spectra, parent = None):
        Qt.QMainWindow.__init__(self, parent)
        self.setWindowTitle(os.path.basename(inputname))

        self.locale = Qt.QLocale(Qt.QLocale.English)
        Qt.QLocale.setDefault(self.locale)


        self.mz = mz
        self.mean_spectra = mean_spectra
        self.spectrum = np.array([self.mz, self.mean_spectra])
        self.spectrum = (self.spectrum.T / [self.spectrum[0].max(), self.spectrum[1].max()]).T
        self.current_mz = self.mz.min()
        self.current_mz_denom = -1
        self.current_intensity = 0
        self.is_text_editing = False

        self.frame = Qt.QFrame()
        self.vl = Qt.QGridLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vtkWidget.keyPressEvent = self.newKeyPressEvent
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


        self.tol = 0.2

        # a figure instance to plot on
        self.figure = plt.figure(figsize=(10,2))

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.hboxLayout = Qt.QHBoxLayout()
        self.hboxLayout.setSizeConstraint(Qt.QLayout.SetMinAndMaxSize)

        self.label_mz = QtWidgets.QLabel()
        self.label_mz.setText("m/z")

        self.edit_mz = QtWidgets.QLineEdit()
        self.edit_mz.setValidator(QtGui.QDoubleValidator(self.mz.min(), self.mz.max(), 4))
        self.edit_mz.setText(str(round(self.current_mz, 4)))
        self.edit_mz.setMaximumWidth(100)

        self.edit_mz_denom = QtWidgets.QLineEdit()
        self.edit_mz_denom.setValidator(QtGui.QDoubleValidator(-1, self.mz.max(), 4))
        self.edit_mz_denom.setText(str(-1))
        self.edit_mz_denom.setMaximumWidth(100)

        self.edit_mz.textEdited.connect(self.changeMzValue)
        self.edit_mz.returnPressed.connect(self.updateMzValue)

        self.edit_mz_denom.textEdited.connect(lambda text:self.changeMzValue(text, True))
        self.edit_mz_denom.returnPressed.connect(self.updateMzValue)

        self.label_tol = QtWidgets.QLabel()
        self.label_tol.setText("+/-")

        self.edit_tol = QtWidgets.QLineEdit(str(self.tol))
        self.edit_tol.setValidator(QtGui.QDoubleValidator())
        self.edit_tol.setMaximumWidth(70)
        self.edit_tol.textEdited.connect(self.changeToleranceValue)

        # Just some button connected to `plot` method
        self.plot()
        self.figure.tight_layout()

        self.hboxLayout.addStretch()
        self.hboxLayout.addWidget(self.label_mz)
        self.hboxLayout.addWidget(self.edit_mz)
        self.hboxLayout.addWidget(self.edit_mz_denom)
        self.hboxLayout.addWidget(self.label_tol)
        self.hboxLayout.addWidget(self.edit_tol)

        self.vl.addLayout(self.hboxLayout, 3, 0)
        self.vl.addWidget(self.canvas, 4, 0)
        self.vl.addWidget(self.toolbar, 5, 0)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        pos = self.vp.camera.GetPosition()
        self.vp.camera.Azimuth(180)
        self.vp.show(interactive=0, interactorStyle=0, camera={"viewup":(0, -1, 0)})

        self.show()
        self.iren.Initialize()
        self.iren.Start()


    def newKeyPressEvent(self, event):
        if event.key() != 0:
            QVTKRenderWindowInteractor.keyPressEvent(self.vtkWidget, event)


    def changeMzValue(self, text, denominator=False):
        self.is_text_editing = True
        number, is_converted = self.locale.toDouble(text)
        if is_converted:
            if denominator:
                self.current_mz_denom = number
            else:
                self.current_mz = number
            self.get_points_on_spectrum()

    def updateMzValue(self):
        self.is_text_editing = False
        self.get_points_on_spectrum()


    def changeToleranceValue(self, text):
        self.is_text_editing = True
        number, is_converted = self.locale.toDouble(text)
        if is_converted:
            self.tol = number
            self.get_points_on_spectrum()


    def mz_to_indices(self, denom=False):
        current_mz = self.current_mz
        point = self.point_red
        rect = self.rect_red
        if denom:
            current_mz = self.current_mz_denom
            point = self.point_blue
            rect = self.rect_blue

        x1 = current_mz - self.tol/2
        x2 = x1 + self.tol

        y1 = 0
        y2 = np.amax(self.mean_spectra) + 0.1

        mask = (self.mz > min(x1,x2)) & (self.mz < max(x1,x2)) & \
               (self.mean_spectra > min(y1,y2)) & (self.mean_spectra < max(y1,y2))
        no_intersection = not mask.any()
        if no_intersection:
            mask_index = min(bisect.bisect_left(self.mz, current_mz), len(self.mz)-1)
            if mask_index > 0 and \
               abs(self.mz[mask_index-1]-current_mz) < \
               abs(self.mz[mask_index]-current_mz):
                mask_index = mask_index-1
            if mask_index < len(self.mz) - 1 and \
               abs(self.mz[mask_index+1]-current_mz) < \
               abs(self.mz[mask_index]-current_mz):
                mask_index = mask_index+1
            mask[mask_index] = True

        xmasked = self.mz[mask]
        ymasked = self.mean_spectra[mask]
        indices = np.argwhere(mask == True)


        if len(xmasked) > 0:
            xmax = xmasked
            ymax = ymasked
            point.set_data([xmax],[ymax])
            rect.set_width(x2 - x1)
            rect.set_height(y2 - y1)

            if no_intersection:
                current_mz = np.median(xmax)
                rect.set_xy((current_mz-self.tol/2, 0))
            else:
                rect.set_xy((x1, 0))

            if denom:
                self.current_mz_denom = current_mz
            else:
                self.current_mz = current_mz

            self.figure.canvas.draw_idle()

        return indices

    def get_points_on_spectrum(self):
        indices = self.mz_to_indices()
        indices_denom = self.mz_to_indices(True)
        if indices.any():
            num = np.mean(image[..., indices.flatten()], axis=-1)
            if indices_denom.any():
                denom = np.mean(image[..., indices_denom.flatten()], axis=-1)
                divided = np.zeros_like(num, dtype=np.float64)
                np.divide(num, denom, out=divided, where=denom!=0)
                num = divided
            vol = vedo.Volume(num)
            vol.spacing([1, 1, spacing])
            vol.interpolation(1)
            self.vp.update(vol)

        if not self.is_text_editing:
            self.edit_mz.setText(str(round(self.current_mz, 4)))
            self.edit_mz_denom.setText(str(round(self.current_mz_denom, 4)))



    def plot(self):
        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        self.ax = self.figure.add_subplot(111)
        # discards the old graph
        # ax.hold(False) # deprecated, see above

        # plot data
        self.ax.plot(self.mz, self.mean_spectra, "#344E5C")
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
        self.point_red, = self.ax.plot([],[], marker="o", color="crimson")
        self.point_blue, = self.ax.plot([],[], marker="o", color="blue")


        def line_select_callback(event):
            if self.toolbar._actions['zoom'].isChecked() or self.toolbar._actions['pan'].isChecked() or not event.xdata:
                return
            self.is_text_editing = False
            self.current_intensity = event.ydata
            if self.iren.GetControlKey():
                self.current_mz_denom = event.xdata
                self.edit_mz_denom.setText(str(round(self.current_mz_denom, 4)))
            else:
                self.current_mz = event.xdata
                self.edit_mz.setText(str(round(self.current_mz, 4)))
            self.get_points_on_spectrum()


        def arrow_callback(iren, event):
            key = iren.GetKeySym()
            current_index = bisect.bisect_left(self.mz, self.current_mz)
            if key != 'Right' and key != 'Left':
                return
            if key == 'Right':
                new_index = current_index + 1 if current_index < len(self.mz) - 1 else 0
            elif key == 'Left':
                new_index = current_index - 1 if current_index > 0 else len(self.mz) - 1

            self.current_mz = self.mz[new_index]
            self.edit_mz.setText(str(round(self.current_mz, 4)))
            self.get_points_on_spectrum()

        self.rect_red = Rectangle((0,0), 0, 0, alpha=0.1, fc='r')
        self.rect_blue = Rectangle((0,0), 0, 0, alpha=0.1, fc='b')
        self.ax.add_patch(self.rect_red)
        self.ax.add_patch(self.rect_blue)
        self.ax.figure.canvas.mpl_connect('button_press_event', line_select_callback)
        self.vp.interactor.AddObserver("KeyPressEvent", arrow_callback)

        # refresh canvas
        self.canvas.draw()


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input 3D ITK image or imzML file")
parser.add_argument("-s", "--spacing", help="Space between each slice", default=1)
parser.add_argument("--memmap", help="Create and read a memmap file", action="store_true")

args = parser.parse_args()

inputname = args.input
spacing = int(args.spacing)
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
        spectra = imzmlio.get_full_spectra(imzml)
        max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
        max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
        max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
        image = imzmlio.get_images_from_spectra(spectra, (max_x, max_y, max_z))
        print(image.shape)
        if len(image.shape) == 3:
            shape_4D = image.shape[:-1] +  (1, image.shape[-1])
            image = np.reshape(image, shape_4D)
        image = np.transpose(image, (2,1,0,3))

        if is_memmap:
            os.makedirs(memmap_dir, exist_ok=True)
            np.save(memmap_image_filename, image)
            np.save(memmap_spectra_filename, spectra)
    vol = vedo.Volume(image[..., 0])
    mean_spectra = sp.spectra_mean(spectra)
else:
    vol = vedo.load(inputname) # load Volume

vedo.printHistogram(vol, logscale=True)

vol.spacing([1, 1, spacing])
vol.mode(0).color("jet").jittering(True)
vol.interpolation(1)


app = Qt.QApplication(sys.argv)
window = MainWindow(inputname, vol, spectra[0, 0], mean_spectra)
sys.exit(app.exec_())
