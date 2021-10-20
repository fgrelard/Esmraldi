import pyqtgraph as pg
import numpy as np
import matplotlib
import os
import time
import inspect

import numbers

import esmraldi.spectraprocessing as sp
from gui.viewboxdirac import ViewBoxDirac

#Allows to use QThreads without freezing
#the main application
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QApplication

import pyqtgraph as pg
from pyqtgraph.functions import affineSlice
from pyqtgraph.graphicsItems.ROI import ROI, CircleROI, PolyLineROI
from pyqtgraph.graphicsItems.GraphicsObject import GraphicsObject
from pyqtgraph.dockarea import DockArea, Dock

from qtrangeslider import QLabeledRangeSlider
from collections import ChainMap




def PaintingROI(parent, roi, brush, **kwargs):
    class WrapROI(parent):
        def __init__(self, roi, brush=None):
            c = ChainMap({"angle":roi.angle(), "invertible":roi.invertible, "maxBounds":roi.maxBounds, "snapSize":roi.snapSize, "scaleSnap":roi.scaleSnap, "translateSnap":roi.translateSnap, "rotateSnap":roi.rotateSnap, "parent":None, "pen":roi.pen, "hoverPen":roi.hoverPen, "handlePen":roi.handlePen, "handleHoverPen":roi.handleHoverPen, "movable":roi.translatable, "rotatable":roi.rotatable, "resizable":roi.resizable, "removable":roi.removable}, **kwargs)
            if parent == PolyLineROI:
                positions = [[handle["pos"].x(), handle["pos"].y()] for handle in roi.handles]
                super().__init__(positions, roi.closed, pos=roi.pos(), **c)
            else:
                super().__init__(pos=roi.pos(), size=roi.size(), **c)
            self.brush =  brush

        def setBrush(self, brush):
            self.brush = brush

        def paint(self, p, opt, widget):
            p.setBrush(self.brush)
            if parent == PolyLineROI:
                p.setPen(self.currentPen)
                p.fillPath(self.shape(), self.brush)
            else:
                super().paint(p, opt, widget)

    return WrapROI(roi, brush)


def addNewGradientFromMatplotlib( name):
    """
    Generic function to add a gradient from a
    matplotlib colormap

    Parameters
    ----------
    name: str
        name of matplotlib colormap
    """
    gradient = cm.get_cmap(name)
    L = []
    nb = 10
    for i in range(nb):
        normI = float(i/(nb-1))
        elemColor = ((normI, tuple(int(elem*255) for elem in gradient(normI))))
        L.append(elemColor)
    pg.graphicsItems.GradientEditorItem.Gradients[name] = {'ticks':L, 'mode': 'rgb'}

def setHistogramRange(obj, mn, mx, padding):
    lmin = float(mn)
    lmax = float(mx)
    obj.vb.enableAutoRange(obj.vb.YAxis, False)
    obj.vb.setYRange(lmin, lmax, padding)

class ImageViewExtended(pg.ImageView):
    """
    Image view extending pyqtgraph.ImageView

    Attributes
    ----------
    label: pg.LabelItem
        Display pixel values and positions
    threads: list
        List of threads
    mouse_x: int
        Mouse position x
    mouse_y: int
        Mouse position y
    """

    signal_abort = QtCore.pyqtSignal()
    signal_progress_export = QtCore.pyqtSignal(int)
    signal_start_export = QtCore.pyqtSignal()
    signal_end_export = QtCore.pyqtSignal()
    signal_image_change = QtCore.pyqtSignal(int)
    signal_mz_change = QtCore.pyqtSignal(float)
    signal_roi_changed = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None, name="ImageView", view=None, imageItem=None, *args):
        pg.setConfigOptions(imageAxisOrder='row-major')
        addNewGradientFromMatplotlib("jet")
        addNewGradientFromMatplotlib("viridis")
        addNewGradientFromMatplotlib("plasma")
        addNewGradientFromMatplotlib("inferno")
        addNewGradientFromMatplotlib("magma")
        addNewGradientFromMatplotlib("cividis")
        grayclip = pg.graphicsItems.GradientEditorItem.Gradients["greyclip"]
        pg.graphicsItems.GradientEditorItem.Gradients["segmentation"] = {'ticks': [(0.0, (0, 0, 0, 255)), (1.0-np.finfo(float).eps, (255, 255, 255, 255)), (1.0, (255, 0, 0, 255))], 'mode': 'rgb'}
        super().__init__(parent, name, view, imageItem, *args)

        self.imageItem.getHistogram = self.getImageItemHistogram
        self.imageItem.mouseClickEvent = self.mouseClickEventImageItem
        self.imageItem.mouseDragEvent = self.mouseClickEventImageItem

        self.timeLine.setPen('g')

        self.ui.histogram.sigLevelsChanged.connect(self.levelsChanged)

        self.ui.histogram.gradient.loadPreset("viridis")
        self.gradient = self.ui.histogram.gradient.colorMap()

        self.ui.histogram.gradient.updateGradient()
        self.ui.histogram.gradientChanged()

        self.ui.labelRoiChange = QtWidgets.QLabel(self.ui.layoutWidget)
        self.ui.labelRoiChange.setObjectName("labelRoiChange")
        self.ui.labelRoiChange.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.ui.labelRoiChange.hide()
        self.ui.gridLayout.addWidget(self.ui.labelRoiChange, 2, 0, 1, 1)


        self.ui.spectraBtn = QtWidgets.QPushButton(self.ui.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.ui.spectraBtn.sizePolicy().hasHeightForWidth())
        # self.ui.spectraBtn.setSizePolicy(sizePolicy)
        self.ui.spectraBtn.setCheckable(True)
        self.ui.spectraBtn.setObjectName("spectraBtn")
        self.ui.gridLayout.addWidget(self.ui.spectraBtn, 2, 2, 1, 1)
        self.ui.spectraBtn.setText(QtCore.QCoreApplication.translate("Form", "Spectra"))
        self.ui.spectraBtn.clicked.connect(self.spectraToggled)
        self.ui.normAutoRadio = QtWidgets.QRadioButton(self.ui.normGroup)
        self.ui.normAutoRadio.setObjectName("normAutoRadio")


        self.hide_partial()

        for i in reversed(range(self.ui.gridLayout_2.count())):
            self.ui.gridLayout_2.itemAt(i).widget().setParent(None)

        self.label = pg.LabelItem(justify='right')
        self.scene.addItem(self.label)
        self.scene.sigMouseMoved.connect(self.on_hover_image)

        self.threads = []

        self.mouse_x = 0
        self.mouse_y = 0

        self.mask_roi = None
        self.coords_roi = None

        self.plot = None
        self.displayed_spectra = None

        self.is_clickable = False
        self.is_drawable = False

        self.pen_size = 1
        self.imageCopy = None
        self.imageItem.drawAt = self.drawAt
        self.imageItem.render = self.render

        self.levelMin, self.levelMax = None, None
        self.isNewImage = False
        self.isNewNorm = False
        self.normDivideRadioChecked = False

        self.ui.histogram.setHistogramRange = lambda mn, mx, padding=0.1: setHistogramRange(self.ui.histogram, mn, mx, padding)

        # self.plot = pg.ScatterPlotItem(size=5, pen=pg.mkPen(255,255, 255, 230), brush=pg.mkBrush(220, 220, 220, 230),hoverable=True,hoverPen=pg.mkPen(242, 38, 19),hoverSize=5,hoverBrush=pg.mkBrush(150, 40, 27))

        vb = ViewBoxDirac()
        self.winPlot = pg.PlotWidget(viewBox=vb, size=(1,1))
        self.plot = pg.BarGraphItem(x=[], height=[], width=0)

        self.clickedPen = pg.mkPen("b")
        self.lastPointsClicked = []

        self.winPlot.setMaximumHeight(100)
        self.winPlot.addItem(self.plot)
        self.plot.getViewBox().mouseDragEvent = self.draggedSpectra

        self.ui.gridLayout_3.addWidget(self.winPlot)

        self.previousRoiSize = 10
        self.previousRoiPositions = [[0,0], [10, 0], [5, 5]]

        self.build_roi_group()

        # self.winPlot.setVisible(False)

    def build_roi_group(self):
        self.ui.roiButtonGroup = QtWidgets.QButtonGroup(self.ui.normGroup)

        self.ui.roiGroup = QtWidgets.QGroupBox(self)
        self.ui.roiGroup.setObjectName("roiGroup")
        self.ui.gridLayout_roi = QtWidgets.QGridLayout(self.ui.roiGroup)
        self.ui.gridLayout_roi.setContentsMargins(0, 0, 0, 0)
        self.ui.gridLayout_roi.setSpacing(0)
        self.ui.gridLayout_roi.setObjectName("gridLayout_roi")

        self.ui.label_roi_type = QtWidgets.QLabel(self.ui.roiGroup)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ui.label_roi_type.setFont(font)
        self.ui.label_roi_type.setObjectName("label_roi_type")
        self.ui.gridLayout_roi.addWidget(self.ui.label_roi_type, 0, 0, 1, 1)
        self.ui.label_roi_type.setText("Type:")

        self.ui.label_roi_range = QtWidgets.QLabel(self.ui.roiGroup)
        self.ui.label_roi_range.setFont(font)
        self.ui.label_roi_range.setObjectName("label_roi_range")
        self.ui.gridLayout_roi.addWidget(self.ui.label_roi_range, 1, 0, 1, 1)
        self.ui.label_roi_range.setText("Range:")


        self.ui.roiSquare = QtWidgets.QRadioButton(self.ui.roiGroup)
        self.ui.roiSquare.setChecked(True)
        self.ui.roiSquare.setObjectName("roiSquare")
        self.ui.gridLayout_roi.addWidget(self.ui.roiSquare, 0, 1, 1, 1)
        self.ui.roiSquare.setText(QtCore.QCoreApplication.translate("Form", "Square"))

        self.ui.roiCircle = QtWidgets.QRadioButton(self.ui.roiGroup)
        self.ui.roiCircle.setObjectName("roiCircle")
        self.ui.gridLayout_roi.addWidget(self.ui.roiCircle, 0, 2, 1, 1)
        self.ui.roiCircle.setText(QtCore.QCoreApplication.translate("Form", "Circle"))

        self.ui.roiPolygon = QtWidgets.QRadioButton(self.ui.roiGroup)
        self.ui.roiPolygon.setObjectName("roiPolygon")
        self.ui.gridLayout_roi.addWidget(self.ui.roiPolygon, 0, 3, 1, 1)
        self.ui.roiPolygon.setText(QtCore.QCoreApplication.translate("Form", "Polygon"))

        self.ui.rangeSliderThreshold = QLabeledRangeSlider(QtCore.Qt.Horizontal)
        self.ui.rangeSliderThreshold.setMinimum(0)
        self.ui.rangeSliderThreshold.setMaximum(100)
        self.ui.rangeSliderThreshold.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.LabelIsValue)
        self.ui.rangeSliderThreshold.setValue((0, 100))
        self.ui.rangeSliderThreshold.valueChanged.connect(self.sliderValueChanged)
        self.ui.gridLayout_roi.addWidget(self.ui.rangeSliderThreshold, 1, 1, 1, 3)

        self.ui.plotSpectraButton = QtWidgets.QPushButton(self.ui.roiGroup)
        self.ui.plotSpectraButton.setObjectName("plotSpectraButton")
        self.ui.plotSpectraButton.setText("Plot spectra")
        self.ui.gridLayout_roi.addWidget(self.ui.plotSpectraButton, 2, 0, 1, 1)

        self.winPlotROI = QtWidgets.QMainWindow()
        self.area = DockArea()
        self.winPlotROI.setCentralWidget(self.area)
        self.winPlotROI.setWindowTitle("Plot ROI")
        self.winPlotROI.resize(600, 480)
        self.winPlotROI.closeEvent = lambda ev: self.winPlotROI.hide

        self.ui.gridLayout_3.addWidget(self.ui.roiGroup)

        self.ui.roiButtonGroup.addButton(self.ui.roiSquare)
        self.ui.roiButtonGroup.addButton(self.ui.roiCircle)
        self.ui.roiButtonGroup.addButton(self.ui.roiPolygon)
        self.ui.roiSquare.clicked.connect(self.roiRadioChanged)
        self.ui.roiCircle.clicked.connect(self.roiRadioChanged)
        self.ui.roiPolygon.clicked.connect(self.roiRadioChanged)
        self.ui.plotSpectraButton.clicked.connect(self.plotSpectraROI)
        self.roiRadioChanged()

    def hide_partial(self):
        """
        Hide some elements from the parent GUI
        """
        # self.ui.roiBtn.hide()
        self.ui.label_4.hide()
        self.ui.label_8.hide()
        self.ui.label_9.hide()
        self.ui.label_10.hide()
        self.ui.normXBlurSpin.hide()
        self.ui.normYBlurSpin.hide()
        self.ui.normTBlurSpin.hide()
        self.ui.normFrameCheck.hide()
        self.ui.normSubtractRadio.hide()
        self.ui.normAutoRadio.hide()


    def render(self):
        pg.ImageItem.render(self.imageItem)
        if not self.ui.roiBtn.isChecked() or self.coords_roi is None or self.imageItem is None or self.imageItem.qimage is None:
            return

        pixel_value = QtGui.qRgb(255, 0, 0)
        point = self.roi.boundingRect().topLeft()

        if self.imageItem.qimage.format() < 4:
            self.imageItem.qimage = self.imageItem.qimage.convertToFormat(QtGui.QImage.Format_RGBA8888)


        for x, y in self.coords_roi.T:
            self.imageItem.qimage.setPixel(x, y, pixel_value)
        self.imageItem._renderRequired = False
        self.imageItem._unrenderable = False


    def mouseClickEventImageItem(self, ev):
        pg.ImageItem.mouseClickEvent(self.imageItem, ev)
        if self.is_drawable:
            pos = ev.pos()
            pos = [int(pos.x()), int(pos.y())]
            if ev.button() == QtCore.Qt.RightButton:
                ev.accept()
                shift = self.pen_size//2
                min_x, max_x = pos[0] - shift, pos[0] + shift
                min_y, max_y = pos[1] - shift, pos[1] + shift
                local_values = self.imageCopy[self.currentIndex, min_y:max_y, min_x:max_x]
                self.update_pen(pen_size=self.pen_size, array=local_values)
            else:
                self.update_pen(pen_size=self.pen_size, array=None)
            self.drawAt(pos, ev)



    def drawAt(self, pos, ev=None):
        order = pg.getConfigOption("imageAxisOrder")
        if order == 'row-major':
            pos = QtCore.QPoint(pos[1], pos[0])
        pg.ImageItem.drawAt(self.imageItem, pos, ev)

    def setDrawable(self, is_drawable, pen_size=1):
        self.is_drawable = is_drawable
        self.updateImage()
        if self.is_drawable:
            self.normDivideRadioChecked = self.ui.normDivideRadio.isChecked()
            self.isNewNorm = False
            self.ui.normOffRadio.setChecked(True)
            self.normalize(self.image)
            self.update_pen(pen_size)
            self.gradient = self.ui.histogram.gradient.colorMap()
            self.ui.histogram.gradient.loadPreset("segmentation")
        else:
            if self.normDivideRadioChecked:
                self.ui.normDivideRadio.setChecked(True)
            self.imageItem.setDrawKernel(kernel=None)
            self.ui.histogram.gradient.setColorMap(self.gradient)

    def update_pen(self, pen_size, array=None):
        self.pen_size = pen_size
        if self.is_drawable:
            if array is None:
                array = np.full((self.pen_size, self.pen_size), self.pen_value)
            self.imageItem.setDrawKernel(kernel=array, center=(self.pen_size//2, self.pen_size//2), mode='set')

    def setClickable(self, is_clickable):
        self.is_clickable = is_clickable



    def setImage(self, img, autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True):
        """
        Sets a new image

        When changing an image, tries to keep the old z-index

        Changes the wheel-event from zoom-out
        to slice change
        """
        #Saves previous z-index
        previousIndex = self.currentIndex
        is_shown = False
        if self.imageDisp is not None:
            previousShape = self.imageDisp.shape
            is_shown = True

        self.isNewImage = True
        super().setImage(img, autoRange, autoLevels, levels, axes, xvals, pos, scale, transform, autoHistogramRange)


        self.buildPlot()

        self.imageCopy = self.imageDisp.copy()
        self.pen_value = np.amax(self.imageDisp)+1

        #Changes wheel event
        self.ui.roiPlot.setMouseEnabled(True, True)
        self.ui.roiPlot.wheelEvent = self.roi_scroll_bar
        max_t = self.imageDisp.shape[0]
        self.normRgn.setRegion((1, max_t//2))
        if not is_shown:
            return
        #Shows image at previous z-index if in range
        if self.imageDisp.ndim > 2 and previousIndex < self.imageDisp.shape[0] and self.axes["t"] is not None:
            self.setCurrentIndex(previousIndex)

    def roi_scroll_bar(self, ev):
        """
        Changes the z-index of the 3D image
        when scrolling the z-bar

        Parameters
        ----------
        ev: QWheelEvent
            the wheel event
        """
        new_index = self.currentIndex + 1 if ev.angleDelta().y() < 0 else self.currentIndex - 1
        self.setCurrentIndex(new_index)


    def on_hover_image(self, evt):
        """
        Updates the mouse positions and pixel values
        when hovering over the image

        Parameters
        ----------
        evt: QMouseEvent
            the mouse event
        """
        pos = evt
        mousePoint = self.view.mapSceneToView(pos)
        self.mouse_x = int(mousePoint.x())
        self.mouse_y = int(mousePoint.y())
        image = self.imageDisp
        if image is None:
            return
        self.update_label()

    def evalKeyState(self):
        if len(self.keysPressed) == 1:
            key = list(self.keysPressed.keys())[0]
            if key == QtCore.Qt.Key_Right:
                self.play(20)
                self.jumpFrames(1)
                self.lastPlayTime = pg.ptime.time() + 0.2
            elif key == QtCore.Qt.Key_Left:
                self.play(-20)
                self.jumpFrames(-1)
                self.lastPlayTime = pg.ptime.time() + 0.2
            elif key == QtCore.Qt.Key_Up:
                self.signal_image_change.emit(-1)
            elif key == QtCore.Qt.Key_Down:
                self.signal_image_change.emit(1)
            elif key == QtCore.Qt.Key_PageUp:
                self.play(-1000)
            elif key == QtCore.Qt.Key_PageDown:
                self.play(1000)
        else:
            self.play(0)



    def update_label(self):
        """
        Updates the label with mouse position
        and pixel values relative to the image
        """
        if not (self.mouse_x >= 0 and self.mouse_x < self.imageItem.image.shape[1] and \
                self.mouse_y >= 0 and self.mouse_y < self.imageItem.image.shape[0]):
            self.mouse_x = 0
            self.mouse_y = 0
        position = "(" + str(self.mouse_x) + ", " + str(self.mouse_y) + ")"
        position_z = ""

        if self.axes["t"] is not None:
            position_z = str(self.tVals[self.currentIndex])

        value = self.imageItem.image[(self.mouse_y, self.mouse_x)]
        if isinstance(value, numbers.Number):
            str_value = "{:.3e}".format(value)
        else:
            str_value = ",".join([str(v) for v in value])
        self.label.setText("<p><span>" + position + "</span><span style='font-weight:bold; color: green;'>: " + str_value + "</span>" + "</p><p><span style='font-weight:bold'>" + position_z + "</span></p>")


    def setCurrentIndex(self, ind):
        super().setCurrentIndex(ind)
        self.signal_mz_change.emit(self.tVals[self.currentIndex])
        self.update_label()

    def timeLineChanged(self):
        super().timeLineChanged()
        self.signal_mz_change.emit(self.tVals[self.currentIndex])
        self.max_thresh = self.imageItem.image.max()
        self.roiChanged()
        self.update_label()

    def updateNorm(self):
        self.isNewNorm = True
        if self.ui.normTimeRangeCheck.isChecked():
            self.normRgn.show()
        else:
            self.normRgn.hide()

        if self.ui.normROICheck.isChecked():
            self.normRoi.show()
        else:
            self.normRoi.hide()

        if not self.ui.normOffRadio.isChecked():
            self.updateImage()
            self.autoLevels()
            self.roiChanged()
            self.sigProcessingChanged.emit(self)

    def getProcessedImage(self):
        if self.isNewImage and self.levelMin is not None:
            self.imageDisp = self.image
        elif self.imageDisp is None or self.isNewNorm:
            self.imageDisp = self.normalize(self.image)
            if self.axes['t'] is not None and self.ui.normOffRadio.isChecked():
                curr_img = self.imageDisp[self.currentIndex, ...]
                self.levelMin, self.levelMax = np.amin(curr_img), np.amax(curr_img)
            else:
                self.levelMin, self.levelMax = np.amin(self.imageDisp), np.amax(self.imageDisp)
        if self.is_drawable:
            self.levelMin, self.levelMax = np.amin(self.imageDisp), self.pen_value
        if self.levelMin == self.levelMax:
            self.levelMin = self.levelMax - 1
        self.autoLevels()
        self.isNewImage = False
        self.isNewNorm = False
        return self.imageDisp

    def autoLevels(self):
        self._imageLevels = [(self.levelMin, self.levelMax)]
        super().autoLevels()


    def normRadioChanged(self):
        self.isNewNorm = True
        super().normRadioChanged()

    def roiRadioChanged(self):
        pen = "y"
        brush = pg.mkBrush(220, 100, 100, 200)
        roiSquareChecked = self.ui.roiSquare.isChecked()
        roiCircleChecked = self.ui.roiCircle.isChecked()
        self.roi.hide()
        if roiSquareChecked:
            self.roi = ROI(pos=self.roi.pos(), size=self.previousRoiSize)
            self.roi.addScaleHandle([1, 1], [0, 0])
            self.roi.addScaleHandle([0, 0], [1, 1])
            self.roi.setZValue(10000)
            self.roi.setPen(pen)
            self.roi.show()
        elif roiCircleChecked:
            self.roi = CircleROI(pos=self.roi.pos(), size=self.previousRoiSize)
            self.roi.setPen(pen)
            self.roi.setZValue(20)
            self.roi.show()
        else:
            self.roi = PolyLineROI(positions=self.previousRoiPositions, pos=self.roi.pos(), closed=True, hoverPen="r")
            self.roi.setPen(pen)
        self.view.addItem(self.roi)
        self.roi.sigRegionChangeFinished.connect(self.roiChanged)
        self.roiChanged()

    def intensity_value_slider(self, image):
        min_slider, max_slider = self.ui.rangeSliderThreshold.value()
        max_value = self.ui.rangeSliderThreshold.maximum()
        min_thresh = min_slider * np.amax(image) / max_value
        max_thresh = max_slider * np.amax(image) / max_value

        min_thresh = min_thresh - np.finfo(min_thresh.dtype).eps
        max_thresh = max_thresh + np.finfo(max_thresh.dtype).eps
        return min_thresh, max_thresh

    def sliderValueChanged(self):
        self.roiChanged()


    def roi_to_coordinates(self, image):
        min_t, max_t = self.intensity_value_slider(self.imageItem.image)
        topLeft = self.roi.boundingRect().topLeft()
        coords_roi = np.argwhere((self.mask_roi > 0) & (image >= min_t) & (image <= max_t))
        coords_roi = np.around(coords_roi + np.array(self.roi.pos()) + np.array([topLeft.x(), topLeft.y()])).astype(int)
        coords_roi = np.clip(coords_roi, 0, np.subtract(self.imageItem.image.T.shape, 1))
        coords_roi = coords_roi.T

        return coords_roi

    def roi_to_mean_spectra(self, image):
        axes = tuple([i for i in range(image.ndim)  if i != image.spectral_axis])
        if self.imageItem.axisOrder == "col-major":
            axes = axes[::-1]
        data, coords = self.roi.getArrayRegion(
            image, img=self.imageItem, axes=axes,
            returnMappedCoords=True, order=0)

        mean = []
        for i in range(data.shape[image.spectral_axis]):
            index = [i if j == image.spectral_axis else slice(None) for j in range(image.ndim)]
            im2D = data[tuple(index)].T
            coords_2D = self.roi_to_coordinates(im2D)
            linear = np.ravel_multi_index(coords_2D, self.imageItem.image.T.shape)
            linear = np.unique(linear)
            index = [linear, Ellipsis]
            spectra = self.imageDisp.spectra[tuple(index)]
            mean_value = np.mean(spectra[..., 1, i])
            mean.append(mean_value)
        return mean


    def roiChanged(self):
        self.isNewNorm = True
        if self.image is None:
            return

        if self.ui.roiSquare.isChecked() or \
           self.ui.roiCircle.isChecked():
            self.previousRoiSize = self.roi.size()
        elif self.ui.roiPolygon.isChecked():
            self.previousRoiPositions = [[handle["pos"].x(), handle["pos"].y()] for handle in self.roi.handles]

        image = self.getProcessedImage()

        current_image = self.imageItem.image
        colmaj = self.imageItem.axisOrder == 'col-major'
        if colmaj:
            axes = (1, 0)
        else:
            axes = (0, 1)
            current_image = current_image.T

        data, coords = self.roi.getArrayRegion(
            self.imageItem.image, img=self.imageItem, axes=axes,
            returnMappedCoords=True, order=0)

        if data is None:
            return

        image_roi = data.T
        self.mask_roi = self.roi.renderShapeMask(image_roi.shape[axes[0]], image_roi.shape[axes[1]])

        self.coords_roi = self.roi_to_coordinates(image_roi)
        image_roi = current_image[tuple(self.coords_roi)]

        if not image_roi.size:
            image_roi = [0]

        mean_roi = np.mean(image_roi)
        stddev_roi = np.std(image_roi)
        string_roi = "\u03BC="+ "{:.3e}".format(mean_roi)+ "\t\t\u03C3="+ "{:.3e}".format(stddev_roi)

        self.updateImage()
        self.ui.labelRoiChange.setText(string_roi)


    def roiClicked(self):
        super().roiClicked()
        try:
            self.ui.labelRoiChange.setVisible(self.ui.roiBtn.isChecked())
            self.ui.roiGroup.setVisible(self.ui.roiBtn.isChecked())
        except:
            pass
        self.ui.splitter.setSizes([self.height()-35, 35])
        self.updateImage()


    def plotSpectraROI(self):
        if self.coords_roi is None or self.axes["t"] is None:
            return

        dock = Dock("ROI " + str(len(self.area.docks)), size=(500,300), closable=True)
        self.area.addDock(dock, "below")
        vb = ViewBoxDirac()
        plot = pg.PlotWidget(viewBox=vb, enableMenu=False)

        min_slider, max_slider = self.ui.rangeSliderThreshold.value()
        min_value = self.ui.rangeSliderThreshold.minimum()
        max_value = self.ui.rangeSliderThreshold.maximum()
        if min_slider == min_value and max_value == max_slider:
            linear = np.ravel_multi_index(self.coords_roi, self.imageItem.image.T.shape)
            linear = np.unique(linear)
            ind = tuple([linear, Ellipsis])
            spectra = self.imageDisp.spectra[ind]
            mean_spectra = sp.spectra_mean(spectra)
        else:
            mean_spectra = self.roi_to_mean_spectra(self.imageDisp)

        bg = pg.BarGraphItem(x=self.tVals, height=mean_spectra, width=0)
        plot.addItem(bg)
        dock.addWidget(plot)

        self.winPlotROI.show()


    def normalize(self, image):
        return image

    def export(self, filename, index):
        """
        Export image view to file through Matplotlib
        Saves a scalebar on the side
        Accepted formats are .pdf, .png and .svg

        Parameters
        ----------
        self: type
            description
        filename: str
            image name
        index: int
            z-index of the image to save
        """
        if self.imageDisp is None:
            return

        if self.imageDisp.ndim == 2:
            img = self.imageDisp
        else:
            img = self.imageDisp[index]
        current_cm = self.ui.histogram.gradient.colorMap().getColors()
        current_cm = current_cm.astype(float)
        current_cm /= 255.0
        nb = len(current_cm[...,0])
        red, green, blue = [], [], []
        for i in range(nb):
            red.append([float(i/(nb-1)), current_cm[i, 0], current_cm[i, 0]])
            green.append([float(i/(nb-1)), current_cm[i, 1], current_cm[i, 1]])
            blue.append([float(i/(nb-1)), current_cm[i, 2], current_cm[i, 2]])
        cdict = {'red': np.array(red),
                 'green': np.array(green),
                 'blue': np.array(blue)}
        newcmp = colors.LinearSegmentedColormap("current_cmap", segmentdata=cdict)
        pos = plt.imshow(img, cmap=newcmp)
        plt.colorbar()
        plt.clim(self.levelMin, self.levelMax)
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.clf()
        plt.close()

    def setLevels(self, *args, **kwds):
        if self.imageDisp is None:
            return
        super().setLevels(*args, **kwds)

    def exportClicked(self):
        """
        Called when the "Export" button is clicked
        """
        fileName, image_format = QtWidgets.QFileDialog.getSaveFileName(self.parentWidget(), "Save image as...", "", "PNG images (*.png);;Portable Document Format (*.pdf);;Scalable Vector Graphics (*.svg)")
        if not fileName:
            return
        root, ext = os.path.splitext(fileName)
        if not ext:
            if "png" in image_format:
                ext = ".png"
            if "svg" in image_format:
                ext = ".svg"
            if "pdf" in image_format:
                ext = ".pdf"
        if ext == ".png" or ext == ".svg" or ext == ".pdf":
            self.export(root + ext, self.currentIndex)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No extension or wrong extension specified")
            msg.setInformativeText("Please specify a valid extension (.png, .svg or .pdf) ")
            msg.setWindowTitle("Wrong image format")
            msg.exec_()

    def exportSlicesClicked(self):
        """
        Called when the "Export all slices" button is clicked
        """
        path = QtWidgets.QFileDialog.getExistingDirectory(self.parentWidget(), "Select a directory", "")
        if len(self.threads) > 0:
            self.signal_abort.emit()
            for thread, worker in self.threads:
                thread.quit()
                thread.wait()
        if not path:
            return
        self.previousIndex = self.currentIndex
        worker = WorkerExport(self, path)
        thread = QtCore.QThread()
        worker.moveToThread(thread)
        worker.signal_end.connect(self.reset_index)
        worker.signal_start.connect(self.signal_start_export.emit)
        worker.signal_end.connect(self.signal_end_export.emit)
        worker.signal_progress.connect(lambda progress: self.signal_progress_export.emit(progress))
        self.signal_abort.connect(worker.abort)
        thread.started.connect(worker.work)
        thread.start()
        self.threads.append((thread, worker))

    def reset_index(self):
        """
        Called when the end signal of export slices is emitted
        """
        self.currentIndex = self.previousIndex

    def levelsChanged(self):
        """
        Called when the levels of the histogram are changed
        """
        self.levelMin, self.levelMax = self.ui.histogram.getLevels()


    def spectraToggled(self):
        self.winPlot.setVisible(self.ui.spectraBtn.isChecked())

    def draggedSpectra(self, event):
        ViewBoxDirac.mouseDragEvent(self.plot.getViewBox(), event)
        if event.isFinish():
            indices = self.plot.getViewBox().x_selected
            if not indices.size:
                return
            self.setCurrentIndices(indices)

    def clickedSpectra(self, scatter, points):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        is_adding = False
        if modifiers == QtCore.Qt.ControlModifier:
            is_adding = True
        if not is_adding:
            for p in self.lastPointsClicked:
                p.resetPen()
            self.lastPointsClicked = []
        for p in points:
            p.setPen(self.clickedPen)
        self.lastPointsClicked = np.append(self.lastPointsClicked, points)
        indices = [p.pos().x() for p in self.lastPointsClicked]
        self.setCurrentIndices(indices)

    def setCurrentIndices(self, times):
        self.ignorePlaying = True
        median_val = np.median(times)
        self.timeLine.setValue(median_val)

        indices = np.argwhere(np.in1d(self.image.mzs, times)).flatten()
        self.currentIndex = indices
        self.updateImage()
        self.currentIndex = np.int64(np.median(indices))
        self.ignorePlaying = False

    def buildPlot(self):
        if self.image is None:
            return
        if self.axes["t"] is None:
            return
        self.displayed_spectra = self.image.mean_spectra
        x = self.image.mzs
        self.plot.setOpts(x=x, height=self.displayed_spectra)
        # spots = [{'pos': [x[i], self.displayed_spectra[i]], 'data': 1} for i in range(len(x))]
        # self.plot.addPoints(spots)
        self.winPlot.autoRange()


    def buildMenu(self):
        """
        Adds the "Export all slices" option to the menu
        """
        super().buildMenu()
        self.exportSlicesAction = QtWidgets.QAction("Export all slices", self.menu)
        self.exportSlicesAction.triggered.connect(self.exportSlicesClicked)
        self.menu.addAction(self.exportSlicesAction)

    def getImageItemHistogram(self, bins='auto', step='auto', targetImageSize=200, targetHistogramSize=500, **kwds):
        """Returns x and y arrays containing the histogram values for the current image.
        For an explanation of the return format, see numpy.histogram().xb

        The *step* argument causes pixels to be skipped when computing the histogram to save time.
        If *step* is 'auto', then a step is chosen such that the analyzed data has
        dimensions roughly *targetImageSize* for each axis.

        The *bins* argument and any extra keyword arguments are passed to
        np.histogram(). If *bins* is 'auto', then a bin number is automatically
        chosen based on the image characteristics:

        * Integer images will have approximately *targetHistogramSize* bins,
          with each bin having an integer width.
        * All other types will have *targetHistogramSize* bins.

        This method is also used when automatically computing levels.
        """
        return None,None
