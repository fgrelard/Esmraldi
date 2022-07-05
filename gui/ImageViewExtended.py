import pyqtgraph as pg
import numpy as np
import matplotlib
import os
import time
import inspect

import numbers

import esmraldi.spectraprocessing as sp
from gui.viewboxdirac import ViewBoxDirac
from gui.scatterplotitemdirac import ScatterPlotItemDirac
from gui.crosshair import Crosshair, CrosshairDrawing
from gui.signal import Signal

#Allows to use QThreads without freezing
#the main application
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import skimage.color as color

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QApplication

import pyqtgraph as pg
from pyqtgraph.functions import affineSlice
from pyqtgraph.graphicsItems.ROI import ROI, CircleROI, PolyLineROI
from pyqtgraph.graphicsItems.GraphicsObject import GraphicsObject
from pyqtgraph.dockarea import DockArea, Dock

from superqt import QLabeledRangeSlider
from collections import ChainMap

import tracemalloc


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
        self.full_init = False
        self.current_image = None
        self.levelMin, self.levelMax = None, None
        self.isNewImage = False

        tracemalloc.start()
        self.number = 0

        super().__init__(parent, name, view, imageItem, *args)

        self.imageItem.getHistogram = self.getImageItemHistogram
        self.imageItem.mouseClickEvent = self.mouseClickEventImageItem
        self.imageItem.mouseDragEvent = self.mouseClickEventImageItem
        self.imageItem.mouseDoubleClickEvent = self.mouseDoubleClickEventImageItem

        self.timeLine.setPen('g')

        self.ui.histogram.sigLevelsChanged.connect(self.levelsChanged)

        self.ui.histogram.vb.setMaximumWidth(10)
        self.ui.histogram.setMaximumWidth(10)
        self.ui.gridLayout.addWidget(self.ui.graphicsView, 0, 0, 2, 2)
        self.ui.gridLayout.addWidget(self.ui.histogram, 0, 2, 1, 1)

        self.ui.histogram.gradient.loadPreset("viridis")
        self.gradient = self.ui.histogram.gradient.colorMap()

        self.ui.histogram.gradient.updateGradient()
        self.ui.histogram.gradientChanged()

        self.ui.labelRoiChange = QtWidgets.QLabel(self.ui.layoutWidget)
        self.ui.labelRoiChange.setObjectName("labelRoiChange")
        self.ui.labelRoiChange.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.ui.labelRoiChange.hide()
        self.ui.gridLayout.addWidget(self.ui.labelRoiChange, 2, 0, 1, 1)

        self.ui.menuBtn.clicked.connect(self.normToggled)
        self.ui.spectraBtn = QtWidgets.QPushButton(self.ui.layoutWidget)
        self.ui.spectraBtn.setCheckable(True)
        self.ui.spectraBtn.setObjectName("spectraBtn")
        self.ui.gridLayout.addWidget(self.ui.spectraBtn, 2, 2, 1, 1)
        self.ui.spectraBtn.setText(QtCore.QCoreApplication.translate("Form", "Spectra"))
        self.ui.spectraBtn.clicked.connect(self.spectraToggled)
        self.ui.roiBtn.setMaximumWidth(self.ui.spectraBtn.width())


        self.ui.normAutoRadio = QtWidgets.QRadioButton(self.ui.normGroup)
        self.ui.normAutoRadio.setObjectName("normAutoRadio")


        self.hide_partial()

        for i in reversed(range(self.ui.gridLayout_2.count())):
            self.ui.gridLayout_2.itemAt(i).widget().setParent(None)

        self.label = pg.LabelItem(justify='right')
        self.scene.addItem(self.label)
        self.scene.sigMouseMoved.connect(self.on_hover_image)

        self.actualIndex = 0

        self.threads = []

        self.mouse_x = 0
        self.mouse_y = 0

        self.mask_roi = None
        self.coords_roi = None
        self.coords_threshold = None

        self.plot = None
        self.displayed_spectra = None


        self.crossdrawer = CrosshairDrawing()
        self.is_clickable = False
        self.is_drawable = False

        self.pen_size = 1
        self.imageItem.drawAt = self.drawAt
        self.imageItem.render = self.render

        self.crosshair_move = Crosshair("")
        self.crosshair_move.setPenVisible(False)
        self.imageItem.getViewBox().addItem(self.crosshair_move)


        self.normDivideRadioChecked = False

        self.imageChangedSignal = Signal()

        self.ui.histogram.setHistogramRange = lambda mn, mx, padding=0.1: setHistogramRange(self.ui.histogram, mn, mx, padding)


        vb = ViewBoxDirac()
        self.winPlot = pg.PlotWidget(viewBox=vb, size=(1,1), enableMenu=False)
        self.plot = ScatterPlotItemDirac(pen="w")

        self.clickedPen = pg.mkPen("b")
        self.points = []

        self.winPlot.setMaximumHeight(100)
        self.winPlot.addItem(self.plot)
        self.plot.getViewBox().mouseDragEvent = self.draggedSpectra

        self.ui.gridLayout_3.addWidget(self.winPlot)

        self.previousRoiSize = 10
        self.previousRoiPositions = [[0,0], [10, 0], [5, 5]]

        self.build_norm_group()
        self.build_roi_group()


        self.is_focused = False
        self.is_linked = False

        self.enterEvent = lambda e: self.setFocus(e, True)
        self.leaveEvent = lambda e: self.setFocus(e, False)

        self.winPlot.setVisible(False)

        self.full_init = True

    def setFocus(self, e, focus):
        self.is_focused = focus
        if self.is_focused or not self.is_linked:
            self.crosshair_move.setPenVisible(False)
            self.scene.update()
        else:
            self.crosshair_move.setPenVisible(True)

    def build_norm_group(self):
        normButtonGroup = QtWidgets.QButtonGroup(self.ui.normGroup)

        self.ui.gridLayout_norm = self.ui.gridLayout_2

        self.ui.label_norm_type = QtWidgets.QLabel(self.ui.normGroup)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ui.label_norm_type.setFont(font)
        self.ui.label_norm_type.setObjectName("label_norm_type")
        self.ui.gridLayout_norm.addWidget(self.ui.label_norm_type, 0, 0, 1, 1)
        self.ui.label_norm_type.setText("Type:")

        self.ui.normOff = QtWidgets.QRadioButton(self.ui.normGroup)
        self.ui.normOff.setChecked(True)
        self.ui.normOff.setObjectName("normOff")
        self.ui.gridLayout_norm.addWidget(self.ui.normOff, 0, 1, 1, 1)
        self.ui.normOff.setText(QtCore.QCoreApplication.translate("Form", "Off"))

        self.ui.normTIC = QtWidgets.QRadioButton(self.ui.normGroup)
        self.ui.normTIC.setObjectName("normTIC")
        self.ui.gridLayout_norm.addWidget(self.ui.normTIC, 0, 2, 1, 1)
        self.ui.normTIC.setText(QtCore.QCoreApplication.translate("Form", "TIC"))

        self.ui.normIon = QtWidgets.QRadioButton(self.ui.normGroup)
        hbox = QtWidgets.QHBoxLayout()
        self.ui.normIon.setObjectName("normIon")
        hbox.addWidget(self.ui.normIon)
        self.ui.normIon.setText(QtCore.QCoreApplication.translate("Form", "Ion (m/z)"))

        self.ui.editNorm = QtWidgets.QLineEdit(self.ui.normGroup)
        self.ui.editNorm.setText(str(-1))
        self.ui.editNorm.setMaximumWidth(100)
        self.ui.editNorm.returnPressed.connect(lambda: self.normalize_ms(True))

        self.ui.labelCutoff = QtWidgets.QLabel(self.ui.normGroup)
        self.ui.label_norm_type.setObjectName("label_cut_off")
        self.ui.labelCutoff.setText(QtCore.QCoreApplication.translate("Form", "Cut-off"))

        self.ui.editCutoff = QtWidgets.QLineEdit(self.ui.normGroup)
        self.ui.editCutoff.setText(str(0))
        self.ui.editCutoff.setMaximumWidth(40)
        self.ui.editCutoff.returnPressed.connect(lambda: self.normalize_ms(True))

        hbox.addWidget(self.ui.editNorm)
        hbox.addWidget(self.ui.labelCutoff)
        hbox.addWidget(self.ui.editCutoff)

        # hbox.addStretch()
        self.ui.gridLayout_norm.addLayout(hbox, 0, 3, 1, 2)


        normButtonGroup.addButton(self.ui.normOff)
        normButtonGroup.addButton(self.ui.normTIC)
        normButtonGroup.addButton(self.ui.normIon)
        normButtonGroup.buttonToggled.connect(self.changeNorm)


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

        hbox = QtWidgets.QHBoxLayout()
        self.ui.roiImage = QtWidgets.QRadioButton(self.ui.roiGroup)
        self.ui.roiImage.setObjectName("roiImage")
        self.ui.roiImage.setText(QtCore.QCoreApplication.translate("Form", "Image"))
        self.ui.comboRoiImage = QtWidgets.QComboBox(self.ui.roiGroup)
        self.ui.comboRoiImage.setFixedWidth(100)
        self.ui.comboRoiImage.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        hbox.addWidget(self.ui.roiImage)
        hbox.addWidget(self.ui.comboRoiImage)
        self.ui.gridLayout_roi.addLayout(hbox, 0, 4, 1, 2)


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

        self.ui.resetROIButton = QtWidgets.QPushButton(self.ui.roiGroup)
        self.ui.resetROIButton.setObjectName("resetROIButton")
        self.ui.resetROIButton.setText("Reset ROI")
        self.ui.gridLayout_roi.addWidget(self.ui.resetROIButton, 2, 1, 1, 1)

        self.winPlotROI = QtWidgets.QMainWindow()
        self.area = DockArea()
        self.winPlotROI.setCentralWidget(self.area)
        self.winPlotROI.setWindowTitle("Plot ROI")
        self.winPlotROI.resize(600, 480)
        self.winPlotROI.closeEvent = self.hide_win_roi

        self.ui.gridLayout_3.addWidget(self.ui.roiGroup)

        self.ui.roiButtonGroup.addButton(self.ui.roiSquare)
        self.ui.roiButtonGroup.addButton(self.ui.roiCircle)
        self.ui.roiButtonGroup.addButton(self.ui.roiPolygon)
        self.ui.roiButtonGroup.addButton(self.ui.roiImage)

        self.ui.roiSquare.clicked.connect(self.roiRadioChanged)
        self.ui.roiCircle.clicked.connect(self.roiRadioChanged)
        self.ui.roiPolygon.clicked.connect(self.roiRadioChanged)
        self.ui.roiImage.clicked.connect(self.roiRadioChanged)

        self.ui.plotSpectraButton.clicked.connect(self.plotSpectraROI)
        self.ui.resetROIButton.clicked.connect(self.resetROI)
        self.roiRadioChanged()
        self.roiClicked()

    def hide_win_roi(self, ev):
        ev.ignore()
        self.winPlotROI.hide()

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


    def join_pixmap(self, p1, p2, mode=QtGui.QPainter.CompositionMode_SourceOver):
        s = p1.size().expandedTo(p2.size())
        result =  QtGui.QPixmap(s)
        result.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(result)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.drawPixmap(QtCore.QPoint(), p1)
        painter.setCompositionMode(mode)
        painter.drawPixmap(result.rect(), p2, p2.rect())
        painter.end()
        return result

    def render(self):
        pg.ImageItem.render(self.imageItem)
        if self.imageItem is None or self.imageItem.qimage is None:
            return

        pixel_value = QtGui.qRgba(231, 76, 60, 150)
        point = self.roi.boundingRect().topLeft()

        pixmap_image = QtGui.QPixmap.fromImage(self.imageItem.qimage)

        qimage_roi = self.imageItem.qimage.convertToFormat(QtGui.QImage.Format_RGBA8888)
        qimage_roi.fill(0)
        if self.ui.roiBtn.isChecked() and self.coords_roi is not None and slice(None) not in self.coords_roi:
            for x, y in self.coords_roi.T:
                qimage_roi.setPixel(x, y, pixel_value)

        pixmap_roi = QtGui.QPixmap.fromImage(qimage_roi)
        pixmap_overlay = self.join_pixmap(pixmap_image, pixmap_roi)
        self.imageItem.qimage = QtGui.QPixmap.toImage(pixmap_overlay)

        pixel_value = QtGui.qRgba(255, 0, 0, 255)
        if self.coords_threshold is not None:
            coords_threshold = np.array(self.coords_threshold)
            npoints = coords_threshold.shape[-1]
            th_npoints = np.prod(self.imageItem.image.shape[:2])
            if npoints != th_npoints:
                for x,y in coords_threshold.T:
                    self.imageItem.qimage.setPixel(x, y, pixel_value)

        self.imageItem._renderRequired = False
        self.imageItem._unrenderable = False


    def renderRoi(self, current_image):
        if False and self.ui.roiBtn.isChecked() and self.coords_roi is not None:
            roi_image = np.zeros_like(current_image)
            coords = tuple(c for c in self.coords_roi[::-1])
            roi_image[coords] = current_image[coords]
            self.imageItem.updateImage(roi_image, autoLevels=True)
        else:
            self.imageItem.updateImage(current_image, autoLevels=True)


    def mouseClickEventImageItem(self, ev):
        pg.ImageItem.mouseClickEvent(self.imageItem, ev)
        pos = ev.pos()
        if self.is_drawable:
            pos = [int(pos.x()), int(pos.y())]
            if ev.button() == QtCore.Qt.RightButton:
                ev.accept()
                shift = self.pen_size//2
                min_x, max_x = pos[0] - shift, pos[0] + shift
                min_y, max_y = pos[1] - shift, pos[1] + shift
                local_values = self.imageDisp[self.currentIndex, min_y:max_y, min_x:max_x]
                self.update_pen(pen_size=self.pen_size, array=local_values)
            else:
                self.update_pen(pen_size=self.pen_size, array=None)
            self.drawAt(pos, ev)


    def resetCross(self):
        children = self.imageItem.getViewBox().allChildren()
        self.points = []
        for child in children:
            if isinstance(child, Crosshair) and child != self.crosshair_move:
                self.imageItem.getViewBox().removeItem(child)
        self.crossdrawer = CrosshairDrawing()


    def mouseDoubleClickEventImageItem(self, ev):
        pg.ImageItem.mouseClickEvent(self.imageItem, ev)
        pos = ev.pos()
        if self.is_clickable:
            x, y = pos.x(), pos.y()
            self.points.append([x, y])
            cross = self.crossdrawer.get_drawable_crosshair(x, y)
            self.imageItem.getViewBox().addItem(cross)

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
            self.ui.normOffRadio.setChecked(True)
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
        self.actualIndex = self.currentIndex
        is_shown = False
        if self.imageDisp is not None:
            previousShape = self.imageDisp.shape
            is_shown = True

        self.isNewImage = True
        self.displayed_spectra = None
        self.current_image = None
        super().setImage(img, autoRange, autoLevels, levels, axes, xvals, pos, scale, transform, autoHistogramRange)
        self.imageChangedSignal.signal.emit()

        #Changes wheel event
        self.ui.roiPlot.setMouseEnabled(True, True)
        self.ui.roiPlot.wheelEvent = self.roi_scroll_bar
        max_t = self.imageDisp.shape[0]
        self.normRgn.setRegion((1, max_t//2))

        if self.image is None or not hasattr(self.image, "mean_spectra"):
            self.timeLine.hide()
            self.winPlot.hide()
        else:
            self.buildPlot()
            self.plot.getViewBox().setXLink(self.timeLine.getViewBox())
            self.timeLine.show()
            self.winPlot.autoRange()

        self.pen_value = np.amax(self.imageDisp)+1

        if not is_shown:
            return
        #Shows image at previous z-index if in range
        if self.imageDisp.ndim > 2 and previousIndex < self.imageDisp.shape[0] and self.hasTimeAxis():
            self.setCurrentIndex(previousIndex)
        else:
            self.get_current_image()

        snapshot = tracemalloc.take_snapshot()
        snapshot.dump("snapshot"+str(self.number)+".pickle")
        self.number += 1

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

    def update_crosshair_move(self, pos):
         self.crosshair_move.setPos(pos)

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
        if not self.is_focused:
            self.update_crosshair_move(mousePoint)

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


        if self.hasTimeAxis():
            position_z = str(self.tVals[self.currentIndex])

        value = self.imageItem.image[(self.mouse_y, self.mouse_x)]
        if isinstance(value, numbers.Number):
            str_value = "{:.3e}".format(value)
        else:
            str_value = ",".join([str(v) for v in value])
        self.label.setText("<p><span>" + position + "</span><span style='font-weight:bold; color: green;'>: " + str_value + "</span>" + "</p><p><span style='font-weight:bold'>" + position_z + "</span></p>")


    def setCurrentIndex(self, ind):
        super().setCurrentIndex(ind)
        self.actualIndex = self.currentIndex
        self.signal_mz_change.emit(self.tVals[self.currentIndex])
        self.update_label()

    def timeLineChanged(self):
        super().timeLineChanged()
        self.current_image = None
        self.actualIndex = self.currentIndex
        self.setCurrentIndices(self.actualIndex)
        self.signal_mz_change.emit(self.tVals[self.currentIndex])
        self.update_label()
        self.roiChanged()


    def getProcessedImage(self):
        if self.isNewImage and self.levelMin is not None:
            self.imageDisp = self.image
        elif self.imageDisp is None:
            self.imageDisp = self.normalize(self.image)
        if self.hasTimeAxis():
            self.get_current_image()
            self.levelMin, self.levelMax = np.amin(self.current_image), np.amax(self.current_image)
        else:
            self.levelMin, self.levelMax = np.amin(self.imageDisp), np.amax(self.imageDisp)
        if self.levelMin == self.levelMax:
            self.levelMin = self.levelMax - 1
        self.autoLevels()
        self.isNewImage = False
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
        roiPolygonChecked = self.ui.roiPolygon.isChecked()
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
        elif roiPolygonChecked:
            self.roi = PolyLineROI(positions=self.previousRoiPositions, pos=self.roi.pos(), closed=True, hoverPen="r")
            self.roi.setPen(pen)
        if  self.ui.roiImage.isChecked():
            self.coords_roi = np.array([slice(None) for i in range(2)])
        else:
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


    def roi_to_coordinates(self, image, min_t, max_t, offset=np.array([0, 0]), mask_roi=None):
        if mask_roi is None:
            mask_roi = np.ones_like(image)
        coords_mask = np.argwhere(mask_roi > 0)
        coords_roi = np.around(coords_mask + offset).astype(int)
        condition = (coords_roi >= (0,0)) & (coords_roi < image.shape)
        coords_roi = coords_roi[condition.all(axis=-1)]

        coords_threshold = np.argwhere((image >= min_t) & (image <= max_t))
        rindex_roi = np.ravel_multi_index(coords_roi.T, image.shape)
        rindex_threshold = np.ravel_multi_index(coords_threshold.T, image.shape)

        rindex = np.intersect1d(rindex_roi, rindex_threshold)
        coords_roi = np.unravel_index(rindex, image.shape)
        coords_roi = np.array(coords_roi)
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
            min_t, max_t = self.intensity_value_slider(im2D)
            offset = np.array(self.roi.pos()) + np.array([self.roi.boundingRect().topLeft().x(), self.roi.boundingRect().topLeft().y()])
            coords_2D = self.roi_to_coordinates(im2D, min_t, max_t, offset, self.mask_roi)
            linear = np.ravel_multi_index(coords_2D, self.imageItem.image.T.shape)
            linear = np.unique(linear)
            index = [linear, Ellipsis]
            spectra = self.imageDisp.spectra[tuple(index)]
            current_spectra = spectra[..., 1, i]
            mean_value = 0
            if current_spectra.size:
                mean_value = np.mean(current_spectra)
            mean.append(mean_value)
        return mean


    def roiChanged(self):
        if self.image is None:
            return

        if self.ui.roiSquare.isChecked() or \
           self.ui.roiCircle.isChecked():
            self.previousRoiSize = self.roi.size()
        elif self.ui.roiPolygon.isChecked():
            self.previousRoiPositions = [[handle["pos"].x(), handle["pos"].y()] for handle in self.roi.handles]

        self.get_current_image()
        current_image = self.current_image
        colmaj = self.imageItem.axisOrder == 'col-major'
        dim = len(current_image.shape)

        coords_image = current_image
        if dim >= 3:
            coords_image = (color.rgb2gray(current_image[..., :3]) * 255).astype(np.uint8)
        if colmaj:
            axes = (1, 0)
        else:
            axes = (0, 1)
            current_image = current_image.T
            coords_image = coords_image.T

        data, coords = self.roi.getArrayRegion(
            self.imageItem.image, img=self.imageItem, axes=axes,
            returnMappedCoords=True, order=0)

        if data is None:
            return
        if len(data.shape) >= 3:
            data = data.max(axis=-1)
        image_roi = data.T
        self.mask_roi = self.roi.renderShapeMask(image_roi.shape[axes[0]], image_roi.shape[axes[1]])

        min_t, max_t = self.intensity_value_slider(coords_image)
        offset = np.array(self.roi.pos()) + np.array([self.roi.boundingRect().topLeft().x(), self.roi.boundingRect().topLeft().y()])
        if not self.ui.roiImage.isChecked():
            self.coords_roi = self.roi_to_coordinates(coords_image, min_t, max_t, offset, self.mask_roi)
        # if len(current_image.shape) >= 3 and not any([c is Ellipsis for c in self.coords_roi]):
        #     self.coords_roi = (Ellipsis,) + tuple(self.coords_roi)
        self.finalize_roi_change()



    def finalize_roi_change(self):
        self.setCurrentIndices(self.actualIndex)
        current_image = self.normalize_ms()

        if len(current_image.shape) >= 3:
            return

        image_roi = current_image.T[tuple(self.coords_roi)]

        if not image_roi.size:
            image_roi = [0]

        mean_roi = np.mean(image_roi)
        stddev_roi = np.std(image_roi)
        string_roi = "\u03BC="+ "{:.3e}".format(mean_roi)+ "\t\t\u03C3="+ "{:.3e}".format(stddev_roi)
        self.ui.labelRoiChange.setText(string_roi)


    def roiClicked(self):
        super().roiClicked()

        if self.full_init and self.ui.roiImage.isChecked():
            self.roi.hide()

        if not self.ui.roiBtn.isChecked():
            self.updateImage()

        if not self.hasTimeAxis():
            self.timeLine.hide()
            self.ui.roiPlot.setVisible(False)

        try:
            self.ui.labelRoiChange.setVisible(self.ui.roiBtn.isChecked())
            self.ui.roiGroup.setVisible(self.ui.roiBtn.isChecked())
        except:
            pass
        self.ui.splitter.setSizes([self.height()-35, 35])


    def plotSpectraROI(self):
        if self.coords_roi is None or not self.hasTimeAxis():
            return

        dock = Dock("ROI " + str(len(self.area.docks.valuerefs())), size=(500,300), closable=True)

        containers, _ = self.area.findAll()
        if len(containers) <= 1 and (len(containers) == 0 or isinstance(containers[-1], pg.dockarea.Container.TContainer)):
            self.area.addDock(dock, "below")
        else:
            self.area.addDock(dock, "below", self.area.docks.valuerefs()[-1]())
        vb = ViewBoxDirac(selectable=False)
        plot = pg.PlotWidget(viewBox=vb, enableMenu=False)

        min_slider, max_slider = self.ui.rangeSliderThreshold.value()
        min_value = self.ui.rangeSliderThreshold.minimum()
        max_value = self.ui.rangeSliderThreshold.maximum()

        linear = np.ravel_multi_index(self.coords_roi, self.imageItem.image.T.shape, order="F")
        linear = np.unique(linear)
        ind = tuple([linear, Ellipsis])
        spectra = self.imageDisp.spectra[ind]
        self.get_current_image()
        norm_img = np.ones_like(self.current_image)
        if self.imageDisp.normalization_image is not None:
            norm_img = self.imageDisp.normalization_image
        norm_img = norm_img.flatten()[linear]
        mean_spectra = self.imageDisp.compute_mean_spectra(spectra, norm_img)

        scatter = ScatterPlotItemDirac(pen="w")
        scatter.setDiracs([self.tVals, mean_spectra])
        plot.addItem(scatter)
        dock.addWidget(plot)

        self.winPlotROI.show()

    def get_current_image(self):
        if self.current_image is not None:
            return
        if self.hasTimeAxis():
            if np.amax(self.actualIndex) < self.imageDisp.shape[0]:
                self.current_image = self.imageDisp[self.actualIndex]
            else:
                self.current_image = self.imageDisp[0]
        else:
            self.current_image = self.imageDisp

    def resetROI(self, event):
        self.previousRoiSize = 10
        self.previousRoiPositions = [[0,0], [10, 0], [5, 5]]
        self.roiRadioChanged()
        self.roi.setPos((0,0))
        self.autoRange()

    def changeNorm(self, button, is_toggled):
        if not is_toggled:
            return
        self.normalize_ms(True)

    def normalize_ms(self, new_value=False):
        self.get_current_image()
        current_image = self.current_image
        normed_image = np.zeros_like(current_image)
        if self.imageItem.image is not None and self.hasTimeAxis():
            if self.ui.normTIC.isChecked():
                tic = self.image.tic
                if new_value:
                    self.image.normalization_image = tic.reshape(current_image.shape)

            elif self.ui.normIon.isChecked():
                text = self.ui.editNorm.text()
                try:
                    value = float(text)
                except:
                    value = 0
                try:
                    cut_off = float(self.ui.editCutoff.text())
                except:
                    cut_off = 0
                if new_value:
                    actual_value = self.image.mzs[np.abs(value - self.image.mzs).argmin()]
                    if self.image.is_ppm:
                        tol = self.image.tolerance_ppm(actual_value)
                    else:
                        tol = self.image.tolerance
                    norm_img = self.image.get_ion_image_mzs(actual_value, tol, tol)
                    norm_img[norm_img <= cut_off] = -1
                    self.image.normalization_image = norm_img

            if not self.ui.normOff.isChecked():
                np.divide(current_image, self.image.normalization_image, out=normed_image, where=self.image.normalization_image>0)
                current_image = normed_image
            if new_value:
                self.buildPlot()

        self.renderRoi(current_image)
        self.levelMin, self.levelMax = np.amin(self.imageItem.image), np.amax(self.imageItem.image)
        self.autoLevels()
        return current_image


    def updateImage(self, autoHistogramRange=True):
        if self.image is None:
            return
        self.getProcessedImage()
        self.get_current_image()
        self.imageItem.updateImage(self.current_image)
        if self.full_init:
            self.normalize_ms()

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
        self.winPlot.autoRange()

    def draggedSpectra(self, event):
        ViewBoxDirac.mouseDragEvent(self.plot.getViewBox(), event)
        if event.isFinish() and event.button() == QtCore.Qt.MouseButton.LeftButton:
            indices = self.plot.getViewBox().x_selected
            if not indices.size:
                return
            self.setCurrentTimes(indices)


    def setCurrentTimes(self, times):
        self.ignorePlaying = True
        indices = np.argwhere(np.in1d(self.image.mzs, times)).flatten()
        median_val = np.median(times)
        self.timeLine.setValue(median_val)
        self.setCurrentIndices(indices)
        self.ignorePlaying = False

    def setCurrentIndices(self, indices):
        self.actualIndex = indices
        self.currentIndex = self.actualIndex
        self.updateImage()

        self.levelMin, self.levelMax = np.amin(self.imageItem.image), np.amax(self.imageItem.image)
        self.autoLevels()
        if self.displayed_spectra is not None:
            data = self.tVals[self.actualIndex], self.displayed_spectra[self.actualIndex]
            if isinstance(self.actualIndex, numbers.Number):
                data = [data[0]], [data[1]]
            self.plot.setSelectedPoints(data)
        self.currentIndex = np.int64(np.median(indices))

    def buildPlot(self):
        if self.image.normalization_image is not None:
            self.displayed_spectra = self.image.compute_mean_spectra(self.image.spectra.copy())
        else:
            self.displayed_spectra = self.image.mean_spectra
        x = self.image.mzs
        self.plot.clear()
        self.plot.setDiracs([x, self.displayed_spectra])
        self.winPlot.autoRange()


    def buildMenu(self):
        """
        Adds the "Export all slices" option to the menu
        """
        pass

    def menuClicked(self):
        pass

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
