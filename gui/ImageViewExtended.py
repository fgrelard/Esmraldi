import pyqtgraph as pg
import numpy as np
import matplotlib

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
import os
import pyqtgraph as pg
import time


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

        self.ui.roiGroup = QtWidgets.QButtonGroup(self.ui.normGroup)
        self.ui.normRadioGroup = QtWidgets.QButtonGroup(self.ui.normGroup)
        self.ui.label_roi = QtWidgets.QLabel(self.ui.normGroup)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ui.label_roi.setFont(font)
        self.ui.label_roi.setText(QtWidgets.QApplication.translate("Form", "ROI:", None))
        self.ui.roiSquareRadio = QtWidgets.QRadioButton(self.ui.normGroup)

        self.ui.roiSquareRadio.setText(QtWidgets.QApplication.translate("Form", "Square", None))
        self.ui.roiCircleRadio = QtWidgets.QRadioButton(self.ui.normGroup)
        self.ui.roiCircleRadio.setText(QtWidgets.QApplication.translate("Form", "Circle", None))

        self.ui.roiGroup.addButton(self.ui.roiSquareRadio)
        self.ui.roiGroup.addButton(self.ui.roiCircleRadio)

        self.ui.normAutoRadio.setText(QtWidgets.QApplication.translate("Form", "Stack", None))
        self.ui.normOffRadio.setText(QtWidgets.QApplication.translate("Form", "Manual", None))
        self.ui.normTimeRangeCheck.setText(QtWidgets.QApplication.translate("Form", "Slice range", None))
        self.ui.normDivideRadio.setText(QtWidgets.QApplication.translate("Form", "Auto", None))
        self.ui.label_5.setText(QtWidgets.QApplication.translate("Form", "Type:", None))

        self.ui.roiSquareRadio.clicked.connect(self.roiRadioChanged)
        self.ui.roiCircleRadio.clicked.connect(self.roiRadioChanged)
        self.ui.normAutoRadio.clicked.connect(self.normRadioChanged)

        self.ui.normRadioGroup.addButton(self.ui.normDivideRadio)
        self.ui.normRadioGroup.addButton(self.ui.normOffRadio)

        self.ui.roiSquareRadio.setChecked(True)
        self.ui.normDivideRadio.setChecked(True)


        self.hide_partial()

        for i in reversed(range(self.ui.gridLayout_2.count())):
            self.ui.gridLayout_2.itemAt(i).widget().setParent(None)

        self.ui.gridLayout_2.addWidget(self.ui.label_roi, 0, 0, 1, 1)
        self.ui.gridLayout_2.addWidget(self.ui.roiSquareRadio, 0, 1, 1, 1)
        self.ui.gridLayout_2.addWidget(self.ui.roiCircleRadio, 0, 2, 1, 1)
        self.ui.gridLayout_2.addWidget(self.ui.label_5, 1, 0, 1, 1)
        self.ui.gridLayout_2.addWidget(self.ui.normDivideRadio, 1, 1, 1, 1)

        self.ui.gridLayout_2.addWidget(self.ui.normOffRadio, 1, 2, 1, 1)
        self.ui.gridLayout_2.addWidget(self.ui.normFrameCheck, 2, 1, 1, 1)
        self.ui.gridLayout_2.addWidget(self.ui.label_3, 2, 0, 1, 1)
        self.ui.gridLayout_2.addWidget(self.ui.normROICheck, 2, 1, 1, 1)
        self.ui.gridLayout_2.addWidget(self.ui.normTimeRangeCheck, 2, 2, 1, 1)
        self.label = pg.LabelItem(justify='right')
        self.scene.addItem(self.label)
        self.scene.sigMouseMoved.connect(self.on_hover_image)

        self.threads = []

        self.mouse_x = 0
        self.mouse_y = 0

        self.plot = None
        self.displayed_spectra = None

        self.is_clickable = False
        self.is_drawable = False

        self.pen_size = 1
        self.imageCopy = None
        self.imageItem.drawAt = self.drawAt

        self.levelMin, self.levelMax = None, None
        self.isNewImage = False
        self.isNewNorm = False
        self.normDivideRadioChecked = False

        self.ui.histogram.setHistogramRange = lambda mn, mx, padding=0.1: setHistogramRange(self.ui.histogram, mn, mx, padding)

        self.plot = pg.ScatterPlotItem(size=5, pen=pg.mkPen(255,255, 255, 230), brush=pg.mkBrush(220, 220, 220, 230),hoverable=True,hoverPen=pg.mkPen(242, 38, 19),hoverSize=5,hoverBrush=pg.mkBrush(150, 40, 27))

        self.clickedPen = pg.mkPen("b")
        self.lastPointsClicked = []
        self.plot.sigClicked.connect(self.clickedSpectra)

        self.winPlot = pg.plot(size=(1,1))
        self.winPlot.addItem(self.plot)
        self.ui.gridLayout_3.addWidget(self.winPlot)
        self.winPlot.setMaximumHeight(100)
        # self.winPlot.setVisible(False)

    def roiRadioChanged(self):
        roiSquareChecked = self.ui.roiSquareRadio.isChecked()
        self.normRoi.hide()
        if roiSquareChecked:
            self.normRoi = pg.graphicsItems.ROI.ROI(pos=self.normRoi.pos(), size=self.normRoi.size())
            self.normRoi.addScaleHandle([1, 1], [0, 0])
            self.normRoi.addRotateHandle([0, 0], [0.5, 0.5])
            self.normRoi.setZValue(10000)
            self.normRoi.setPen('y')
            self.normRoi.show()
        else:
            self.normRoi = pg.graphicsItems.ROI.CircleROI(pos=self.normRoi.pos(), size=self.normRoi.size())
            self.normRoi.setPen("y")
            self.normRoi.setZValue(20)
            self.normRoi.show()
        self.view.addItem(self.normRoi)
        self.updateNorm()
        self.normRoi.sigRegionChangeFinished.connect(self.updateNorm)


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


    def hide_partial(self):
        """
        Hide some elements from the parent GUI
        """
        self.ui.roiBtn.hide()
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
        if self.imageDisp.ndim > 2 and previousIndex < self.imageDisp.shape[0]:
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
        if not (self.mouse_x >= 0 and self.mouse_x < self.imageDisp.shape[-1] and
            self.mouse_y >= 0 and self.mouse_y < self.imageDisp.shape[-2]):
            self.mouse_x = 0
            self.mouse_y = 0
        position = "(" + str(self.mouse_x) + ", " + str(self.mouse_y) + ")"
        position_z = ""

        if self.imageDisp.ndim == 3:
            position_z = str(self.tVals[self.currentIndex])
        value = "{:.3e}".format(self.imageItem.image[(self.mouse_y, self.mouse_x)])
        self.label.setText("<p><span>" + position + "</span><span style='font-weight:bold; color: green;'>: " + value + "</span>" + "</p><p><span style='font-weight:bold'>" + position_z + "</span></p>")


    def setCurrentIndex(self, ind):
        super().setCurrentIndex(ind)
        self.signal_mz_change.emit(self.tVals[self.currentIndex])
        self.update_label()

    def timeLineChanged(self):
        super().timeLineChanged()
        self.signal_mz_change.emit(self.tVals[self.currentIndex])
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

    def roiChanged(self):
        self.isNewNorm = True
        # super().roiChanged()

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
        median_val = np.median(indices)
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
        self.plot.clear()
        self.displayed_spectra = self.image.mean_spectra
        x = self.image.mzs
        spots = [{'pos': [x[i], self.displayed_spectra[i]], 'data': 1} for i in range(len(x))]
        self.plot.addPoints(spots)
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
