import numpy as np

from esmraldi.utils import button_tooltip_on_hover, msimage_for_visualization
from gui.signal import Signal
from PyQt5 import QtCore

class WorkerThreshold(QtCore.QObject):
    signal_end = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, image, coords):
        super().__init__()
        self.image = image
        self.coords = coords

    @QtCore.pyqtSlot()
    def work(self):
        thresholded_image = np.zeros_like(self.image).T
        thresholded_image[tuple(self.coords)] = 1
        self.signal_end.emit(thresholded_image.T)

    def abort(self):
        self.is_abort = True

class ThresholdingController:

    def __init__(self, view, imageview, range_slider):
        self.view = view
        self.imageview = imageview
        self.range_slider = range_slider

        self.trigger_compute = Signal()
        self.trigger_end = Signal()

        self.worker = None
        self.thread = None

        self.range_slider.valueChanged.connect(self.threshold)

        self.view.buttonBox.accepted.connect(self.generate_thresholded_image)
        self.view.buttonBox.rejected.connect(self.end)

    def threshold(self):
        image = self.imageview.imageItem.image.T
        max_x, max_y = image.shape
        new_positions = [[0, 0], [max_x, max_y]]
        min_slider, max_slider = self.range_slider.value()
        min_thresh = min_slider - np.finfo(float).eps
        max_thresh = max_slider + np.finfo(float).eps
        self.imageview.coords_threshold = self.imageview.roi_to_coordinates(image, min_thresh, max_thresh)
        self.imageview.updateImage()

    def generate_thresholded_image(self):
        self.worker = WorkerThreshold(self.imageview.imageItem.image, self.imageview.coords_threshold)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.work)
        self.trigger_compute.signal.emit()

    def end(self):
        self.trigger_end.signal.emit()
