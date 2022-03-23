import numpy as np

from esmraldi.utils import button_tooltip_on_hover, msimage_for_visualization
from gui.signal import Signal
from PyQt5 import QtCore

class WorkerThreshold(QtCore.QObject):
    signal_end = QtCore.pyqtSignal(object)

    def __init__(self, threshold_min, threshold_max):
        pass

    @QtCore.pyqtSlot()
    def work(self):
        pass

    def abort(self):
        self.is_abort = True

class ThresholdingController:
    def __init__(self, view, imageview, range_slider):
        self.view = view
        self.imageview = imageview
        self.range_slider = range_slider

        self.trigger_compute = Signal()
        self.trigger_end = Signal()

        self.range_slider.valueChanged.connect(self.threshold)

        self.view.buttonBox.accepted.connect(self.threshold)
        self.view.buttonBox.rejected.connect(self.end)

    def threshold(self):
        image = self.imageview.imageDisp
        roi = self.imageview.roi
        min_slider, max_slider = self.range_slider.value()
        min_thresh = min_slider - np.finfo(float).eps
        max_thresh = max_slider - np.finfo(float).eps
        condition = np.argwhere((image >= min_thresh) & (image <= max_thresh))
        image[condition] =
        print(min_thresh, max_thresh)

    def end(self):
        self.trigger_end.signal.emit()
