import numpy as np

from esmraldi.utils import button_tooltip_on_hover, msimage_for_visualization
from gui.signal import Signal
from PyQt5 import QtCore

class WorkerExtractChannels(QtCore.QObject):
    signal_end = QtCore.pyqtSignal(object, int)

    def __init__(self, image, number):
        super().__init__()
        self.image = image
        self.number = number

    @QtCore.pyqtSlot()
    def work(self):
        axis = getattr(self.image, "spectral_axis", -1)
        if len(self.image.shape) >= 3 and self.number > 0 and self.number <= self.image.shape[axis]:
            image = self.image[..., self.number-1]
        else:
            image = None
        self.signal_end.emit(image, self.number)

    def abort(self):
        self.is_abort = True


class ExtractChannelController:
    def __init__(self, view, imageview):
        self.view = view
        self.imageview = imageview

        self.trigger_compute = Signal()
        self.trigger_end = Signal()

        self.worker = None
        self.thread = None

        self.view.pushButton = button_tooltip_on_hover(self.view.pushButton)
        self.view.buttonBox.accepted.connect(self.extract_channels)
        self.view.buttonBox.rejected.connect(self.end)


    def extract_channels(self):
        channel_number = int(self.view.lineEdit_channel.text())
        image = self.imageview.image
        self.worker = WorkerExtractChannels(image, channel_number)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.work)
        self.trigger_compute.signal.emit()

    def end(self):
        self.trigger_end.signal.emit()
