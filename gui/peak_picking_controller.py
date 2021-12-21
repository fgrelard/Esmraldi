import numpy as np

import esmraldi.spectraprocessing as sp

from esmraldi.utils import button_tooltip_on_hover, msimage_for_visualization
from gui.signal import Signal
from PyQt5 import QtCore


class WorkerPeakPicking(QtCore.QObject):
    signal_end = QtCore.pyqtSignal(np.ndarray)
    signal_progress = QtCore.pyqtSignal(int)

    def __init__(self, spectra, prominence, noise):
        super().__init__()
        self.spectra = spectra
        self.prominence = prominence
        self.noise = noise

    def determine_wlen(self):
        mz, I = self.spectra[0]
        min_diff = mz[1] - mz[0]
        if min_diff == 0:
            is_maybe_densify = self.spectra.is_maybe_densify
            self.spectra.is_maybe_densify = False
            first_mz, second_mz = max(self.spectra[:, 0, 0]), max(self.spectra[:, 0, 1])
            self.spectra.is_maybe_densify = is_maybe_densify
            min_diff = second_mz - first_mz
        wlen = max(int(len(mz)/10), int(1.0 / min_diff))
        return wlen

    @QtCore.pyqtSlot()
    def work(self):
        wlen = self.determine_wlen()
        if self.noise <= 0:
            peak_indices = sp.spectra_peak_indices_adaptative(self.spectra, factor=self.prominence, wlen=wlen)
        else:
            peak_indices = sp.spectra_peak_indices_adaptative_noiselevel(self.spectra, factor=self.prominence, wlen=wlen, noise_level=self.noise)

        peaks = np.array([[]])
        if peak_indices.size:
            peaks = np.array([x[peak_indices[i]] for i, (x,y) in enumerate(self.spectra)], dtype=object)
        self.signal_end.emit(peaks)

    def abort(self):
        self.is_abort = True

class PeakPickingController:
    def __init__(self, view, imageview):
        self.view = view
        self.imageview = imageview

        self.trigger_compute = Signal()
        self.trigger_end = Signal()

        self.worker = None
        self.thread = None

        self.view.pushButton = button_tooltip_on_hover(self.view.pushButton)
        self.view.pushButton_2 = button_tooltip_on_hover(self.view.pushButton_2)

        self.view.buttonBox.accepted.connect(self.peak_pick)
        self.view.buttonBox.rejected.connect(self.end)

    def peak_pick(self):
        if not hasattr(self.imageview.image, "spectra"):
            return
        image = self.imageview.image
        spectra = image.spectra
        prominence = float(self.view.lineEdit_prominence.text())
        noise = float(self.view.lineEdit_noise.text())
        self.worker = WorkerPeakPicking(spectra, prominence, noise)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.work)
        self.trigger_compute.signal.emit()

    def end(self):
        self.trigger_end.signal.emit()
