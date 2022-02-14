import numpy as np

from esmraldi.peakdetectionmeanspectrum import PeakDetectionMeanSpectrum

from esmraldi.utils import button_tooltip_on_hover, msimage_for_visualization

from gui.signal import Signal
from PyQt5 import QtCore


class WorkerPeakPickingMeanSpectrum(QtCore.QObject):
    signal_end = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    signal_progress = QtCore.pyqtSignal(int)

    def __init__(self, mzs, mean_spectrum, prominence, step):
        super().__init__()
        self.mzs = mzs
        self.mean_spectrum = mean_spectrum
        self.prominence = prominence
        self.step = step

    @QtCore.pyqtSlot()
    def work(self):
        peak_detection = PeakDetectionMeanSpectrum(self.mzs, self.mean_spectrum, self.prominence, self.step)
        peak_indices = peak_detection.extract_peaks()
        peaks = self.mzs[peak_indices]
        intensities = self.mean_spectrum[peak_indices]
        self.signal_end.emit(peaks, intensities)

    def abort(self):
        self.is_abort = True

class PeakPickingMeanSpectrumController:
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
        mean_spectra = image.mean_spectra
        mzs = image.mzs
        prominence = float(self.view.lineEdit_prominence.text())
        step = float(self.view.lineEdit_step.text())
        self.worker = WorkerPeakPickingMeanSpectrum(mzs, mean_spectra, prominence, step)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.work)
        self.trigger_compute.signal.emit()

    def end(self):
        self.trigger_end.signal.emit()
