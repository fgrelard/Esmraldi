import numpy as np

import esmraldi.spectraprocessing as sp
import esmraldi.imzmlio as imzmlio

from esmraldi.utils import button_tooltip_on_hover, msimage_for_visualization
from esmraldi.msimage import MSImage

from gui.signal import Signal
from PyQt5 import QtCore

class WorkerSpectraAlignment(QtCore.QObject):
    signal_end = QtCore.pyqtSignal(object)
    signal_progress = QtCore.pyqtSignal(int)

    def __init__(self, msimage, step, is_ppm):
        super().__init__()
        print(step, is_ppm)
        self.msimage = msimage
        self.step = step
        self.is_ppm = is_ppm
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        image_size = self.msimage.shape[1:][::-1]
        if self.msimage.peaks is not None:
            realigned_spectra = sp.realign_generic(self.msimage.spectra, self.msimage.peaks, self.step, self.is_ppm)
        else:
            realigned_spectra = sp.realign_generic(self.msimage.spectra, self.msimage.spectra[:, 0], self.step, self.is_ppm)

        print("End realign")
        if self.is_abort:
            return
        full_spectra_sparse = imzmlio.get_full_spectra_sparse(realigned_spectra, np.prod(image_size))
        if self.is_abort:
            return
        print("End full spectra sparse")
        image = MSImage(full_spectra_sparse, image=None, shape=image_size, tolerance=0.003)
        # import tracemalloc
        # snapshot = tracemalloc.take_snapshot()
        # snapshot.dump("snapshot_align.pickle")
        print("ms image for visualization")
        image = msimage_for_visualization(image)
        print(image.shape)
        self.signal_end.emit(image)

    def abort(self):
        self.is_abort = True

class SpectraAlignmentController:
    def __init__(self, view, imageview):
        self.view = view
        self.imageview = imageview

        self.trigger_compute = Signal()
        self.trigger_end = Signal()

        self.worker = None
        self.thread = None

        self.view.pushButton = button_tooltip_on_hover(self.view.pushButton)
        self.view.buttonBox.accepted.connect(self.spectra_alignment)
        self.view.buttonBox.rejected.connect(self.end)

    def spectra_alignment(self):
        if not hasattr(self.imageview.image, "spectra"):
            return
        is_ppm = False
        index = self.view.comboBox.currentIndex()
        if index > 0:
            is_ppm = True
        image = self.imageview.image
        step = float(self.view.lineEdit_step.text())
        self.worker = WorkerSpectraAlignment(image, step, is_ppm)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.work)
        self.trigger_compute.signal.emit()

    def end(self):
        self.trigger_end.signal.emit()
