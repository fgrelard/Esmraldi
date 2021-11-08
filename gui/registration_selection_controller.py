import numpy as np
from PyQt5 import QtCore
from gui.signal import Signal

import SimpleITK as sitk

import esmraldi.registration as reg

from skimage.color import rgb2gray

class WorkerRegistrationSelection(QtCore.QObject):

    signal_end = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, fixed, moving, points_fixed, points_moving):
        super().__init__()
        self.fixed = fixed
        self.moving = moving
        self.points_fixed = points_fixed
        self.points_moving = points_moving
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        print("starting")
        self.fixed = rgb2gray(self.fixed)
        self.moving = rgb2gray(self.moving)
        fixed_itk = sitk.GetImageFromArray(self.fixed)
        moving_itk = sitk.GetImageFromArray(self.moving)
        landmark_transform = sitk.LandmarkBasedTransformInitializer(sitk.AffineTransform(2), self.points_fixed, self.points_moving)

        resampler = reg.initialize_resampler(fixed_itk, landmark_transform)
        deformed_itk = resampler.Execute(moving_itk)
        deformed = sitk.GetArrayFromImage(deformed_itk)
        print(deformed.dtype)
        self.signal_end.emit(deformed)

    def abort(self):
        self.is_abort = True


class RegistrationSelectionController:

    def __init__(self, view, imageview, imageview2):
        self.view = view
        self.imageview = imageview
        self.imageview2 = imageview2

        self.trigger_compute = Signal()
        self.trigger_end = Signal()

        self.worker = None
        self.thread = None

        self.view.pushButton.clicked.connect(self.clear_selection)
        self.view.buttonBox.accepted.connect(self.compute_transformation)
        self.view.buttonBox.rejected.connect(self.end)


    def start(self):
        self.set_clickable(True)
        self.imageview.points = [[41, 44], [525, 48], [533, 409]]
        self.imageview2.points = [[177, 34], [354, 213], [224, 348]]
        self.compute_transformation()

    def set_clickable(self, clickable):
        self.imageview.setClickable(clickable)
        self.imageview2.setClickable(clickable)

    def clear_selection(self):
        self.imageview.resetCross()
        self.imageview2.resetCross()

    def compute_transformation(self):
        print("compute transfo")
        fixed = self.imageview.imageItem.image
        moving = self.imageview2.imageItem.image
        list1 = [c for p in self.imageview.points for c in p]
        list2 = [c for p in self.imageview2.points for c in p]
        self.worker = WorkerRegistrationSelection(fixed, moving, list1, list2)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.work)
        self.trigger_compute.signal.emit()
        # print(landmark_transform)
        # self.end()

    def end(self):
        self.clear_selection()
        self.set_clickable(False)
        self.trigger_end.signal.emit()
