import numpy as np
from PyQt5 import QtCore
from gui.signal import Signal

import SimpleITK as sitk

import esmraldi.registration as reg

from skimage.color import rgb2gray, rgba2rgb, gray2rgb

class WorkerRegistrationSelection(QtCore.QObject):

    signal_end = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, fixed, moving, input_register, points_fixed, points_moving):
        super().__init__()
        self.fixed = fixed
        self.moving = moving
        self.input_register = input_register
        self.points_fixed = points_fixed
        self.points_moving = points_moving
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        fixed_shape = ((2,) if self.fixed.ndim == 3 else ()) + (0, 1)
        moving_shape = ((2,) if self.moving.ndim == 3 else ()) + (0, 1)
        self.fixed = np.transpose(self.fixed, fixed_shape)
        self.moving = np.transpose(self.moving, moving_shape)

        lower = np.amin(self.points_fixed[::2]), np.amin(self.points_fixed[1::2])
        upper = np.amax(self.points_fixed[::2])+1, np.amax(self.points_fixed[1::2])+1
        self.fixed = self.fixed[:, lower[1]:upper[1], lower[0]:upper[0]]
        self.points_fixed = [int(p - lower[i%2]) for i, p in enumerate(self.points_fixed)]

        fixed_itk = sitk.GetImageFromArray(self.fixed)
        moving_itk = sitk.GetImageFromArray(self.moving)
        landmark_transform = sitk.LandmarkBasedTransformInitializer(sitk.AffineTransform(2), self.points_fixed, self.points_moving)

        dim_fixed = fixed_itk.GetDimension()
        if dim_fixed == 2:
            resampler = reg.initialize_resampler(fixed_itk, landmark_transform)
        if dim_fixed == 3:
            resampler = reg.initialize_resampler(fixed_itk[:,:,0], landmark_transform)

        size = moving_itk.GetSize()
        dim = moving_itk.GetDimension()

        if dim == 2:
            deformed_itk = resampler.Execute(moving_itk)
        elif dim == 3:
            pixel_type = moving_itk.GetPixelID()
            fixed_size = fixed_itk.GetSize()
            deformed_itk = sitk.Image(fixed_size[0], fixed_size[1], size[2], pixel_type )

            for i in range(size[2]):
                img_slice  = moving_itk[:, :, i]
                img_slice.SetSpacing([1, 1])
                out_slice = resampler.Execute(img_slice)
                out_slice = sitk.JoinSeries(out_slice)
                deformed_itk = sitk.Paste(deformed_itk, out_slice, out_slice.GetSize(), destinationIndex=[0, 0, i])

        deformed = sitk.GetArrayFromImage(deformed_itk)
        print(deformed.shape)
        if dim == 3:
            deformed = np.transpose(deformed, np.roll(moving_shape, 1))
        print(deformed.shape)
        print(self.input_register.shape)
        self.signal_end.emit(self.fixed, deformed)

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
        self.imageview.points = [[1110, 715], [1374, 696], [1374, 905]]
        self.imageview2.points = [[0, 0], [9, 0], [9, 9]]
        self.compute_transformation()

    def set_clickable(self, clickable):
        self.imageview.setClickable(clickable)
        self.imageview2.setClickable(clickable)

    def clear_selection(self):
        self.imageview.resetCross()
        self.imageview2.resetCross()

    def compute_transformation(self):
        fixed = self.imageview.imageItem.image
        moving = self.imageview2.imageItem.image
        input_register = self.imageview2.image
        list1 = [c for p in self.imageview.points for c in p]
        list2 = [c for p in self.imageview2.points for c in p]
        self.worker = WorkerRegistrationSelection(fixed, moving, input_register, list1, list2)
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
