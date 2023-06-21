import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from gui.signal import Signal

import SimpleITK as sitk

import esmraldi.registration as reg
import esmraldi.imzmlio as imzmlio
from esmraldi.msimage import MSImage
from esmraldi.utils import msimage_for_visualization
from skimage.color import rgb2gray, rgba2rgb, gray2rgb


class WorkerRegistrationSelection(QtCore.QObject):
    """
    Worker for fiducial based registration
    """
    signal_end = QtCore.pyqtSignal(object)
    signal_progress = QtCore.pyqtSignal(int)

    def __init__(self, fixed, moving, points_fixed, points_moving):
        super().__init__()
        self.fixed = fixed
        self.moving = moving
        self.points_fixed = points_fixed
        self.points_moving = points_moving
        self.is_abort = False

    def preprocess_image(self, image):
        """
        Transpose images so they match dimensions
        before registration
        """
        is_ms_image = hasattr(image, "image")
        print(is_ms_image)
        processed_image = image
        shape = ((2,) if image.ndim == 3 else ()) + (0, 1)
        if is_ms_image:
            processed_image = processed_image.image.T
        else:
            processed_image = np.transpose(image, shape)
        return processed_image, is_ms_image

    def postprocess_image(self, image, ref_ms_image=None):
        """
        Transpose images so they match after registration
        """
        shape = ((2,) if image.ndim == 3 else ()) + (0, 1)
        print(image.shape, ref_ms_image)
        if ref_ms_image is not None:
            spectra = ref_ms_image.spectra
            new_image = MSImage(spectra, image)
            new_image = msimage_for_visualization(new_image, transpose=False)
        else:
            print(image.shape)
            if image.ndim == 3:
                shape = np.roll(shape, 1)
            new_image = np.transpose(image, shape)
        return new_image

    def crop_image(self, image, points):
        """
        Crop images in between space defined by fiducials
        """
        lower = round(np.amin(points[::2])), round(np.amin(points[1::2]))
        upper = round(np.amax(points[::2])+1), round(np.amax(points[1::2])+1)
        if image.ndim == 2:
            image = image[lower[1]:upper[1], lower[0]:upper[0]]
        else:
            image = image[:, lower[1]:upper[1], lower[0]:upper[0]]
        new_points = [int(p - lower[i%2]) for i, p in enumerate(points)]
        return image, new_points

    def apply_registration(self, fixed, register, landmark_transform):
        """
        Computes transformation using fiducial markers
        """
        fixed_dim = fixed.ndim
        dim = register.ndim
        size = np.array(register.shape)[::-1]
        print(fixed.shape, register.shape)
        if fixed_dim == 2:
            fixed_itk = sitk.GetImageFromArray(fixed)
            resampler = reg.initialize_resampler(fixed_itk, landmark_transform)
        if fixed_dim == 3:
            fixed_itk = sitk.GetImageFromArray(fixed[..., 0].T)
            resampler = reg.initialize_resampler(fixed_itk, landmark_transform)

        print(resampler)
        if dim == 2:
            register_itk = sitk.GetImageFromArray(register)
            deformed_itk = resampler.Execute(register_itk)

        elif dim == 3:
            slices = []
            for i in range(size[2]):
                if self.is_abort:
                    break
                QApplication.processEvents()
                img_slice = sitk.GetImageFromArray(register[i, ...])
                img_slice.SetSpacing([1, 1])
                out_slice = resampler.Execute(img_slice)
                out_slice = sitk.JoinSeries(out_slice)
                slices.append(out_slice)
                progress = i*100.0/size[2]
                self.signal_progress.emit(progress)
            stackmaker = sitk.TileImageFilter()
            stackmaker.SetLayout([1, 1, 0])
            deformed_itk = stackmaker.Execute(slices)

        if self.is_abort:
            return

        deformed = sitk.GetArrayFromImage(deformed_itk)
        return deformed


    @QtCore.pyqtSlot()
    def work(self):
        """
        Main function called by worker
        """
        fixed, is_ms_fixed = self.preprocess_image(self.fixed)
        moving, is_ms_moving = self.preprocess_image(self.moving)
        # fixed, self.points_fixed = self.crop_image(fixed, self.points_fixed)

        print(is_ms_moving)
        print(self.points_fixed, self.points_moving)
        landmark_transform = sitk.LandmarkBasedTransformInitializer(sitk.AffineTransform(2), self.points_fixed, self.points_moving)

        deformed = self.apply_registration(fixed, moving, landmark_transform)

        ref_deformed = self.moving if is_ms_moving else None
        deformed = self.postprocess_image(deformed, ref_deformed)
        print("Fixed", fixed.shape, deformed.shape)
        self.signal_end.emit(deformed)

    def abort(self):
        self.is_abort = True


class RegistrationSelectionController:
    """
    Registration with fiducials
    """
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
        self.compute_transformation()

    def set_clickable(self, clickable):
        self.imageview.setClickable(clickable)
        self.imageview2.setClickable(clickable)

    def clear_selection(self):
        self.imageview.resetCross()
        self.imageview2.resetCross()

    def compute_transformation(self):
        fixed = self.imageview.image
        moving = self.imageview2.image
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
