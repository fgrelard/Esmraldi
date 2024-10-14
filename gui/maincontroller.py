import os
import sys

import numpy as np
import qtawesome as qta
import pyqtgraph as pg

from PyQt5 import Qt, QtWidgets, QtCore
from PyQt5.Qt import QVBoxLayout
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QFrame, QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import SimpleITK as sitk
import cv2
import pandas as pd
import skimage.color as color

import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp

from esmraldi.msimage import MSImage
from esmraldi.msimagefly import MSImageOnTheFly
from esmraldi.msimageimpl import MSImageImplementation
from esmraldi.sparsematrix import SparseMatrix
from esmraldi.utils import msimage_for_visualization,  indices_search_sorted

from gui.imagehandlecontroller import ImageHandleController
from gui.peak_picking_controller import PeakPickingController
from gui.peak_picking_mean_spectrum_controller import PeakPickingMeanSpectrumController
from gui.spectraalignmentcontroller import SpectraAlignmentController
from gui.registration_selection_controller import RegistrationSelectionController
from gui.extract_channel_controller import ExtractChannelController
from gui.thresholding_controller import ThresholdingController
from gui.signal import Signal

class WorkerOpen(QObject):
    """
    Class to open a file in a different
    thread
    """
    signal_start = pyqtSignal()
    signal_end = pyqtSignal(object, str)
    signal_progress = pyqtSignal(int)

    def __init__(self, path):
        """
        Parameters
        ----------
        path: str
            path to the filename
        npy_path: str
            path to mean spectra
        npy_indexing_path: str
            path to indexes to improve ion image
            display speed
        """
        super().__init__()
        self.path = path
        root = os.path.splitext(path)[0]
        self.npy_path = root + ".npy"
        self.npy_indexing_path = root + "_indexing.npy"
        self.is_abort = False

    def get_spectra(self, imzml):
        """
        Get spectra from imzML

        Returns
        ----------
        np.ndarray
            n_pixels * 2 (m/z, intensities) * values array
        """
        spectra = []
        coordinates = imzml.coordinates
        length = len(coordinates)
        for i, (x, y, z) in enumerate(coordinates):
            if self.is_abort:
                break
            QApplication.processEvents()

            mz, ints = imzml.getspectrum(i)
            spectra.append([mz, ints])

            progress = float(i/length*100)
            self.signal_progress.emit(progress)

        if spectra and not all(len(l[0]) == len(spectra[0][0]) for l in spectra):
            return np.array(spectra, dtype=object)
        return np.array(spectra)

    def open_imzML(self):
        """
        Opening imzML file

        Returns
        ----------
        MSImageBase
            Returns on-the-fly or dense MS image
            depending on its size (>10Gb or <10Gb, respectively)

        """
        imzml = io.open_imzml(self.path)
        mz, I = imzml.getspectrum(0)
        spectra = self.get_spectra(imzml)

        if spectra.ndim == 2:
            sum_len = sum(len(mz) for mz, I in spectra)
        else:
            sum_len = spectra.shape[-1]
        max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
        max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
        max_z = max(imzml.coordinates, key=lambda item:item[2])[2]

        if max_x*max_y*sum_len > 1e10:
            print("On the fly")
            mean_spectra = None
            indexing = None
            if os.path.isfile(self.npy_path):
                mean_spectra = np.load(self.npy_path)
            if os.path.isfile(self.npy_indexing_path):
                indexing = np.load(self.npy_indexing_path, mmap_mode="r")

            img_data = MSImageOnTheFly(spectra, coords=imzml.coordinates, tolerance=14, mean_spectra=mean_spectra, indexing=indexing)

            img_data = msimage_for_visualization(img_data)
            return img_data

        full_spectra = io.get_full_spectra(imzml, spectra)
        img_data = MSImage(full_spectra, image=None, coordinates=imzml.coordinates, tolerance=14)
        img_data = msimage_for_visualization(img_data)
        return img_data

    def open_other_formats(self):
        """
        Open other imaging formats
        """
        mzs = np.array([])
        if self.path.endswith("tif"):
            return io.open_tif(self.path)
        try:
            im_itk = sitk.ReadImage(self.path)
        except:
            return cv2.imread(self.path), mzs
        return sitk.GetArrayFromImage(im_itk), mzs

    @pyqtSlot()
    def work(self):
        """
        Actual function run when starting worker
        """
        self.signal_start.emit()
        if self.path.lower().endswith(".imzml"):
            img_data = self.open_imzML()
            img_data.mean_spectra
            if not os.path.isfile(self.npy_path):
                np.save(self.npy_path, img_data.mean_spectra)
            if img_data.indexing is not None:
                if not os.path.isfile(self.npy_indexing_path):
                    np.save(self.npy_indexing_path, img_data.indexing)
                img_data.indexing = np.load(self.npy_indexing_path, mmap_mode="r")
        else:
            images, mzs = self.open_other_formats()
            if mzs.size:
                images = images.T
                intensities, _ = io.get_spectra_from_images(images, full=True)
                intensities = np.array(intensities)
                all_mzs = np.tile(mzs, (np.prod(images.shape[:-1]), 1))
                spectra = np.stack([all_mzs, intensities], axis=1)
                img_data = MSImageImplementation(spectra, images, mzs, tolerance=14)
                img_data = msimage_for_visualization(img_data)
            else:
                img_data = images

        self.signal_end.emit(img_data, self.path)

    def abort(self):
        self.is_abort = True

class WorkerSave(QObject):
    """
    Worker to save an image
    """

    signal_start = pyqtSignal()
    signal_end = pyqtSignal()
    signal_progress = pyqtSignal(int)

    def __init__(self, image, path):
        """
        Parameters
        ----------
        image: MSImageBase
            the image to save
        path: str
            path to the filename
        """
        super().__init__()
        self.image = image
        self.path = path
        self.is_abort = False

    def save_imzML(self):
        """
        Save imzML by collecting mz, intensities and
        coordinates from an image
        """
        image = self.image.transpose((2, 1, 0))
        mz = self.image.spectra[:, 0]
        I, coordinates = io.get_spectra_from_images(image)
        if len(mz) != len(I):
            mz = np.tile(self.image.mzs, (len(I), 1))
        io.write_imzml(mz, I, coordinates, self.path)

    def save_other_formats(self):
        """
        Save image as other format
        """
        try:
            if len(self.image.shape) >= 3:
                if self.image.shape[-1] <= 4:
                    sitk.WriteImage(sitk.GetImageFromArray(self.image, isVector=True), self.path)
                else:
                    root, ext = os.path.splitext(self.path)
                    io.to_csv(self.image.mzs, root + ".csv")
                    if ext == ".tif":
                        io.to_tif(self.image.image.astype(np.float32), self.image.mzs, self.path)
                    else:
                        sitk.WriteImage(sitk.GetImageFromArray(self.image.image.astype(np.float32)), self.path)
            else:
                sitk.WriteImage(sitk.GetImageFromArray(self.image), self.path)
        except:
            cv2.imwrite(self.path, self.image)

    @pyqtSlot()
    def work(self):
        """
        Main function run when worker is started
        """
        self.signal_start.emit()
        if self.path.lower().endswith(".imzml"):
            self.save_imzML()
        else:
            self.save_other_formats()
        self.signal_end.emit()

    def abort(self):
        self.is_abort = True

class MainController:
    """
    Main controller connecting all controllers and events

    Attributes
    ----------
    app: QApplication
        application
    mainview: AppWindow
        QMainWindow
    config: configparser.ConfigParser
        configuration file
    """
    def __init__(self, app, mainview, config):
        mainview.closeEvent = self.exit_app
        self.mainview = mainview.ui
        self.mainview.parent = mainview
        self.app = app

        self.sig_abort_workers = Signal()

        self.mainview.actionExit.triggered.connect(self.exit_app)
        self.mainview.actionOpen.triggered.connect(self.open)
        self.mainview.actionSave.triggered.connect(self.save)


        self.imagehandlecontroller = ImageHandleController(self.mainview.imagehandleview)
        self.imagehandlecontroller2 = ImageHandleController(self.mainview.imagehandleview2)

        handle1 = self.imagehandlecontroller
        handle2 = self.imagehandlecontroller2
        combobox1 = handle1.imagehandleview.combobox
        combobox2 = handle2.imagehandleview.combobox
        combobox1.currentIndexChanged[str].connect(lambda i: self.on_combo_changed(i, handle1, handle2))
        combobox2.currentIndexChanged[str].connect(lambda i: self.on_combo_changed(i, handle2, handle1))

        self.mainview.oneViewButton.clicked.connect(lambda event: self.update_number_view(1))

        self.mainview.twoViewButton.clicked.connect(lambda event: self.update_number_view(2))

        self.mainview.stopButton.clicked.connect(self.abort_computation)

        imageview = self.mainview.imagehandleview.imageview
        imageview2 = self.mainview.imagehandleview2.imageview
        imageview.signal_progress_export.connect(self.update_progressbar)
        imageview.signal_start_export.connect(self.mainview.show_run)
        imageview.signal_end_export.connect(self.mainview.hide_run)
        self.mainview.actionPeakPicking.triggered.connect(lambda event: self.mainview.set_frame(self.mainview.peakpickingview))
        self.peakpickingcontroller = PeakPickingController(self.mainview.peakpickingview, imageview)
        self.peakpickingcontroller.trigger_compute.signal.connect(self.peak_picking)
        self.peakpickingcontroller.trigger_end.signal.connect(self.mainview.clear_frame)


        self.mainview.actionPeakPickingMeanSpectrum.triggered.connect(lambda event: self.mainview.set_frame(self.mainview.peakpickingmeanspectrumview))
        self.peakpickingmeanspectrumcontroller = PeakPickingMeanSpectrumController(self.mainview.peakpickingmeanspectrumview, imageview)
        self.peakpickingmeanspectrumcontroller.trigger_compute.signal.connect(self.peak_picking_mean_spectrum)
        self.peakpickingcontroller.trigger_end.signal.connect(self.mainview.clear_frame)

        self.mainview.actionPeakMetaspace.triggered.connect(self.peaks_metaspace)
        self.mainview.actionClearPeaks.triggered.connect(self.clear_peaks)


        self.mainview.actionSpectraAlignment.triggered.connect(lambda event: self.mainview.set_frame(self.mainview.spectraalignmentview))
        self.spectraalignmentcontroller = SpectraAlignmentController(self.mainview.spectraalignmentview, imageview)
        self.spectraalignmentcontroller.trigger_compute.signal.connect(self.spectra_alignment)
        self.spectraalignmentcontroller.trigger_end.signal.connect(self.mainview.clear_frame)


        self.mainview.actionRegistrationSelection.triggered.connect(self.start_registration_selection)
        self.registrationselectioncontroller = RegistrationSelectionController(self.mainview.registrationselectionview, imageview, imageview2)
        self.registrationselectioncontroller.trigger_compute.signal.connect(self.compute_registration_selection)
        self.registrationselectioncontroller.trigger_end.signal.connect(self.mainview.clear_frame)


        self.mainview.actionExtractChannel.triggered.connect(lambda event: self.mainview.set_frame(self.mainview.extractchannelview))
        self.extractchannelcontroller = ExtractChannelController(self.mainview.extractchannelview, imageview)
        self.extractchannelcontroller.trigger_compute.signal.connect(self.extract_channels)
        self.extractchannelcontroller.trigger_end.signal.connect(self.mainview.clear_frame)

        self.mainview.actionThresholding.triggered.connect(lambda event: self.mainview.set_frame(self.mainview.thresholdingview))
        self.thresholdingcontroller = ThresholdingController(self.mainview.thresholdingview, self.imagehandlecontroller.imageview, self.mainview.rangeSliderThreshold)
        self.thresholdingcontroller.trigger_compute.signal.connect(self.manual_thresholding)
        self.thresholdingcontroller.trigger_end.signal.connect(self.mainview.clear_frame)


        self.mainview.actionNewMask.triggered.connect(self.new_mask)
        self.mainview.actionMaskAdd.triggered.connect(self.mask_add)

        #shortcuts
        shortcut_link = QShortcut(QKeySequence('Ctrl+L'), self.mainview.parent)
        shortcut_link.activated.connect(self.link_views)

        shortcut_link = QShortcut(QKeySequence('Ctrl+O'), self.mainview.parent)
        shortcut_link.activated.connect(self.open)

        shortcut_link = QShortcut(QKeySequence('Ctrl+S'), self.mainview.parent)
        shortcut_link.activated.connect(self.save)

        self.config = config
        self.threads = []
        self.is_linked = False

        self.mainview.hide_run()

        # self.open_file("/mnt/d/CouplageMSI-Immunofluo/Scan rate 37° line/immunofluo.png")

        # self.open_file("/mnt/d/CouplageMSI-Immunofluo/Scan rate 37° line/synthetic.imzML")

        # self.open_file("/mnt/d/CouplageMSI-Immunofluo/Scan rate 37° line/random.imzML")
        # self.open_file("/mnt/d/CouplageMSI-Immunofluo/Scan rate 37° line/test.tif")
        # self.open_file("/mnt/d/CouplageMSI-Immunofluo/Scan rate 37° line/msi_gray.png")
        # self.open_file("/mnt/d/CBMN/random.imzML")

        self.mainview.set_frame(self.mainview.peakpickingview)


    def open(self):
        """
        Main function to open a file (GUI + core)
        """

        filenames, ext = QtWidgets.QFileDialog.getOpenFileNames(self.mainview.centralwidget, "Select image", self.config['default']["imzmldir"])
        for filename in filenames:
            if not filename:
                return

            self.config['default']["imzmldir"] = filename
            self.open_file(filename)

    def open_file(self, filename):
        """
        Core function to open a file
        """
        worker = WorkerOpen(path=filename)
        thread = QThread()
        worker.moveToThread(thread)
        worker.signal_start.connect(self.mainview.show_run)
        worker.signal_end.connect(self.end_open)
        worker.signal_progress.connect(self.update_progressbar)

        self.update_progressbar(0)
        self.mainview.progressBar.setMaximum(100)
        self.sig_abort_workers.signal.connect(worker.abort)
        thread.started.connect(worker.work)
        thread.start()

        self.threads.append((thread, worker))


    def save(self):
        """
        Main function to save a file (GUI + core)
        """
        widget = self.mainview.gridLayout.itemAtPosition(0, 2).widget()
        index = 0
        if widget.isVisible():
            name1 = self.imagehandlecontroller.current_name
            name2 = self.imagehandlecontroller2.current_name
            items = [name1, name2]
            item, ok = QtWidgets.QInputDialog.getItem(self.mainview.centralwidget, "Select image to save", "Image to save", items, 0, editable=False)
            index = items.index(item)
        filename, ext = QtWidgets.QFileDialog.getSaveFileName(self.mainview.centralwidget, "Select image filename", self.config['default']["imzmldir"])
        if not filename:
            return
        self.save_file(filename, index)

    def save_file(self, filename, index=0):
        """
        Core function to save a file
        """
        data = self.imagehandlecontroller.img_data
        if index == 1:
            data = self.imagehandlecontroller2.img_data
        worker = WorkerSave(image=data, path=filename)
        thread = QThread()
        worker.moveToThread(thread)
        worker.signal_start.connect(self.mainview.show_run)
        worker.signal_end.connect(lambda:self.mainview.hide_run())
        worker.signal_progress.connect(self.update_progressbar)

        self.update_progressbar(0)
        self.mainview.progressBar.setMaximum(0)
        self.sig_abort_workers.signal.connect(worker.abort)
        thread.started.connect(worker.work)
        thread.start()

        self.threads.append((thread, worker))


    def exit_app(self, ev=None):
        """
        Exits the app and save configuration
        preferences
        """
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        self.mainview.imagehandleview.imageview.winPlotROI.close()
        self.mainview.imagehandleview2.imageview.winPlotROI.close()
        self.app.quit()


    def end_open(self, image, filename, first=None):
        """
        First: boolean whether the image is to be displayed in the first pane or the second.
        """
        self.mainview.hide_run()
        if first == True or first is None:
            self.imagehandlecontroller.image_to_view(image, filename)
        elif first == False or first is None:
            self.imagehandlecontroller2.image_to_view(image, filename)
        self.mainview.progressBar.setMaximum(100)


    def update_number_view(self, number=1):
        """
        Updating the number of view (1 or 2)
        """
        if number > 1:
            self.mainview.show_second_view()
        else:
            self.mainview.hide_second_view()

    def on_combo_changed(self, index, handle1, handle2):
        """
        Updating image when changing combobox value
        """
        def set_combo(combo_ref, combo_target):
            items = [combo_ref.itemText(i) for i in range(combo_ref.count())]
            combo_target.clear()
            combo_target.addItems(items)
            current_index = combo_target.findText(handle2.current_name)
            if current_index == -1:
                current_index = combo_target.findText(index)
            combo_target.setCurrentIndex(current_index)

        if index == "":
            return
        combobox1 = handle1.imagehandleview.combobox
        combobox2 = handle2.imagehandleview.combobox
        comboboxRoi1 = handle1.imagehandleview.imageview.ui.comboRoiImage
        comboboxRoi2 = handle2.imagehandleview.imageview.ui.comboRoiImage

        combobox1.blockSignals(True)
        combobox2.blockSignals(True)
        comboboxRoi1.blockSignals(True)
        comboboxRoi2.blockSignals(True)

        set_combo(combobox1, combobox2)
        set_combo(combobox1, comboboxRoi1)
        set_combo(combobox1, comboboxRoi2)

        combobox1.blockSignals(False)
        combobox2.blockSignals(False)
        comboboxRoi1.blockSignals(False)
        comboboxRoi2.blockSignals(False)


    def display_peaks_mean_spectrum(self, peaks):
        """
        Display peaks in the mean spectrum
        """
        imageview = self.mainview.imagehandleview.imageview
        imageview.winPlot.setVisible(True)
        unique = np.unique(np.hstack(peaks) if peaks.size else peaks)
        intensities = imageview.displayed_spectra
        mzs = imageview.tVals
        indices = indices_search_sorted(unique, mzs)
        data = [unique, intensities[indices]]
        imageview.plot.setPoints(data[0], data[1], size=5, brush=pg.mkBrush("r"))

    def peak_picking(self):
        """
        Peak picking in each spectrum
        Can be very slow
        """
        def end_computation(peaks):
            imageview = self.mainview.imagehandleview.imageview
            imageview.image.peaks = peaks
            self.display_peaks_mean_spectrum(peaks)
            self.mainview.peakpickingview.label_peaks.setEnabled(True)
            self.mainview.progressBar.setMaximum(100)
            self.mainview.hide_run()
        self.update_progressbar(0)
        self.mainview.progressBar.setMaximum(0)
        self.mainview.show_run()
        self.peakpickingcontroller.worker.signal_end.connect(end_computation)
        self.sig_abort_workers.signal.connect(self.peakpickingcontroller.worker.abort)
        self.peakpickingcontroller.thread.start()
        self.threads.append((self.peakpickingcontroller.thread, self.peakpickingcontroller.worker))

    def peak_picking_mean_spectrum(self):
        """
        Peak picking from mean spectrum
        """
        def end_computation(peaks, intensities):
            imageview = self.mainview.imagehandleview.imageview
            imageview.image.peaks = peaks
            imageview.winPlot.setVisible(True)
            imageview.plot.setPoints(peaks, intensities, size=5, brush=pg.mkBrush("r"))
            self.mainview.peakpickingmeanspectrumview.label_peaks.setEnabled(True)
            self.mainview.progressBar.setMaximum(100)
            self.mainview.hide_run()
        self.update_progressbar(0)
        self.mainview.progressBar.setMaximum(0)
        self.mainview.show_run()
        self.peakpickingmeanspectrumcontroller.worker.signal_end.connect(end_computation)
        self.sig_abort_workers.signal.connect(self.peakpickingmeanspectrumcontroller.worker.abort)
        self.peakpickingmeanspectrumcontroller.thread.start()
        self.threads.append((self.peakpickingmeanspectrumcontroller.thread, self.peakpickingmeanspectrumcontroller.worker))

    def peaks_metaspace(self):
        """
        Add peaks from METASPACE
        """
        filenames, ext = QtWidgets.QFileDialog.getOpenFileNames(self.mainview.centralwidget, "Select METASPACE .csv file", self.config['default']["imzmldir"])
        for filename in filenames:
            if not filename:
                return

            data = pd.read_csv(filename, header=2, delimiter=",")
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setWindowTitle("Peaks from METASPACE (.csv)")
            if hasattr(data, "mz"):
                mzs_annotated = data.mz
                mzs_annotated = np.unique(mzs_annotated)
            else:
                mzs_annotated = np.loadtxt(filename)

            if mzs_annotated.dtype == float:
                imageview = self.mainview.imagehandleview.imageview
                if imageview.image.peaks is not None:
                    imageview.image.peaks = np.concatenate((imageview.image.peaks, mzs_annotated))
                else:
                    imageview.image.peaks = mzs_annotated

                groups = sp.index_groups_start_end(imageview.image.peaks, imageview.image.tolerance, is_ppm=imageview.image.is_ppm)
                before_len = len(imageview.image.peaks)
                imageview.image.peaks = np.array([np.median(g) for g in groups])
                after_len = len(imageview.image.peaks)
                added_len = len(mzs_annotated) - (before_len - after_len)
                self.display_peaks_mean_spectrum(imageview.image.peaks)
                msg.setText("Added " + str(added_len) + " peaks from METASPACE")
            else:
                msg.setText("Not a valid METASPACE file.")
                msg.setInformativeText("Please supply a .csv file exported from METASPACE")
            msg.exec_()

    def clear_peaks(self):
        """
        Remove all peaks
        """
        imageview = self.mainview.imagehandleview.imageview
        imageview.image.peaks = np.array([])
        self.display_peaks_mean_spectrum(imageview.image.peaks)

    def spectra_alignment(self):
        """
        Align spectra to have a common m/z axis
        """
        def end_computation(image):
            name = self.imagehandlecontroller.current_name
            new_name = "aligned_" + name
            self.end_open(image, new_name, first=True)
            self.mainview.progressBar.setMaximum(100)
            self.mainview.hide_run()
        self.update_progressbar(0)
        self.mainview.progressBar.setMaximum(0)
        self.mainview.show_run()
        self.spectraalignmentcontroller.worker.signal_end.connect(end_computation)
        self.sig_abort_workers.signal.connect(self.spectraalignmentcontroller.worker.abort)
        self.spectraalignmentcontroller.thread.start()
        self.threads.append((self.spectraalignmentcontroller.thread, self.spectraalignmentcontroller.worker))

    def start_registration_selection(self, event):
        """
        Allows to add markers in the image
        for fiducial-based registration
        """
        self.mainview.show_second_view()
        self.registrationselectioncontroller.set_clickable(True)
        self.mainview.set_frame(self.mainview.registrationselectionview)

    def compute_registration_selection(self):
        """
        Compute the transform from fiducial markers
        """
        def end_computation(registered):
            name = self.imagehandlecontroller2.current_name
            new_name = "registered_" + name
            self.end_open(registered, new_name, first=False)
            self.mainview.hide_run()
        self.mainview.show_run()
        self.registrationselectioncontroller.worker.signal_end.connect(end_computation)
        self.registrationselectioncontroller.worker.signal_progress.connect(self.update_progressbar)
        self.sig_abort_workers.signal.connect(self.registrationselectioncontroller.worker.abort)
        self.registrationselectioncontroller.thread.start()
        self.threads.append((self.registrationselectioncontroller.thread, self.registrationselectioncontroller.worker))

    def extract_channels(self):
        """
        Extract a channel in a multichannel image
        by its number
        """
        def end_computation(image, number):
            name = self.imagehandlecontroller.current_name
            new_name = "channel_" + str(number) + "_" + name
            if image is not None:
                self.end_open(image, new_name, first=True)
            self.mainview.hide_run()
        self.mainview.show_run()
        self.extractchannelcontroller.worker.signal_end.connect(end_computation)
        self.sig_abort_workers.signal.connect(self.extractchannelcontroller.worker.abort)
        self.extractchannelcontroller.thread.start()
        self.threads.append((self.extractchannelcontroller.thread, self.extractchannelcontroller.worker))

    def manual_thresholding(self):
        """
        Threshold an image by selecting values
        """
        def end_computation(image):
            name = self.imagehandlecontroller.current_name
            new_name = "threshold_" + name
            if image is not None:
                self.end_open(image, new_name, first=True)
        self.thresholdingcontroller.worker.signal_end.connect(end_computation)
        self.sig_abort_workers.signal.connect(self.thresholdingcontroller.worker.abort)
        self.thresholdingcontroller.thread.start()
        self.threads.append((self.thresholdingcontroller.thread, self.thresholdingcontroller.worker))

    def new_mask(self):
        """
        Create a new blank image the same dimensions
        as current image
        """
        iview = self.imagehandlecontroller.imagehandleview.imageview
        iview2 = self.imagehandlecontroller2.imagehandleview.imageview
        image = np.zeros_like(iview.current_image, dtype=np.uint8)
        self.end_open(image, "Mask", first=False)
        self.mainview.show_second_view()


    def mask_add(self):
        """
        Add current selected ROI to mask
        """
        iview2 = self.imagehandlecontroller2.imagehandleview.imageview
        name = self.imagehandlecontroller2.current_name
        if name != "Mask":
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setWindowTitle("Adding ROI to mask")
            msg.setText("No mask image created.")
            msg.setInformativeText("Please go to Segmentation > New mask")
            msg.exec_()
        else:
            iview1 = self.imagehandlecontroller.imagehandleview.imageview
            colmaj = iview1.imageItem.axisOrder == "col-major"
            coords = iview1.coords_roi
            if not colmaj:
                image = iview2.current_image.T
            image[tuple(coords)] = image.max()+1
            iview2.updateImage()

    def link_views(self):
        """
        Link the display between the two image views
        This means the cursor at one position in an image will
        be at the same position in another image
        """
        def disconnect(signal, oldhandler):
            try:
                while True:
                    signal.disconnect(oldhandler)
            except TypeError:
                pass

        self.is_linked = not self.is_linked
        iview1 = self.imagehandlecontroller.imagehandleview.imageview
        iview2 = self.imagehandlecontroller2.imagehandleview.imageview
        iview1.is_linked = self.is_linked
        iview2.is_linked = self.is_linked
        iview1.setFocus(None, iview1.is_focused)
        iview2.setFocus(None, iview2.is_focused)
        if self.is_linked:
            iview1.view.setXLink(iview2.view)
            iview1.view.setYLink(iview2.view)
            iview2.scene.sigMouseMoved.connect(iview1.on_hover_image)
            iview1.scene.sigMouseMoved.connect(iview2.on_hover_image)
        else:
            iview1.view.setXLink(None)
            iview1.view.setYLink(None)
            disconnect(iview1.scene.sigMouseMoved, iview2.on_hover_image)
            disconnect(iview2.scene.sigMouseMoved, iview1.on_hover_image)

    def abort_computation(self):
        """
        Stops any computation in progress
        Hides the progress bar and stop button
        """
        self.sig_abort_workers.signal.emit()
        imageview = self.mainview.imagehandleview.imageview
        imageview.signal_abort.emit()

        imageview2 = self.mainview.imagehandleview2.imageview
        imageview2.signal_abort.emit()

        all_threads = self.threads + imageview.threads + imageview2.threads
        for thread, worker in all_threads:
            thread.quit()
            thread.wait()
        self.mainview.hide_run()


    def update_progressbar(self, progress):
        """
        Updates the progress bar each time
        An iteration in a controller has passed

        Parameters
        ----------
        progress: int
            progress value (/100)

        """
        self.mainview.progressBar.setValue(progress)
