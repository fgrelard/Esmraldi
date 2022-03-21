import os
import sys
import webbrowser

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

import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp


from esmraldi.msimage import MSImage
from esmraldi.msimagefly import MSImageOnTheFly
from esmraldi.sparsematrix import SparseMatrix
from esmraldi.utils import msimage_for_visualization

from gui.imagehandlecontroller import ImageHandleController
from gui.peak_picking_controller import PeakPickingController
from gui.peak_picking_mean_spectrum_controller import PeakPickingMeanSpectrumController
from gui.spectraalignmentcontroller import SpectraAlignmentController
from gui.registration_selection_controller import RegistrationSelectionController
from gui.extract_channel_controller import ExtractChannelController
from gui.signal import Signal

class WorkerOpen(QObject):

    signal_start = pyqtSignal()
    signal_end = pyqtSignal(object, str)
    signal_progress = pyqtSignal(int)

    def __init__(self, path):
        """
        Parameters
        ----------
        path: str
            path to the filename
        """
        super().__init__()
        self.path = path
        self.npy_path = os.path.splitext(path)[0] + ".npy"
        self.is_abort = False

    def get_spectra(self, imzml):
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
        imzml = io.open_imzml(self.path)
        mz, I = imzml.getspectrum(0)
        spectra = self.get_spectra(imzml)

        if spectra.ndim == 2:
            sum_len = sum(len(mz) for mz, I in spectra)
        else:
            sum_len = spectra.shape[-1]
        max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
        max_y = max(imzml.coordinates, key=lambda item:item[1])[1]

        if max_x*max_y*sum_len > 1e4:
            mean_spectra = None
            if os.path.isfile(self.npy_path):
                mean_spectra = np.load(self.npy_path)
            img_data = MSImageOnTheFly(spectra, coords=imzml.coordinates, tolerance=0.003, mean_spectra=mean_spectra)
            img_data = msimage_for_visualization(img_data)
            return img_data

        full_spectra = io.get_full_spectra(imzml)
        img_data = MSImage(full_spectra, image=None, coordinates=imzml.coordinates, tolerance=0.003)
        img_data = msimage_for_visualization(img_data)
        return img_data

    def open_other_formats(self):
        try:
            im_itk = sitk.ReadImage(self.path)
        except:
            return cv2.imread(self.path)

        return sitk.GetArrayFromImage(im_itk)

    @pyqtSlot()
    def work(self):
        self.signal_start.emit()
        if self.path.lower().endswith(".imzml"):
            img_data = self.open_imzML()
            img_data.mean_spectra
            if not os.path.isfile(self.npy_path):
                np.save(self.npy_path, img_data.mean_spectra)
        else:
            img_data = self.open_other_formats()
        self.signal_end.emit(img_data, self.path)

    def abort(self):
        self.is_abort = True

class WorkerSave(QObject):

    signal_start = pyqtSignal()
    signal_end = pyqtSignal()
    signal_progress = pyqtSignal(int)

    def __init__(self, image, path):
        """
        Parameters
        ----------
        ive: ImageViewExtended
            the image view
        path: str
            path to the filename
        """
        super().__init__()
        self.image = image
        self.path = path
        self.is_abort = False

    def save_imzML(self):
        image = self.image.transpose((2, 1, 0))
        mz = self.image.spectra[:, 0]
        I, coordinates = io.get_spectra_from_images(image)
        if len(mz) != len(I):
            mz = np.tile(self.image.mzs, (len(I), 1))
        io.write_imzml(mz, I, coordinates, self.path)

    def save_other_formats(self):
        try:
            if self.image.shape[-1] <= 4:
                sitk.WriteImage(sitk.GetImageFromArray(self.image, isVector=True), self.path)
            else:
                sitk.WriteImage(sitk.GetImageFromArray(self.image), self.path)
        except:
            cv2.imwrite(self.path, self.image)

    @pyqtSlot()
    def work(self):
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
    img_data: np.ndarray
        current displayed image
    images: dict
        dictionary mapping filename to open images
    threads: list
        thread pool
    expfitcontroller: ExpFitController
        controller for expfit dialog
    nlmeanscontroller: NLMeansController
        controller for nlmeans dialog
    tpccontroller: TPCController
        controller for tpc dialog
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

        #shortcuts
        shortcut_link = QShortcut(QKeySequence('Ctrl+L'), self.mainview.parent)
        shortcut_link.activated.connect(self.link_views)

        self.config = config
        self.threads = []
        self.is_linked = False

        self.mainview.hide_run()

        # self.open_file("/mnt/d/CouplageMSI-Immunofluo/Scan rate 37° line/immunofluo.png")

        # self.open_file("/mnt/d/CouplageMSI-Immunofluo/Scan rate 37° line/synthetic.imzML")

        self.open_file("/mnt/d/CouplageMSI-Immunofluo/Scan rate 37° line/random.imzML")

        self.mainview.set_frame(self.mainview.peakpickingview)


    def open(self):
        """
        Opens Bruker directory
        """

        filename, ext = QtWidgets.QFileDialog.getOpenFileName(self.mainview.centralwidget, "Select image", self.config['default']["imzmldir"])
        if not filename:
            return

        self.config['default']["imzmldir"] = filename
        self.open_file(filename)

    def open_file(self, filename):
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
        filename, ext = QtWidgets.QFileDialog.getSaveFileName(self.mainview.centralwidget, "Select image filename", self.config['default']["imzmldir"])
        if not filename:
            return
        self.save_file(filename)

    def save_file(self, filename):
        worker = WorkerSave(image=self.imagehandlecontroller.img_data, path=filename)
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
        item = self.mainview.gridLayout.itemAtPosition(0,2)
        if number > 1:
            self.mainview.show_second_view()
        else:
            self.mainview.hide_second_view()

    def on_combo_changed(self, index, handle1, handle2):
        if index == "":
            return
        combobox1 = handle1.imagehandleview.combobox
        combobox2 = handle2.imagehandleview.combobox

        combobox1.blockSignals(True)
        combobox2.blockSignals(True)
        items = [combobox1.itemText(i) for i in range(combobox1.count())]
        combobox2.clear()
        combobox2.addItems(items)
        current_index = combobox2.findText(handle2.current_name)
        if current_index == -1:
            current_index = combobox2.findText(index)
        combobox2.setCurrentIndex(current_index)
        combobox1.blockSignals(False)
        combobox2.blockSignals(False)


    def peak_picking(self):
        def end_computation(peaks):
            imageview = self.mainview.imagehandleview.imageview
            imageview.image.peaks = peaks
            imageview.winPlot.setVisible(True)
            unique = np.unique(np.hstack(peaks))
            intensities = imageview.displayed_spectra
            mzs = imageview.tVals
            indices = np.searchsorted(mzs, unique)
            data = [unique, intensities[indices]]
            imageview.plot.setPoints(data[0], data[1], size=5, brush=pg.mkBrush("r"))
            self.mainview.peakpickingview.label_peaks.setEnabled(True)
            self.mainview.peakpickingview.label_peaks.setText(str(len(indices)) + " peaks found.")
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
        def end_computation(peaks, intensities):
            imageview = self.mainview.imagehandleview.imageview
            imageview.image.peaks = peaks
            imageview.winPlot.setVisible(True)
            imageview.plot.setPoints(peaks, intensities, size=5, brush=pg.mkBrush("r"))
            self.mainview.peakpickingmeanspectrumview.label_peaks.setEnabled(True)
            self.mainview.peakpickingmeanspectrumview.label_peaks.setText(str(len(peaks)) + " peaks found.")
            self.mainview.progressBar.setMaximum(100)
            self.mainview.hide_run()
        self.update_progressbar(0)
        self.mainview.progressBar.setMaximum(0)
        self.mainview.show_run()
        self.peakpickingmeanspectrumcontroller.worker.signal_end.connect(end_computation)
        self.sig_abort_workers.signal.connect(self.peakpickingmeanspectrumcontroller.worker.abort)
        self.peakpickingmeanspectrumcontroller.thread.start()
        self.threads.append((self.peakpickingmeanspectrumcontroller.thread, self.peakpickingmeanspectrumcontroller.worker))

    def spectra_alignment(self):
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
        self.imagehandlecontroller.choose_image("immunofluo")
        self.imagehandlecontroller2.choose_image("synthetic")

        self.mainview.show_second_view()
        self.registrationselectioncontroller.start()
        self.mainview.set_frame(self.mainview.registrationselectionview)

    def compute_registration_selection(self):
        def end_computation(fixed_cropped, registered):
            name_cropped = self.imagehandlecontroller.current_name
            new_name_cropped = "registered_" + name_cropped
            name = self.imagehandlecontroller2.current_name
            new_name = "registered_" + name
            self.end_open(fixed_cropped, new_name_cropped, first=True)
            self.end_open(registered, new_name, first=False)
            self.mainview.hide_run()
        self.mainview.show_run()
        self.registrationselectioncontroller.worker.signal_end.connect(end_computation)
        self.registrationselectioncontroller.worker.signal_progress.connect(self.update_progressbar)
        self.sig_abort_workers.signal.connect(self.registrationselectioncontroller.worker.abort)
        self.registrationselectioncontroller.thread.start()
        self.threads.append((self.registrationselectioncontroller.thread, self.registrationselectioncontroller.worker))

    def extract_channels(self):
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


    def link_views(self):
        self.is_linked = not self.is_linked
        iview1 = self.imagehandlecontroller.imagehandleview.imageview
        iview2 = self.imagehandlecontroller2.imagehandleview.imageview
        if self.is_linked:
            iview1.view.setXLink(iview2.view)
            iview1.view.setYLink(iview2.view)
        else:
            iview1.view.setXLink(None)
            iview1.view.setYLink(None)

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
