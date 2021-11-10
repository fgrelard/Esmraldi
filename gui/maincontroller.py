import os
import sys
import webbrowser

import numpy as np
import qtawesome as qta

from PyQt5 import Qt, QtWidgets, QtCore
from PyQt5.Qt import QVBoxLayout
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QFrame
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import SimpleITK as sitk

import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp


from esmraldi.msimage import MSImage, MSImageImplementation
from esmraldi.sparsematrix import SparseMatrix

from gui.imagehandlecontroller import ImageHandleController
from gui.registration_selection_controller import RegistrationSelectionController
from gui.signal import Signal

class WorkerOpen(QObject):

    signal_start = pyqtSignal()
    signal_end = pyqtSignal(object, str)
    signal_progress = pyqtSignal(int)

    def __init__(self, path):
        """
        Parameters
        ----------
        ive: ImageViewExtended
            the image view
        path: str
            path to the filename
        """
        super().__init__()
        self.path = path
        self.is_abort = False

    def open_imzML(self):
        imzml = io.open_imzml(self.path)
        mz, I = imzml.getspectrum(0)
        spectra = io.get_full_spectra(imzml)
        img_data = MSImage(spectra, image=None, coordinates=imzml.coordinates, tolerance=0.003)
        img_data.is_maybe_densify = True
        img_data.spectral_axis = 0
        new_order = (2, 1, 0)
        img_data = img_data.transpose(new_order)
        return img_data

    def open_other_formats(self):
        im_itk = sitk.ReadImage(self.path)
        return sitk.GetArrayFromImage(im_itk)

    @pyqtSlot()
    def work(self):
        self.signal_start.emit()
        if self.path.lower().endswith(".imzml"):
            img_data = self.open_imzML()
            img_data.compute_mean_spectra()
        else:
            img_data = self.open_other_formats()
        self.signal_end.emit(img_data, self.path)

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

        self.mainview.actionRegistrationSelection.triggered.connect(self.start_registration_selection)

        self.registrationselectioncontroller = RegistrationSelectionController(self.mainview.registrationselectionview, imageview, imageview2)
        self.registrationselectioncontroller.trigger_compute.signal.connect(self.compute_registration_selection)
        self.registrationselectioncontroller.trigger_end.signal.connect(self.mainview.clear_frame)


        self.config = config
        self.threads = []

        self.mainview.hide_run()


        self.open_file("/mnt/d/CouplageMSI-Immunofluo/test_registration/reference.png")
        self.open_file("/mnt/d/CouplageMSI-Immunofluo/test_registration/target.png")


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
        self.mainview.progressBar.setMaximum(0)
        self.sig_abort_workers.signal.connect(worker.abort)
        thread.started.connect(worker.work)
        thread.start()

        self.threads.append((thread, worker))


    def save(self):
        pass


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


    def end_open(self, image, filename):
        self.mainview.hide_run()
        self.imagehandlecontroller.image_to_view(image, filename)
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

    def start_registration_selection(self, event):
        self.imagehandlecontroller.choose_image("reference")

        self.mainview.show_second_view()
        self.registrationselectioncontroller.start()
        self.mainview.set_frame(self.mainview.registrationselectionview)

    def compute_registration_selection(self):
        def end_computation(registered):
            name = self.imagehandlecontroller2.current_name
            new_name = "registered_" + name
            self.end_open(registered, new_name)
        self.registrationselectioncontroller.worker.signal_end.connect(end_computation)
        self.registrationselectioncontroller.thread.start()
        self.threads.append((self.registrationselectioncontroller.thread, self.registrationselectioncontroller.worker))

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

        for thread, worker in self.threads:
            thread.quit()
            thread.wait()
        for thread, worker in imageview.threads:
            thread.quit()
            thread.wait()
        for thread, worker in imageview2.threads:
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
