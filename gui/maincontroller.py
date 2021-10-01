import os
import sys
import webbrowser

import numpy as np
import qtawesome as qta

from PyQt5 import Qt, QtWidgets
from PyQt5.Qt import QVBoxLayout
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem
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

class Signal(QObject):
    """
    Class wrapping a PyQt5 signal

    Attributes
    ----------
    signal: pyqtSignal
        the signal
    """
    signal = pyqtSignal()

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

        self.locale = Qt.QLocale(Qt.QLocale.English)
        Qt.QLocale.setDefault(self.locale)

        self.mainview = mainview.ui
        self.mainview.parent = mainview
        self.app = app

        self.sig_abort_workers = Signal()

        self.mainview.actionExit.triggered.connect(self.exit_app)
        self.mainview.actionOpen.triggered.connect(self.open)
        self.mainview.actionSave.triggered.connect(self.save)


        self.mainview.lineEdit.textEdited.connect(self.change_mz_value)
        self.mainview.lineEdit.returnPressed.connect(self.update_mz_value)

        self.mainview.lineEditTol.textEdited.connect(self.change_tolerance)
        self.mainview.lineEditTol.returnPressed.connect(self.update_tolerance)

        self.mainview.stopButton.clicked.connect(self.abort_computation)
        self.mainview.combobox.activated[str].connect(self.choose_image)
        self.mainview.trashButton.clicked.connect(lambda : self.remove_image(self.current_name(self.img_data), manual=True))

        self.mainview.editButton.clicked.connect(self.edit_name)

        self.mainview.imageview.scene.sigMouseClicked.connect(self.on_click_image)

        self.mainview.imageview.signal_progress_export.connect(self.update_progressbar)
        self.mainview.imageview.signal_start_export.connect(self.mainview.show_run)
        self.mainview.imageview.signal_end_export.connect(self.mainview.hide_run)
        self.mainview.imageview.signal_image_change.connect(self.change_image_combobox)
        self.mainview.imageview.signal_mz_change.connect(lambda mz: self.mainview.lineEdit.setText("{:.4f}".format(mz)))


        self.is_edit = False

        self.is_text_editing = False

        self.current_mz = 1.0
        self.tolerance = 0.003


        self.app.aboutToQuit.connect(self.exit_app)
        self.config = config
        self.images = OrderedDict()
        self.metadata = OrderedDict()
        self.mainview.hide_run()

        self.threads = []

        self.img_data = None

        self.mouse_x = 0
        self.mouse_y = 0
        self.z = 0

        mzs = np.arange(1000)
        x = np.random.random((10, 100, 1000))
        x[x < 0.9] = 0  # fill most of the array with zeros
        x_r = x.reshape((np.prod(x.shape[-1]), x.shape[-1]))
        spectra = np.stack((np.tile(mzs, (1000,1)), x_r), axis=1)
        sm = SparseMatrix(x, is_maybe_densify=False)
        mzs = SparseMatrix(spectra, is_maybe_densify=False)
        mss = MSImage(spectra, sm, tolerance=0.0003)
        mss.spectral_axis = 0
        mss = mss.transpose((2,1,0))
        self.img_data = mss
        self.add_image(self.img_data, "test")
        self.choose_image("test")
        self.filename = "test"


    def image_to_view(self, filename):
        name = os.path.basename(filename)
        name = os.path.splitext(name)[0]
        self.mainview.combobox.setCurrentIndex(self.mainview.combobox.findText(name))
        self.add_image(self.img_data, name)
        self.choose_image(name)
        self.filename = name

    def open(self):
        """
        Opens Bruker directory
        """

        filename, ext = QtWidgets.QFileDialog.getOpenFileName(self.mainview.centralwidget, "Select image", self.config['default']["imzmldir"])
        if not filename:
            return

        self.config['default']["imzmldir"] = filename

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


    def get_image(self):
        return self.img_data

    def exit_app(self):
        """
        Exits the app and save configuration
        preferences
        """
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        self.app.quit()

    def change_mz_value(self, text):
        self.is_text_editing = True
        number, is_converted = self.locale.toDouble(text)
        if is_converted:
            self.current_mz = number

    def update_mz_value(self):
        self.is_text_editing = False
        try:
            ind = (np.abs(self.mainview.imageview.tVals - self.current_mz)).argmin()
            self.mainview.imageview.setCurrentIndex(ind)
        except Exception as e:
            pass

    def change_tolerance(self, text):
        self.is_text_editing = True
        number, is_converted = self.locale.toDouble(text)
        if is_converted:
            self.tolerance = number

    def update_tolerance(self):
        self.is_text_editing = False
        try:
            self.mainview.imageview.imageDisp.tolerance = self.tolerance
            self.mainview.imageview.updateImage()
        except Exception as e:
            print("error", e)


    def end_open(self, image, filename):
        self.mainview.hide_run()
        self.img_data = image
        self.image_to_view(filename)
        self.mainview.progressBar.setMaximum(100)



    def abort_computation(self):
        """
        Stops any computation in progress
        Hides the progress bar and stop button
        """
        self.sig_abort_workers.signal.emit()
        self.mainview.imageview.signal_abort.emit()
        for thread, worker in self.threads:
            thread.quit()
            thread.wait()
        for thread, worker in self.mainview.imageview.threads:
            thread.quit()
            thread.wait()
        self.mainview.hide_run()

    def add_image(self, image, name):
        """
        Adds an image to the combobox
        and to the self.images dictionary

        Parameters
        ----------
        image: np.ndarray
            the image
        name: str
            combobox name
        """
        self.mainview.combobox.addItem(name)
        image_with_metadata = image
        self.images[name] = image_with_metadata
        img_data_name = self.current_name(self.img_data)
        self.metadata[name] = self.metadata[img_data_name] if img_data_name in self.metadata else None

    def current_name(self, image):
        list_keys = list(self.images.keys())
        list_values = list(self.images.values())
        try:
            key = [np.all(image == array) for array in list_values].index(True)
        except Exception as e:
            key = -1
        if len(list_keys) > 0:
            img_data_name = list_keys[key]
        else:
            img_data_name = "No image"
        return img_data_name

    def edit_name(self):
        self.is_edit = not self.is_edit
        if self.is_edit:
            fa_check = qta.icon('fa.check', color="green")
            self.mainview.editButton.setIcon(fa_check)
        else:
            self.mainview.combobox.update()
            old_name = self.current_name(self.img_data)
            new_name = self.mainview.combobox.currentText()
            if old_name != "No image":
                self.change_name(old_name, new_name)
                self.mainview.combobox.clear()
                self.mainview.combobox.addItems(list(self.images.keys()))
                index = self.mainview.combobox.findText(new_name)
                self.mainview.combobox.setCurrentIndex(index)
            fa_edit = qta.icon('fa.edit')
            self.mainview.editButton.setIcon(fa_edit)
        self.mainview.combobox.setEditable(self.is_edit)


    def change_name(self, old_name, new_name):
        if old_name in self.images:
            self.images = OrderedDict([(new_name, v) if k == old_name else (k, v) for k, v in self.images.items()])
            self.metadata = OrderedDict([(new_name, v) if k == old_name else (k, v) for k, v in self.metadata.items()])



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

    def change_image_combobox(self, value):
        current_index = self.mainview.combobox.currentIndex()
        count = self.mainview.combobox.count() - 1
        new_index = max(0, min(current_index + value, count))
        name = self.mainview.combobox.itemText(new_index)
        self.choose_image(name)



    def remove_image(self,  name, manual=False):
        if name in self.metadata:
            del self.metadata[name]
        if name in self.images:
            del self.images[name]
            index = self.mainview.combobox.findText(name)
            self.mainview.combobox.removeItem(index)
            if len(self.images.keys()) > 0:
                if manual:
                    self.choose_image(list(self.images.keys())[index-1])
                else:
                    self.choose_image(list(self.images.keys())[-1])
            else:
                self.choose_image("No image")

    def choose_image(self, name, preview=False, autoLevels=True):
        """
        Choose an image among available image
        The name must be in self.images

        Parameters
        ----------
        name: str
            name of the image, must be in self.images.keys()
        """
        if name == "No image":
            return
        if name not in self.images:
            return
        if not preview:
            self.img_data = self.images[name]
        self.mainview.combobox.setCurrentIndex(self.mainview.combobox.findText(name))
        try:
            xvals = self.images[name].mzs
        except AttributeError:
            xvals = None
        self.mainview.imageview.setImage(self.images[name], xvals=xvals)

    def image_to_visualization(self, img):
        """
        Modifies the image so it can be rendered
        Converts n-D image to 3D

        Parameters
        ----------
        img: np.ndarray
            n-D image loaded by the imageio module
        """
        img2 = np.reshape(img, (img.shape[0], img.shape[1]) + (-1,), order='F')
        return img2


    def on_click_image(self, evt):
        pos = evt
        ive = self.mainview.imageview
        image = ive.imageDisp
        if image is None:
            return
