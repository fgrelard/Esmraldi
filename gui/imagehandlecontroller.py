import os
import numpy as np
import qtawesome as qta
from esmraldi.msimage import MSImage, MSImageImplementation
from esmraldi.sparsematrix import SparseMatrix
import esmraldi.imzmlio as io
from collections import OrderedDict
from PyQt5 import Qt, QtWidgets



class ImageHandleController:
    def __init__(self, imagehandleview):
        self.locale = Qt.QLocale(Qt.QLocale.English)
        Qt.QLocale.setDefault(self.locale)

        self.imagehandleview = imagehandleview
        self.imageview = self.imagehandleview.imageview

        self.imagehandleview.lineEdit.textEdited.connect(self.change_mz_value)
        self.imagehandleview.lineEdit.returnPressed.connect(self.update_mz_value)

        self.imagehandleview.lineEditTol.textEdited.connect(self.change_tolerance)
        self.imagehandleview.lineEditTol.returnPressed.connect(self.update_tolerance)


        self.imagehandleview.combobox.activated[str].connect(self.choose_image)
        self.imagehandleview.trashButton.clicked.connect(lambda : self.remove_image(self.current_name, manual=True))

        self.imagehandleview.editButton.clicked.connect(self.edit_name)

        self.imageview.scene.sigMouseClicked.connect(self.on_click_image)
        self.imageview.signal_image_change.connect(self.change_image_combobox)
        self.imageview.signal_mz_change.connect(lambda mz: self.imagehandleview.lineEdit.setText("{:.4f}".format(mz)))


        self.is_edit = False
        self.is_text_editing = False

        self.current_mz = 1.0
        self.tolerance = 0.003

        self.img_data = None

        self.images = OrderedDict()
        self.metadata = OrderedDict()

        nb = 2000
        mzs = (np.arange(nb)+1)

        x = np.random.random((10, 100, nb))
        x[x < 0.9] = 0  # fill most of the array with zeros
        x_r = x.reshape((np.prod(x.shape[:-1]), x.shape[-1]))
        spectra = np.stack((np.tile(mzs, (np.prod(x.shape[:-1]),1)), x_r), axis=1)
        sm = SparseMatrix(x, is_maybe_densify=False)
        mzs = SparseMatrix(spectra, is_maybe_densify=False)
        mss = MSImage(spectra, sm, tolerance=0.0003)
        mss.spectral_axis = 0
        mss = mss.transpose((2,1,0))
        self.img_data = mss
        self.add_image(self.img_data, "test")
        self.choose_image("test")
        self.filename = "test"

    def image_to_view(self, image, filename):
        self.img_data = image
        name = os.path.basename(filename)
        name = os.path.splitext(name)[0]
        self.imagehandleview.combobox.setCurrentIndex(self.imagehandleview.combobox.findText(name))
        self.add_image(self.img_data, name)
        self.choose_image(name)
        self.filename = name

    def get_image(self):
        return self.img_data

    def change_mz_value(self, text):
        self.is_text_editing = True
        number, is_converted = self.locale.toDouble(text)
        if is_converted:
            self.current_mz = number

    def update_mz_value(self):
        self.is_text_editing = False
        try:
            ind = (np.abs(self.imageview.tVals - self.current_mz)).argmin()
            self.imageview.setCurrentIndex(ind)
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
            self.imageview.imageDisp.tolerance = self.tolerance
            self.imageview.updateImage()
        except Exception as e:
            print("error", e)

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
        self.imagehandleview.combobox.addItem(name)
        image_with_metadata = image
        self.images[name] = image_with_metadata
        self.current_name = name
        self.metadata[name] = self.metadata[name] if name in self.metadata else None

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
            self.imagehandleview.editButton.setIcon(fa_check)
        else:
            self.imagehandleview.combobox.update()
            old_name = self.current_name
            new_name = self.imagehandleview.combobox.currentText()
            if old_name != "No image":
                self.change_name(old_name, new_name)
                self.imagehandleview.combobox.clear()
                self.imagehandleview.combobox.addItems(list(self.images.keys()))
                index = self.imagehandleview.combobox.findText(new_name)
                self.imagehandleview.combobox.setCurrentIndex(index)
            fa_edit = qta.icon('fa.edit')
            self.imagehandleview.editButton.setIcon(fa_edit)
        self.imagehandleview.combobox.setEditable(self.is_edit)


    def change_name(self, old_name, new_name):
        if old_name in self.images:
            self.images = OrderedDict([(new_name, v) if k == old_name else (k, v) for k, v in self.images.items()])
            self.metadata = OrderedDict([(new_name, v) if k == old_name else (k, v) for k, v in self.metadata.items()])

    def change_image_combobox(self, value):
        current_index = self.imagehandleview.combobox.currentIndex()
        count = self.imagehandleview.combobox.count() - 1
        new_index = max(0, min(current_index + value, count))
        self.current_name = self.imagehandleview.combobox.itemText(new_index)
        self.choose_image(self.current_name)

    def remove_image(self,  name, manual=False):
        if name in self.metadata:
            del self.metadata[name]
        if name in self.images:
            del self.images[name]
            index = self.imagehandleview.combobox.findText(name)
            self.imagehandleview.combobox.removeItem(index)
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
        self.imagehandleview.combobox.setCurrentIndex(self.imagehandleview.combobox.findText(name))
        try:
            xvals = self.images[name].mzs
        except AttributeError:
            xvals = None
        self.current_name = name
        self.imageview.setImage(self.images[name], xvals=xvals)

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
        ive = self.imageview
        image = ive.imageDisp
        if image is None:
            return
