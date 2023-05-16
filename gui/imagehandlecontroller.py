import os
import numpy as np
import qtawesome as qta
import skimage.color as color
import esmraldi.imzmlio as io
from esmraldi.msimage import MSImage
from esmraldi.sparsematrix import SparseMatrix
from collections import OrderedDict
from PyQt5 import Qt, QtWidgets

class ImageHandleController:
    """
    Class to handle image view as well as its selection
    in the combobox

    Attributes
    ----------
    imageview: ImageViewExtended
        the image view
    is_edit: bool
        whether the image name can be edited
    is_text_editing: bool
        whether the m/z and tolerance can be edited
    current_mz: float
        current m/z value
    tolerance: float
        tolerance in m/z
    img_data: np.ndarray or MSImageBase
        actual image
    current_name: str
        current image name
    images: OrderedDict
        dictionary mapping name to image content
    metadata: OrderedDict
        dictionary mapping name to metadata
    """
    images = OrderedDict()
    metadata = OrderedDict()

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
        self.imageview.ui.comboRoiImage.activated[str].connect(self.choose_roi_image)

        self.is_edit = False
        self.is_text_editing = False
        self.current_mz = 1.0
        self.tolerance = 0.003

        self.img_data = None
        self.current_name = None


    def image_to_view(self, image, filename):
        """
        Converts an image (np.ndarray, MSImageBase) to a view (ImageViewExtended)
        """
        self.img_data = image
        name = os.path.basename(filename)
        name = os.path.splitext(name)[0]
        self.imagehandleview.combobox.setCurrentIndex(self.imagehandleview.combobox.findText(name))
        self.add_image(self.img_data, name)
        self.choose_image(name)
        self.filename = name

    def get_image(self):
        """
        Returns the actual image
        """
        return self.img_data

    def change_mz_value(self, text):
        """
        Change the m/z value to value "text" in the GUI
        """
        self.is_text_editing = True
        number, is_converted = self.locale.toDouble(text)
        if is_converted:
            self.current_mz = number

    def update_mz_value(self):
        """
        Update the m/z value and updates the
        image accordingly
        """
        self.is_text_editing = False
        try:
            ind = (np.abs(self.imageview.tVals - self.current_mz)).argmin()
            self.imageview.setCurrentIndex(ind)
        except Exception as e:
            pass

    def change_tolerance(self, text):
        """
        Change the m/z tolerance to value "text" in the GUI
        """
        self.is_text_editing = True
        number, is_converted = self.locale.toDouble(text)
        if is_converted:
            self.tolerance = number

    def update_tolerance(self):
        """
        Update the m/z tolerance and updates the image
        accordingly
        """
        self.is_text_editing = False
        try:
            self.imageview.imageDisp.tolerance = self.tolerance
            self.imageview.current_image = None
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
        if name not in self.images:
            self.imagehandleview.combobox.addItem(name)
        image_with_metadata = image
        self.images[name] = image_with_metadata
        self.current_name = name
        self.metadata[name] = self.metadata[name] if name in self.metadata else None


    def edit_name(self):
        """
        Edit image name
        """
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
                self.current_name = new_name
            fa_edit = qta.icon('fa.edit', color="#ccc")
            self.imagehandleview.editButton.setIcon(fa_edit)
        self.imagehandleview.combobox.setEditable(self.is_edit)


    def change_name(self, old_name, new_name):
        """
        Edit image name in dictionaries self.images and
        self.metadata
        """
        if old_name in self.images:
            for _ in range(len(self.images)):
                k, v = self.images.popitem(False)
                self.images[new_name if old_name == k else k] = v
            for _ in range(len(self.metadata)):
                k, v = self.metadata.popitem(False)
                self.metadata[new_name if old_name == k else k] = v

    def change_image_combobox(self, value):
        """
        Updates image when changing combobox value
        """
        current_index = self.imagehandleview.combobox.currentIndex()
        count = self.imagehandleview.combobox.count() - 1
        new_index = max(0, min(current_index + value, count))
        self.current_name = self.imagehandleview.combobox.itemText(new_index)
        self.choose_image(self.current_name)

    def remove_image(self, name, manual=False):
        """
        Removing an image
        """
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

    def choose_image(self, name, preview=False, autoLevels=True, display=True):
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

    def choose_roi_image(self, name):
        """
        Choose an image in the ROI panel
        to define a mask
        """
        if name == "No image":
            return
        if name not in self.images:
            return
        img_data = self.images[name]
        current_image = self.imageview.imageDisp
        if self.imageview.hasTimeAxis():
            current_image = current_image[self.imageview.actualIndex]
        if img_data.shape[-1] <= 4:
            img_data = (color.rgb2gray(img_data[..., :3]) * 255).astype(int)
        if img_data.shape == current_image.shape or img_data.shape == current_image.shape[:2]:
            self.imageview.coords_roi = np.argwhere(img_data.T > 0).T
            self.imageview.roiChanged()

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
        """
        Event raised when image is clocked
        """
        pos = evt
        ive = self.imageview
        image = ive.imageDisp
        if image is None:
            return
