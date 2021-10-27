import numpy as np
from PyQt5 import QtCore
from gui.signal import Signal

class WorkerRegistrationSelection(QtCore.QObject):

    signal_end = QtCore.pyqtSignal(np.ndarray, int)

    def __init__(self, fixed, moving, points_fixed, points_moving):
        self.fixed = fixed
        self.moving = moving
        self.points_fixed = points_fixed
        self.points_moving = points_moving

    @QtCore.pyqtSlot()
    def work(self):
        pass

    def abort(self):
        self.is_abort = True


class RegistrationSelectionController:

    def __init__(self, view):
        self.view = view
        self.trigger = Signal()

        self.view.pushButton.clicked.connect(self.start_selection)
        self.view.buttonBox.accepted.connect(self.compute_transformation)
        self.view.buttonBox.rejected.connect(self.cancel_registration)

    def start_selection(self):
        text = self.view.pushButton.text()
        self.parent.mainview.imagehandleview.imageview.setClickable(True)
        self.parent.mainview.imagehandleview2.imageview.setClickable(True)
        if text == "Start":
            self.view.pushButton.setText("Clear")
        else:
            self.view.pushButton.setText("Start")
            self.clear_selection()

    def clear_selection(self):
        self.parent.mainview.imagehandleview.imageview.resetCross()
        self.parent.mainview.imagehandleview2.imageview.resetCross()


    def compute_transformation(self):
        pass

    def cancel_registration(self):
        pass
