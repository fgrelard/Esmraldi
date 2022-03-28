from PyQt5.QtCore import QObject, pyqtSignal


class Signal(QObject):
    """
    Class wrapping a PyQt5 signal

    Attributes
    ----------
    signal: pyqtSignal
        the signal
    """
    signal = pyqtSignal()
