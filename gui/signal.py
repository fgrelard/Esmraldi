from PyQt5.QtCore import QObject, pyqtSignal


class Signal(QObject):
    """
    Class wrapping a PyQt5 signal

    Attributes
    ----------
    signal: pyqtSignal
        the signal
    """
    def __new__(cls, signal_type=None):
        print("hello", signal_type)
        if signal_type is None:
            cls.signal = pyqtSignal()
        else:
            cls.signal = pyqtSignal(signal_type)
        return cls
