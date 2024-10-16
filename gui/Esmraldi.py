from gui.maincontroller import MainController
from gui.mainview import Ui_MainView

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout
from PyQt5.QtGui import QIcon
import sys
import configparser
import os
import qdarkstyle

os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

def enrich_qss(qss):
    QSS = """
QRangeSlider {
    qproperty-barColor: #6490c4;
    border:none;
}
QSlider{
    background-color: none;
    border: none;
    border-radius: 0px;
}
QSlider::groove:horizontal {
  border: none;
}
QSlider::sub-page:horizontal {
  border: none;
}
QTextEdit {
    border:none;
}
QDialogButtonBox QPushButton {
  /* Issue #194 #248 - Special case of QPushButton inside dialogs, for better UI */
  min-width: 0px;
}

"""
    return qss + QSS

# For high DPI screens
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

class AppWindow(QMainWindow):
    """
    Main window class
    Sets up the main widget
    """
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainView()
        self.ui.setupUi(self)
        self.gridLayout = QGridLayout(self.ui.centralwidget)
        self.gridLayout.addLayout(self.ui.gridLayout, 0, 0, 1, 1)
        self.setWindowIcon(QIcon('Esmraldi.ico'))
        self.show()


def init_configuration():
    """
    Initialisation of the configuration:
    storage preferences

    Parameters
    ----------
    None.

    Returns
    ----------
    configparser.ConfigParser:
        configuration preferences

    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    if 'default' not in config:
        config['default'] = {}
    if 'imzmldir' not in config['default']:
        config['default']['imzmldir'] = os.getcwd()
    if 'otherformatdir' not in config['default']:
        config['default']['otherformatdir'] = os.getcwd()
    return config

def start():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    qss = enrich_qss(qdarkstyle.load_stylesheet())
    app.setStyleSheet(qss)
    main_window = AppWindow()
    config = init_configuration()
    main_controller = MainController(app, main_window, config)
    main_window.show()
    app.exec_()

if __name__=='__main__':
    start()
