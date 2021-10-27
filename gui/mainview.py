# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/mainview.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFrame, QStackedWidget
from gui.imagehandleview import Ui_ImageHandleView
from gui.registration_selection import Ui_RegistrationSelection
import qtawesome as qta

class Ui_MainView(object):
    def setupUi(self, MainView):
        MainView.setObjectName("MainView")
        MainView.setEnabled(True)
        MainView.resize(810, 593)

        self.centralwidget = QtWidgets.QWidget(MainView)
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)

        self.qframe = QFrame(self.gridLayoutWidget)
        self.qframe.setLayout(QtWidgets.QVBoxLayout())

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.qframe.setSizePolicy(sizePolicy)

        self.registrationselectionview = self.initialize_frame(Ui_RegistrationSelection)

        self.imagehandleview = Ui_ImageHandleView()
        self.imagehandleview.setupUi(self.gridLayoutWidget)

        self.imagehandleview2 = Ui_ImageHandleView()
        self.imagehandleview2.setupUi(self.gridLayoutWidget)

        fa_oneview = qta.icon('fa5s.dice-one')
        fa_twoviews = qta.icon('fa5s.th-large')
        self.oneViewButton = QtWidgets.QPushButton(fa_oneview, "")
        self.twoViewButton = QtWidgets.QPushButton(fa_twoviews, "")
        self.labelView = QtWidgets.QLabel(self.gridLayoutWidget)
        self.labelView.setText("View:")

        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setText("Running...")
        self.progressBar = QtWidgets.QProgressBar(self.gridLayoutWidget)
        self.stopButton = QtWidgets.QPushButton(self.gridLayoutWidget)

        self.textEdit = QtWidgets.QTextEdit(self.gridLayoutWidget)

        self.menubar = QtWidgets.QMenuBar(MainView)
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.actionExit = QtWidgets.QAction(MainView)

        self.actionSave = QtWidgets.QAction(MainView)

        self.statusbar = QtWidgets.QStatusBar(MainView)

        self.menuHelp = QtWidgets.QMenu(self.menubar)

        self.menuSegmentation = QtWidgets.QMenu(self.menubar)

        self.menuRegistration = QtWidgets.QMenu(self.menubar)

        self.menuAnalyze = QtWidgets.QMenu(self.menubar)

        self.actionOpen = QtWidgets.QAction(MainView)

        self.actionDenoising_TPC = QtWidgets.QAction(MainView)

        self.actionRegistrationSelection = QtWidgets.QAction(MainView)

        self.actionRegistrationCoordinates = QtWidgets.QAction(MainView)

        self.menuFile = QtWidgets.QMenu(self.menubar)

        self.configure(MainView)

        self.retranslateUi(MainView)
        QtCore.QMetaObject.connectSlotsByName(MainView)

    def configure(self, MainView):
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 750, 500))

        self.gridLayoutWidget.setObjectName("gridLayoutWidget")

        self.gridLayout = QtWidgets.QGridLayout()

        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")


        self.textEdit.setAcceptDrops(False)
        self.textEdit.setAutoFillBackground(True)
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")

        self.gridLayout.addWidget(self.qframe, 0, 0, 1, 1)
        self.progressBar.setEnabled(True)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        self.progressBar.setVisible(False)
        self.progressBar.setTextVisible(True)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.label, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.progressBar, 5, 0, 1, 1)

        self.stopButton.setText("Stop")
        self.stopButton.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout.addWidget(self.stopButton, 6, 0, 1, 1)


        hLayout = QtWidgets.QHBoxLayout()
        hLayout.addWidget(self.labelView)
        hLayout.addWidget(self.oneViewButton)
        hLayout.addWidget(self.twoViewButton)
        hLayout.addStretch()
        hLayout.setAlignment(QtCore.Qt.AlignLeft)
        self.gridLayout.addLayout(hLayout, 3, 0, 1, 1)
        # self.gridLayout.setColumnStretch(0, 2)
        self.gridLayout.setColumnStretch(1, 1)

        widgetImage = QtWidgets.QWidget()
        widgetImage2 = QtWidgets.QWidget()
        widgetImage.setLayout(self.imagehandleview.gridLayout)
        widgetImage2.setLayout(self.imagehandleview2.gridLayout)
        self.gridLayout.addWidget(widgetImage, 0, 1, 2, 1)
        self.gridLayout.addWidget(widgetImage2, 0, 2, 2, 1)

        MainView.setCentralWidget(self.centralwidget)
        widgetImage2.hide()

        self.menubar.setGeometry(QtCore.QRect(0, 0, 810, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile.setObjectName("menuFile")
        self.actionOpen.setObjectName("actionOpen")
        self.actionRegistrationSelection.setObjectName("actionRegistrationSelection")
        self.actionRegistrationCoordinates.setObjectName("actionRegistrationCoordinates")
        self.menuRegistration.setObjectName("menuRegistration")
        self.menuSegmentation.setObjectName("menuSegmentation")
        self.menuAnalyze.setObjectName("menuAnalyze")
        self.menuHelp.setObjectName("menuHelp")
        MainView.setMenuBar(self.menubar)
        self.statusbar.setObjectName("statusbar")
        MainView.setStatusBar(self.statusbar)
        self.actionSave.setObjectName("actionSave")
        self.actionExit.setObjectName("actionExit")

        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)

        self.menuRegistration.addAction(self.actionRegistrationSelection)
        self.menuRegistration.addAction(self.actionRegistrationCoordinates)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSegmentation.menuAction())
        self.menubar.addAction(self.menuRegistration.menuAction())
        self.menubar.addAction(self.menuAnalyze.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())


    def initialize_frame(self, FrameName):
        qframe = QFrame()
        content_frame = FrameName()
        content_frame.setupUi(qframe)
        content_frame.retranslateUi(qframe)
        content_frame.parent = qframe
        return content_frame

    def set_frame(self, frame):
        self.qframe.layout().addWidget(frame.parent)
        for i in range(self.qframe.layout().count()):
            item = self.qframe.layout().itemAt(i)
            widget = item.widget()
            self.qframe.layout().removeWidget(widget)
        self.qframe.setMinimumWidth(frame.parent.width()+1)
        print(frame.parent.width())

    def show_run(self):
        self.label.show()
        self.progressBar.show()
        self.stopButton.show()

    def hide_run(self):
        self.label.hide()
        self.progressBar.hide()
        self.stopButton.hide()

    def show_second_view(self):
        self.gridLayout.setColumnStretch(2, 1)
        widget = self.gridLayout.itemAtPosition(0, 2).widget()
        widget.show()
        self.imagehandleview2.imageview.winPlot.autoRange()

    def hide_second_view(self):
        self.gridLayout.setColumnStretch(2, 0)
        widget = self.gridLayout.itemAtPosition(0, 2).widget()
        widget.hide()

    def retranslateUi(self, MainView):
        _translate = QtCore.QCoreApplication.translate
        MainView.setWindowTitle(_translate("MainView", "Esmraldi"))

        self.textEdit.setHtml(_translate("MainView", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Esmraldi</span></p></body></html>"))
        self.textEdit.setStyleSheet("background: rgba(0,0,0,0%)")
        self.textEdit.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.menuFile.setTitle(_translate("MainView", "File"))

        self.menuRegistration.setTitle(_translate("MainView", "Registration"))
        self.menuSegmentation.setTitle(_translate("MainView", "Segmentation"))
        self.menuAnalyze.setTitle(_translate("MainView", "Analyze"))
        self.menuHelp.setTitle(_translate("MainView", "Help"))
        self.actionOpen.setText(_translate("MainView", "Open"))
        self.actionSave.setText(_translate("MainView", "Save Nifti"))
        self.actionExit.setText(_translate("MainView", "Exit"))

        self.actionRegistrationSelection.setText(_translate("MainView", "Fiducials - Selection"))
        self.actionRegistrationCoordinates.setText(_translate("MainView", "Fiducials - Coordinates"))
