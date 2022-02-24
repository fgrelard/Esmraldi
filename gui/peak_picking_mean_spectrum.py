# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/ui/peak_picking_mean_spectrum.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_PeakPickingMeanSpectrum(object):
    def setupUi(self, PeakPickingMeanSpectrum):
        PeakPickingMeanSpectrum.setObjectName("PeakPickingMeanSpectrum")
        PeakPickingMeanSpectrum.resize(209, 315)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PeakPickingMeanSpectrum.sizePolicy().hasHeightForWidth())
        PeakPickingMeanSpectrum.setSizePolicy(sizePolicy)
        PeakPickingMeanSpectrum.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.buttonBox = QtWidgets.QDialogButtonBox(PeakPickingMeanSpectrum)
        self.buttonBox.setGeometry(QtCore.QRect(70, 280, 131, 24))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonBox.sizePolicy().hasHeightForWidth())
        self.buttonBox.setSizePolicy(sizePolicy)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayoutWidget = QtWidgets.QWidget(PeakPickingMeanSpectrum)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 40, 191, 141))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setMinimumSize(QtCore.QSize(20, 20))
        self.pushButton.setMaximumSize(QtCore.QSize(20, 20))
        self.pushButton.setMouseTracking(True)
        self.pushButton.setAutoDefault(True)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 6, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 6, 0, 1, 1)
        self.lineEdit_prominence = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_prominence.setObjectName("lineEdit_prominence")
        self.gridLayout.addWidget(self.lineEdit_prominence, 6, 1, 1, 1)
        self.lineEdit_step = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_step.setObjectName("lineEdit_step")
        self.gridLayout.addWidget(self.lineEdit_step, 7, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setMaximumSize(QtCore.QSize(20, 20))
        self.pushButton_2.setMouseTracking(True)
        self.pushButton_2.setAutoDefault(True)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 7, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 7, 0, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(self.gridLayoutWidget)
        self.textEdit.setEnabled(True)
        self.textEdit.setBaseSize(QtCore.QSize(0, 0))
        self.textEdit.setMouseTracking(False)
        self.textEdit.setAcceptDrops(True)
        self.textEdit.setAutoFillBackground(True)
        self.textEdit.setStyleSheet("background: rgba(0,0,0,0)")
        self.textEdit.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textEdit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit.setReadOnly(True)
        self.textEdit.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout.addWidget(self.textEdit, 5, 0, 1, 3)
        self.label_peaks = QtWidgets.QLabel(PeakPickingMeanSpectrum)
        self.label_peaks.setGeometry(QtCore.QRect(80, 240, 131, 21))
        self.label_peaks.setObjectName("label_peaks")

        self.retranslateUi(PeakPickingMeanSpectrum)
        QtCore.QMetaObject.connectSlotsByName(PeakPickingMeanSpectrum)

    def retranslateUi(self, PeakPickingMeanSpectrum):
        _translate = QtCore.QCoreApplication.translate
        PeakPickingMeanSpectrum.setWindowTitle(_translate("PeakPickingMeanSpectrum", "Frame"))
        self.pushButton.setToolTip(_translate("PeakPickingMeanSpectrum", "Prominence threshold for peak selection (height relative to neighboring peaks)"))
        self.pushButton.setText(_translate("PeakPickingMeanSpectrum", "?"))
        self.label.setText(_translate("PeakPickingMeanSpectrum", "Prominence"))
        self.lineEdit_prominence.setText(_translate("PeakPickingMeanSpectrum", "75"))
        self.lineEdit_step.setText(_translate("PeakPickingMeanSpectrum", "14"))
        self.pushButton_2.setToolTip(_translate("PeakPickingMeanSpectrum", "Average noise intensity. If 0 (default), assumed as the standard deviation of the minimum spectrum"))
        self.pushButton_2.setText(_translate("PeakPickingMeanSpectrum", "?"))
        self.label_2.setText(_translate("PeakPickingMeanSpectrum", "Step (ppm)"))
        self.textEdit.setHtml(_translate("PeakPickingMeanSpectrum", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Peak picking - Mean spectrum</span></p></body></html>"))
        self.label_peaks.setText(_translate("PeakPickingMeanSpectrum", "No peaks found yet."))