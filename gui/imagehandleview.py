from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFrame
import gui.ImageViewExtended as ive
import qtawesome as qta

class Ui_ImageHandleView(object):
    def setupUi(self, ImageHandleView):
        ImageHandleView.setObjectName("ImageHandleView")
        ImageHandleView.setEnabled(True)

        self.centralwidget = QtWidgets.QWidget(ImageHandleView)
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)

        self.imageview = ive.ImageViewExtended(parent=self.centralwidget)

        self.labelmz =  QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelmz.sizePolicy().hasHeightForWidth())
        self.labelmz.setSizePolicy(sizePolicy)
        self.labelmz.setObjectName("labelmz")
        self.lineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)

        self.labeltol =  QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labeltol.sizePolicy().hasHeightForWidth())
        self.labeltol.setSizePolicy(sizePolicy)
        self.labeltol.setObjectName("labeltol")
        self.lineEditTol = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.labelCombo = QtWidgets.QLabel(self.gridLayoutWidget)
        self.horizontalSpace = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        self.combobox = QtWidgets.QComboBox(self.gridLayoutWidget)
        fa_trash = qta.icon('fa.trash')
        fa_edit = qta.icon('fa.edit')
        self.trashButton = QtWidgets.QPushButton(fa_trash, "")
        self.editButton = QtWidgets.QPushButton(fa_edit, "")

        self.configure(ImageHandleView)

        self.retranslateUi(ImageHandleView)
        QtCore.QMetaObject.connectSlotsByName(ImageHandleView)

    def configure(self, ImageHandleView):
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        # self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 750, 500))

        self.gridLayoutWidget.setObjectName("gridLayoutWidget")

        self.gridLayout = QtWidgets.QGridLayout()

        self.gridLayout.setContentsMargins(20, 10, 0, 0)
        self.gridLayout.setObjectName("gridLayout")


        self.hLayoutmz = QtWidgets.QHBoxLayout()
        self.lineEdit.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.hLayoutmz.addStretch()
        self.hLayoutmz.addWidget(self.labelmz, 0, QtCore.Qt.AlignRight)
        self.hLayoutmz.addWidget(self.lineEdit)
        self.gridLayout.addLayout(self.hLayoutmz, 2, 0, 1, 1)

        self.hLayouttol = QtWidgets.QHBoxLayout()
        self.lineEditTol.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.hLayouttol.addStretch()
        self.hLayouttol.addWidget(self.labeltol, 0, QtCore.Qt.AlignRight)
        self.hLayouttol.addWidget(self.lineEditTol)
        self.gridLayout.addLayout(self.hLayouttol, 3, 0, 1, 1)



        self.labelCombo.setText("Current image: ")
        self.combobox.setFixedWidth(200)
        self.combobox.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        self.hLayout = QtWidgets.QHBoxLayout()
        self.hLayout.addWidget(self.labelCombo)
        self.hLayout.addWidget(self.combobox)
        self.hLayout.addWidget(self.editButton)
        self.hLayout.addWidget(self.trashButton)
        self.hLayout.addStretch()
        self.hLayout.setAlignment(QtCore.Qt.AlignLeft)

        self.gridLayout.addLayout(self.hLayout, 0, 0, 1, 1)

        self.imageview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addWidget(self.imageview, 1, 0, 1, 1)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)


    def retranslateUi(self, ImageHandleView):
        _translate = QtCore.QCoreApplication.translate
        ImageHandleView.setWindowTitle(_translate("ImageHandleView", "Esmraldi"))


        self.labelmz.setText(_translate("ImageHandleView", "m/z"))
        self.lineEdit.setText(_translate("ImageHandleView", "1.0"))

        self.labeltol.setText(_translate("ImageHandleView", "Tolerance"))
        self.lineEditTol.setText(_translate("ImageHandleView", "0.003"))
