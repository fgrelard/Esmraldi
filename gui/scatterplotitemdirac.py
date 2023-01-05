import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import scipy.signal as signal

class ScatterPlotItemDirac(pg.ScatterPlotItem):
    """
    Class allowing to plot diracs (e.g. for spectra)
    """

    def __init__(self, *args, **kwds):

        self.selectedPoints = [[], []]
        self.diracs = [[], []]

        self.x_show, self.y_show = self.diracs

        if "pen" in kwds:
            self.pen = pg.mkPen(kwds["pen"])
        else:
            self.pen = pg.mkPen(QtGui.QColor(220, 220, 220))

        if "selectedPen" in kwds:
            self.selectedPen = pg.mkPen(kwds["selectedPen"])
        else:
            self.selectedPen = pg.mkPen(QtGui.QColor(0, 177, 106))
        pg.ScatterPlotItem.__init__(self, *args, **kwds)
        # self.picture = QtGui.QPicture()
        if self.picture is None:
            self.picture = QtGui.QPicture()
        self.selectedPicture = QtGui.QPicture()
        self.generatePicture()

    def invalidate(self):
        self.update()

    def setDiracs(self, diracs):
        self.diracs = diracs
        self.x_show = diracs[0]
        self.y_show = diracs[1]

        x_minmax = np.amin(self.diracs[0]), np.amax(self.diracs[0])
        y_minmax = np.amin(self.diracs[1]), np.amax(self.diracs[1])
        self.setPoints(x=x_minmax, y=y_minmax, size=0)
        self.generatePicture()

    def generatePicture(self, firstRender=True):
        if self.picture is None:
            return


        n_max = int(1e5)
        if len(self.x_show) > n_max:
            self.x_show = signal.resample(self.x_show, n_max)
            self.y_show = signal.resample(self.y_show, n_max)
        zipped = zip(self.x_show, self.y_show)
        if firstRender:
            self.picture = QtGui.QPicture()
            p = QtGui.QPainter(self.picture)
            p.setPen(self.pen)
            line_list = []
            for x, y in zipped:
                if y > 0:
                    line_list.append(QtCore.QLineF(x, 0, x, y))
            p.drawLines(line_list)
        else:
            self.selectedPicture = QtGui.QPicture()
            p = QtGui.QPainter(self.selectedPicture)
            p.setPen(self.selectedPen)
            x_vals, y_vals = self.selectedPoints
            line_list = []
            for x, y in zip(x_vals, y_vals):
                line_list.append(QtCore.QLineF(x, 0, x, y))
            p.drawLines(line_list)
        p.end()


    def setSelectedPoints(self, selectedPoints):
        self.selectedPoints = selectedPoints
        self.generatePicture(firstRender=False)
        self.getViewBox().update()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
        p.drawPicture(0, 0, self.selectedPicture)
        super().paint(p, *args)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())
