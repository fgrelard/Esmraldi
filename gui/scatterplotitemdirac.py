import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

class ScatterPlotItemDirac(pg.ScatterPlotItem):


    def __init__(self, *args, **kwds):

        self.selectedPoints = [[], []]
        self.diracs = [[], []]

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
        self.setPoints(x=diracs[0], y=diracs[1], size=0)
        self.generatePicture()

    def generatePicture(self, firstRender=True):
        if self.picture is None:
            return

        x_vals, y_vals = self.diracs
        zipped = zip(x_vals, y_vals)
        if firstRender:
            p = QtGui.QPainter(self.picture)
            p.setPen(self.pen)
            for x, y in zipped:
                if y > 0:
                    p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, y))
        else:
            p = QtGui.QPainter(self.selectedPicture)
            p.setPen(self.selectedPen)
            x_vals, y_vals = self.selectedPoints
            for x, y in zip(x_vals, y_vals):
                p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, y))
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
