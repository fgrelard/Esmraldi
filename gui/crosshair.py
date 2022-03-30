from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

class Crosshair(QtGui.QGraphicsItem):
    def __init__(self, text):
        QtGui.QGraphicsItem.__init__(self)
        self.text = text
        self.pen = pg.mkPen('r')
        self.setFlag(self.ItemIgnoresTransformations)

    def paint(self, p, *args):
        p.setPen(self.pen)
        p.drawLine(-10, 0, 10, 0)
        p.drawLine(0, -10, 0, 10)
        p.drawText(3, 15, self.text)

    def setPen(self, pen=pg.mkPen('r')):
        self.pen = pen

    def setPenVisible(self, visible):
        if visible:
            self.setPen()
        else:
            self.setPen(QtGui.QPen(QtGui.QColor(0,0,0,0)))

    def boundingRect(self):
        return QtCore.QRectF(-15, -15, 30, 30)


class CrosshairDrawing:
    def __init__(self):
        self.counter = 0

    def get_drawable_crosshair(self, x, y):
        self.counter += 1
        crosshair = Crosshair(str(self.counter))
        crosshair.setPos(x, y)
        return crosshair
