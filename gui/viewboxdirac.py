import pyqtgraph as pg
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.Point import Point
from gui.scatterplotitemdirac import ScatterPlotItemDirac

class ViewBoxDirac(pg.ViewBox):
    def __init__(self, selectable=True, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)

        self.x_selected = []
        self.y_selected = []

        self.selectable = selectable

        self.setMouseEnabled(True, False)
        self.setMouseMode(self.RectMode)

    def updateRange(self, ev):
        if not hasattr(ev, "isFinish") or ev.isFinish():
            x_range, y_range = self.viewRange()
            x_min, x_max = x_range
            for child in self.allChildren():
                if isinstance(child, ScatterPlotItemDirac):
                    x, y = child.diracs
                    condition = (x > x_min) & (x < x_max)
                    child.x_show = x[condition]
                    child.y_show = y[condition]
                    child.generatePicture()
                    child.update()

            self.setYRange(0, y_range[1], padding=0)

    def mouseDragEvent(self, ev, axis=None):
        self.setLimits(minYRange=0)

        if ev.isFinish() and  ev.button() == QtCore.Qt.MouseButton.LeftButton and self.selectable:
            rect = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(ev.pos()))
            rect = self.childGroup.mapRectFromParent(rect)
            min_x, max_x = rect.left(), rect.right()
            for child in self.allChildren():
                if isinstance(child, ScatterPlotItemDirac):
                    x, y = child.diracs
                    condition = (x > min_x) & (x < max_x)
                    self.x_selected = x[condition]
                    self.y_selected = y[condition]
                    # child.setSelectedPoints(data_selected)
            self.rbScaleBox.hide()
        elif ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.setMouseMode(pg.ViewBox.PanMode)
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)
            self.setMouseMode(pg.ViewBox.RectMode)
            self.updateRange(ev)
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.setMouseEnabled(True, True)
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)
            self.updateRange(ev)
        else:
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)

        self.setMouseEnabled(True, False)
