import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.Point import Point

class ViewBoxDirac(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.x_selected = []
        self.setMenuEnabled(True)
        self.setMouseEnabled(True, False)
        self.setMouseMode(self.RectMode)

    def mouseDragEvent(self, ev, axis=None):
        if ev.isFinish() and  ev.button() == QtCore.Qt.MouseButton.LeftButton:
            rect = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(ev.pos()))
            rect = self.childGroup.mapRectFromParent(rect)
            min_x, max_x = rect.left(), rect.right()
            for child in self.allChildren():
                if isinstance(child, pg.BarGraphItem):
                    x, y = child.getData()
                    condition = (x > min_x) & (x < max_x)

                    brushes = [QtGui.QColor(250, 216, 89) if condition[i] else pg.getConfigOption("foreground") for i in range(len(condition))]
                    child.setOpts(pens=brushes)
                    self.x_selected = x[condition]
            self.rbScaleBox.hide()
        elif ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.setMouseMode(pg.ViewBox.PanMode)
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)
            self.setMouseMode(pg.ViewBox.RectMode)
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.setMouseEnabled(True, True)
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)
        else:
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)
        self.setMouseEnabled(True, False)
