import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from pyqtgraph.Point import Point

class ViewBoxDirac(pg.ViewBox):
    def __init__(self, *args, **kwds):
        kwds['enableMenu'] = False
        pg.ViewBox.__init__(self, *args, **kwds)
        self.x_selected = []
        self.setMouseMode(self.RectMode)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.autoRange()

    def mouseDragEvent(self, ev, axis=None):
        if ev.isFinish():
            rect = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(ev.pos()))
            rect = self.childGroup.mapRectFromParent(rect)
            min_x, max_x = rect.left(), rect.right()
            for child in self.allChildren():
                if isinstance(child, pg.BarGraphItem):
                    x, y = child.getData()
                    self.x_selected = x[(x > min_x) & (x < max_x)]
            self.rbScaleBox.hide()
        else:
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)
