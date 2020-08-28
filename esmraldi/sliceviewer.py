class SliceViewer(object):
    def __init__(self, ax, X):
        self.ax = ax
        self.ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.slices  = X.shape[0]
        self.ind = 0

        self.im = self.ax.imshow(self.X[self.ind, ...])
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, ...])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
