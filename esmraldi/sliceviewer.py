import numpy as np

class SliceViewer(object):
    def __init__(self, ax, *X):
        self.ax = np.array([ax]).flatten()

        self.ax[0].set_title('use scroll wheel to navigate images')
        print(X)
        self.X = X
        self.ind = 0

        self.slices = X[0].shape[0]

        self.im = []
        for i in range(len(self.ax)):
            self.im.append(self.ax[i].imshow(self.X[i][self.ind, ...]))

        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        for i in range(len(self.ax)):
            self.im[i].set_data(self.X[i][self.ind, ...])
            self.ax[i].set_ylabel('slice %s' % self.ind)
            self.im[i].axes.figure.canvas.draw()
