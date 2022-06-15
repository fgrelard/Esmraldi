"""
Basic class to visualize 3D images
with numpy and matplotlib
"""

import numpy as np

class SliceViewer(object):
    def __init__(self, ax, *X, labels=None, **kwargs):
        """
        Parameters
        ----------
        self: type
            description
        ax: matplotlib.Axes
            axes where to show the images
        X: np.ndarray
            image, or series of images
        kwargs: dict
            arguments passed onto the plt.imshow function

        """
        self.ax = np.array([ax]).flatten()

        self.ax[0].set_title('use scroll wheel to navigate images')
        self.X = X
        self.ind = 0

        self.slices = X[0].shape[0]
        self.labels = labels

        self.im = []
        for i in range(len(self.ax)):
            self.im.append(self.ax[i].imshow(self.X[i][self.ind, ...], **kwargs))

        self.update()

    def onscroll(self, event):
        """
        Change slice number on mouse scroll event.

        Parameters
        ----------
        self: type
            description
        event: matplotlib.MouseEvent
            the current event

        """
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        """
        Update the image after event.

        Parameters
        ----------
        self: type
            description

        """
        for i in range(len(self.ax)):
            self.im[i].set_data(self.X[i][self.ind, ...])
            if self.labels is None:
                self.ax[i].set_xlabel('slice %s' % self.ind)
            else:
                self.ax[i].set_xlabel('slice %s' % self.labels[self.ind])
            self.im[i].axes.figure.canvas.draw()
