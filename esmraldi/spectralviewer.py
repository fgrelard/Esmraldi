"""
Simple 2D viewer for MS images
"""

import numpy as np
import esmraldi.spectraprocessing as sp
import matplotlib.pyplot as plt


class SpectralViewer(object):
    def __init__(self, ax, X, spectra, **kwargs):
        """
        Parameters
        ----------
        self: type
            description
        ax: matplotlib.Axes
            axes where to show the MS image
        X: np.ndarray
            the MS image
        spectra: np.ndarray
            the associated full spectra
        kwargs: dict
            arguments passed onto the plt.imshow function

        """
        self.ax = ax

        self.X = X
        self.spectra = spectra
        self.ind = 0

        self.mzs = spectra[0, 0, ...]
        # self.mean_spectrum = sp.spectra_mean(spectra)
        self.mean_spectrum = self.mzs.copy()

        current_slice = self.X[..., 0]
        self.im = self.ax[0].imshow(current_slice, **kwargs)
        self.plot, = self.ax[1].plot(self.mzs, self.mean_spectrum)

        self.spectrum, = self.ax[2].plot([],[])
        self.ax[2].set_visible(False)

        self.ax[1].set_xlabel("m/z")
        self.ax[1].set_ylabel("I")

        self.ax[2].set_xlabel("m/z")
        self.ax[2].set_ylabel("I")
        self.update()

    def onclick(self, event):
        """
        On click event.

        Either:
         - Show the spectrum associated to the picked position on
           the image
         - Or change m/z image associated to the picked position on
           the mean spectrum

        Parameters
        ----------
        self: type
            description
        event: matplotlib.MouseEvent
            the mouse event
        """
        x, y = event.xdata, event.ydata
        if event.inaxes == self.im.axes:
            x, y = int(x), int(y)
            ind = np.ravel_multi_index((y,x), self.X.shape[:-1])
            self.ax[2].set_visible(True)
            self.spectrum.set_xdata(self.mzs)
            self.spectrum.set_ydata(self.spectra[ind, 1, :])
            self.ax[2].relim()
            self.ax[2].autoscale_view()
            self.spectrum.axes.figure.canvas.draw()

        if event.inaxes == self.plot.axes:
            self.ind = np.argmin(np.abs(self.mzs - x))
        self.update()

    def update(self):
        """
        Update the image after event.

        Parameters
        ----------
        self: type
            description
        """
        self.im.set_data(self.X[..., self.ind])
        self.im.axes.get_xaxis().set_visible(False)
        self.im.axes.get_yaxis().set_visible(False)
        self.ax[0].set_title('m/z %s' % self.mzs[self.ind])
        self.im.axes.figure.canvas.draw()
