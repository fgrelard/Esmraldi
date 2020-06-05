import esmraldi.imzmlio as io
import matplotlib.pyplot as plt

imzml = io.open_imzml("data/Mouse_Urinary_Bladder_PXD001283/ms_image.imzML")
spectra = io.get_spectra(imzml)

plt.plot(spectra[0, 0], spectra[0, 1])
plt.show()

image = io.get_image(imzml, 559.01, 0.1)
plt.imshow(image)
plt.show()
