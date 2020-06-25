======
 Data
======

MSI images must be in the `imzML <https://ms-imaging.org/wp/imzml/>`_ format. Various tools are available online to convert from proprietary format to mzML or imzML (e.g. `imzMLConverter <https://github.com/AlanRace/imzMLConverter>`_).

Complementary images can be in any `ITK format <https://itk.org/Wiki/ITK/File_Formats>`_ (.png, .tif, .hdr...)


Processed vs. continuous spectra
================================

We assume the mass list is the same for each spectrum in the MSI images, i.e. the storage type is *continuous*. However, if the mass list is different, i.e. each spectrum is *processed*, it is possible to convert it to continuous spectra.

For this purpose, use the ``examples/same_mz_axis.py`` script. This script selects peaks across all spectra and aligns peaks that are sufficiently close to a common *m/z* value. This also helps reduce the size of the image.

The script has various parameters:

- factor: the local prominence factor used for peak selection
- level: the noise level estimated across the spectra
- nbpeaks: the number of occurrence for a peak to appear and be considered as a valid reference peak in the alignment procedure
- step: the tolerance value (in *m/z*) to create groups of peaks that can be matched to a common *m/z* value

On another note, the spectra can be merged by preserving the information across all spectra, and simply removing duplicates in the mass list. See :func:`esmraldi.spectraprocessing.same_mz_axis`.
