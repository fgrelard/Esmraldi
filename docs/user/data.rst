======
 Data
======

MALDI images must be in the `imzML <https://ms-imaging.org/wp/imzml/>`_ format. Various tools are available online to convert from proprietary format to mzML or imzML (e.g. `imzMLConverter <https://github.com/AlanRace/imzMLConverter>`_).

Complementary images can be in any `ITK format <https://itk.org/Wiki/ITK/File_Formats>`_ (.png, .tif, .hdr...)


Different m/z axis
==================

The dataset can be downloaded by running: ::

  cd data/Mouse_Urinary_Bladder_PXD001283/
  python download.py

After completion (might take a while), the three following downloaded files should be located in the ``data/Mouse_Urinary_Bladder_PXD001283``: ms_image.imzML, ms_image.imzML, and optical_image.tiff.
