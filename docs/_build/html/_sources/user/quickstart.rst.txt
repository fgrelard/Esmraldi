==================
 Quickstart tutorial
==================

Welcome to the quickstart tutorial to Esmraldi! If you have comments or suggestions, please don't hesitate to reach out.

Welcome to Esmraldi!
====================

Esmraldi is an open source Python library used to analyze Imaging Mass Spectrometry images (**IMS**) jointly with Magnetic Resonance Imaging (**MRI**) images. It focuses on precise and efficient methods for the **fusion** of these images. The code in this library is particularly aimed at :

1. Efficient spectra processing of IMS images
2. Precise registration between several modalities
3. Interpretable statistical analysis

This library is designed to be as generic as possible and can be, in principle, used for the fusion of IMS images with any other imaging modality.

Setting up Esmraldi
===================

To install Esmraldi, we strongly recommend using a `scientific Python distribution <https://www.scipy.org/install.html>`_. Esmraldi relies on packages such as NumPy, Matplotlib and SciPy.

All packages dependencies are listed in ``requirements.txt``.

You can set-up Esmraldi with::

   git clone https://github.com/fgrelard/Esmraldi.git
   cd Esmraldi
   cat requirements.txt | xargs -n 1 -L 1 pip install


