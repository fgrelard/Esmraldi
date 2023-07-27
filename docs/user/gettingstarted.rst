==================
 Getting started
==================


Welcome to Esmraldi!
====================

Esmraldi is an open source Python 3 library used to analyze Mass Spectrometry Imaging (**MSI**) jointly with Magnetic Resonance Imaging (**MRI**) images. It focuses on precise and efficient methods for the **fusion** of these images. The code in this library is particularly aimed at :

1. Efficient spectra processing of IMS images
2. Precise registration between several modalities
3. Easily interpretable statistical analysis

This library is designed to be as generic as possible and can be, in principle, used for the fusion of IMS images with any other imaging modality.

Setting up Esmraldi (cross-platform)
====================================

Esmraldi requires Python 3.8. 

Easiest and cross-platform solution is to set up Esmraldi using Anaconda. On Windows, follow `this tutorial <https://www.thewindowsclub.com/how-to-enable-or-disable-win32-long-paths-in-windows-11-10>`_ to enable long paths in the registry.
Then, you can use Anaconda's GUI to import a new environment after downloading Esmraldi. Alternatively, you can paste the following commands in the Anaconda Powershell prompt::

   git clone https://github.com/fgrelard/Esmraldi.git
   cd Esmraldi
   conda env create --file environment.yml


Then, after the installation (might take a while), activate the environment. This process must be done every time you restart the Anaconda command prompt::

   conda activate Esmraldi

The line in the terminal should be prefixed by "(Esmraldi)".

Setting up Esmraldi (Linux)
============================

Alternatively, on Linux, you can set up Esmraldi with::

   git clone https://github.com/fgrelard/Esmraldi.git
   cd Esmraldi
   cat requirements.txt | xargs -n 1 -L 1 pip install

Note that the dependencies listed in ``requirements.txt`` are associated to minimum required versions, tested on Ubuntu 18.04 LTS and 20.04LTS. We do not guarantee that older versions will function properly.

Checking the installation 
=================================

To check whether everything has been installed properly, run::

  python -m examples.check_install

The following output should appear after a few seconds, possibly along with some unimportant missing ``ms_peak_picker`` and ``ms_deisotope`` module warnings::

  Esmraldi set-up successful!
  All required dependencies installed

If it does, congratulations! You can now use Esmraldi.

