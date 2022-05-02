from setuptools import setup

setup(
   name='esmraldi',
   version='1.1',
   description='Multimodal Mass Spectrometry imaging tools',
   author='Florent Grélard',
   author_email='florent.grelard@gmail.com',
   packages=['esmraldi'],  #same as name
   install_requires=['wheel', 'pyimzML', 'numpy', 'seaborn', 'scipy', 'pandas', 'matplotlib', 'opencv_python', 'XlsxWriter', 'nibabel', 'scikit_image', 'ordered_set', 'Pillow', 'scikit_learn', 'SimpleITK', 'scikit-image', 'dtw-python', 'vedo', 'vtk', 'PyQt5', 'PyQt5-sip', 'tifffile', 'pynput', 'bresenham'], #external packages as dependencies
)
