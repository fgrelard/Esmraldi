# Esmraldi

Workflow for the fusion of MALDI and MRI images
Four main steps:
(1) Pre-processing of images: MALDI reduction by peak detection, alignment and deisotoping (based on [ms-deisotope](https://pypi.org/project/ms-deisotope/))
(2) Segmentation: definition of spatial coherence measure, region growing
(3) Registration: rigid registration uses SimpleITK, followed by variational registration of ([Modersitzki et al., 2009](#Modersitzki2009)) in C++ ITK (Module [VariationalRegistration](https://itk.org/Doxygen/html/group__VariationalRegistration.html))
(4) Statistical analysis: MALDI dimension reduction, finding correlations with MRI image by ascending Euclidean distances in the reduced space.

## Data
MALDI images must be in the [imzML](https://ms-imaging.org/wp/imzml/) format. Various tools are available online to convert from proprietary format to mzML or imzML (e.g. [imzMLConverter](https://github.com/AlanRace/imzMLConverter))

MRI images can be in any [ITK format](https://itk.org/Wiki/ITK/File_Formats) (.png, .tif, .hdr...)

## Installation
``` bash
cd Esmraldi

#Install dependencies:
cat requirements.txt | xargs -n 1 -L 1 pip install
```

## Usage
Several examples are available for each module in the `examples` directory. 

``` bash
python -m examples.module_name [--options] [--help]
```
The `help` argument describes the arguments.

A typical worfklow consists in:

  1. Spectra processing:
  ``` bash
  python -m examples.spectra_alignment -i maldi_image.imzML -o peak_selected.imzML -p 100 -s 0.5
  ```
  
  2. Conversion to NifTI:
  ``` bash
  python -m examples.to_nifti -i peak_selected.imzML -o maldi_image.nii
  ```
  
  3. Segmentation:
  ``` bash
  python -m examples.segmentation -i maldi_image.nii -o segmented_maldi_image.nii -f 2100 -t 50
  ```
  
  4. Registration:
  ``` bash
  python -m examples.registration -f segmented_mri_image.png -m segmented_maldi_image.png -r maldi_image.imzML -o registered_maldi_image.imzML -b 15 -s
  ```
  
  5. Applying a transform given by the ITK variational module:
  ``` bash
  python -m examples.apply_itk_transform -i registered_maldi_image.imzML -t transform.mha -o registered_maldi_image.imzML
  ```
  
  6. Statistical analysis:
  ``` bash
  python -m examples.statistical_analysis -i registered_maldi_image.imzML -m segmented_mri_image.png -o sorted_ascending_ion_images.tif
  ```
The code for the evaluation of the proposed methods is available in the `evaluation` subdirectory.

The modules in the `src` directory contain the logic of the code. 

## References

<a name="Modersitzki2009"> Modersitzki, J. (2009). *Fair: Flexible Algorithms for Image Registration.*
    Society for Industrial and Applied Mathematics, Philadelphia, PA, USA.</a>
