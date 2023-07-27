==================
 Typical workflow
==================

A typical workflow consists in:

1. Spectra processing
2. Conversion to NifTI
3. Segmentation
4. Registration
5. Applying a transform given by the ITK variational module
6. Joint statistical analysis

It can be achieved by executing the following examples sequentially: ::

  python -m examples.spectra_alignment -i maldi_image.imzML -o peak_selected.imzML -p 100 -s 0.5
  python -m examples.to_nifti -i peak_selected.imzML -o maldi_image.nii
  python -m examples.segmentation -i maldi_image.nii -o segmented_maldi_image.nii -f 2100 -t 50
  python -m examples.registration -f segmented_mri_image.png -m segmented_maldi_image.png -r maldi_image.imzML -o registered_maldi_image.imzML -b 15 -s
  python -m examples.apply_itk_transform -i registered_maldi_image.imzML -t transform.mha -o registered_maldi_image.imzML
  python -m examples.statistical_analysis -i registered_maldi_image.imzML -m segmented_mri_image.png -o sorted_ascending_ion_images.tif
The parameters should be adapted depending on the use-case.


Other examples
==============

The code for the evaluation of the proposed methods is available in the ``evaluation`` subdirectory.
