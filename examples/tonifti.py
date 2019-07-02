import src.segmentation as seg
import src.imzmlio as imzmlio

image = imzmlio.open_imzml("/mnt/d/MALDI/imzML/MSI_20190419_01/00/peaksel.imzML")
print(image.coordinates)
img_array = imzmlio.to_image_array(image)
imzmlio.to_nifti(img_array, "/mnt/d/MALDI/imzML/MSI_20190419_01/00/peaksel.nii")
#seg.max_variance_sort(image)
