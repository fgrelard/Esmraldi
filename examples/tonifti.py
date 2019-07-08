import src.segmentation as seg
import src.imzmlio as imzmlio

def reduce_image(image):
    mzs = []
    intensities = []
    for i, (x, y, z) in enumerate(image.coordinates):
        mz, ints = image.getspectrum(i)
        mz = mz[:1]
        ints = ints[:1]
        mzs.append(mz)
        intensities.append(ints)
    imzmlio.write_imzml(mzs, intensities, image.coordinates, "/mnt/d/MALDI/imzML/MSI_20190419_01/00/peaksel_small.imzML")

image = imzmlio.open_imzml("/mnt/d/MALDI/imzML/MSI_20190419_01/00/peaksel.imzML")
mz, ints = image.getspectrum(0)
imzmlio.to_nifti(img_array, "/mnt/d/MALDI/imzML/MSI_20190419_01/00/peaksel.nii")
#seg.max_variance_sort(image)
