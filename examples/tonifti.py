import src.segmentation as seg
import src.imzmlio as imzmlio
import argparse

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


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI imzML")
parser.add_argument("-o", "--output", help="Output nii file")
args = parser.parse_args()

inputname = args.input
outname = args.output

image = imzmlio.open_imzml(inputname)
img_array = imzmlio.to_image_array(image)
imzmlio.to_nifti(img_array, outname)
#seg.max_variance_sort(image)
