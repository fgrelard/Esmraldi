"""
Converts an imzML file to NIFTI
"""
import argparse
import os
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


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI imzML")
parser.add_argument("-o", "--output", help="Output nii file")
args = parser.parse_args()

inputname = args.input
outname = args.output

root, ext = os.path.splitext(outname)
image = imzmlio.open_imzml(inputname)
img_array = imzmlio.to_image_array(image)
mzs, y = image.getspectrum(0)
imzmlio.to_nifti(img_array, outname)
imzmlio.to_csv(mzs, root+".csv")
#seg.max_variance_sort(image)
