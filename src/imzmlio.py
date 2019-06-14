import pyimzml.ImzMLWriter as imzmlwriter
import pyimzml.ImzMLParser as imzmlparser
import numpy as np
import nibabel as nib
import os

def open_imzml(filename):
    return imzmlparser.ImzMLParser(filename)

def write_imzml(mzs, intensities, coordinates, filename):
    with imzmlwriter.ImzMLWriter(filename) as writer:
        for i in range(len(mzs)):
            writer.addSpectrum(mzs[i], intensities[i], coordinates[i])

def to_image_array(image):
    x, y = image.getspectrum(0)
    image_list = []
    for mz in x:
        im = imzmlparser.getionimage(image, mz, tol=0.1)
        image_list.append(im)
    img_array = np.transpose(np.asarray(image_list))
    return img_array

def to_nifti(image, filename):
    nibimg = nib.Nifti1Image(image, np.eye(4))
    nibimg.to_filename(filename)
