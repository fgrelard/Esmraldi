import pyimzml.ImzMLWriter as imzmlwriter
import pyimzml.ImzMLParser as imzmlparser
import numpy as np

def open_imzml(filename):
    return imzmlparser.ImzMLParser(filename)

def write_imzml(mzs, intensities, coordinates, filename):
    with imzmlwriter.ImzMLWriter(filename) as writer:
        for i in range(len(mzs)):
            writer.addSpectrum(mzs[i], intensities[i], coordinates[i])
