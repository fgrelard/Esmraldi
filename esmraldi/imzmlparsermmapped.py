import mmap
from pyimzml.ImzMLParser import *

class ImzMLParserMMapped(ImzMLParser):

    def __init__(self,
                 filename,
                 parse_lib=None,
                 ibd_file=INFER_IBD_FROM_IMZML,
                 include_spectra_metadata=None,):
        super().__init__(filename, parse_lib, ibd_file, include_spectra_metadata)
        self.mmap = mmap.mmap(self.m.fileno(), 0, access=mmap.ACCESS_READ)
        self.getspectrum(0)

    def __exit__(self, exc_t, exc_v, trace):
        super().__exit__()
        self.mmap.close()

    def get_spectrum_as_string(self, index):
        offsets = [self.mzOffsets[index], self.intensityOffsets[index]]
        lengths = [self.mzLengths[index], self.intensityLengths[index]]
        lengths[0] *= self.sizeDict[self.mzPrecision]
        lengths[1] *= self.sizeDict[self.intensityPrecision]
        self.mmap.seek(offsets[0])
        mz_string = self.mmap.read(lengths[0])
        self.mmap.seek(offsets[1])
        intensity_string = self.mmap.read(lengths[1])
        return mz_string, intensity_string
