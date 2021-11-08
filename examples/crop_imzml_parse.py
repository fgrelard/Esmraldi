import esmraldi.imzmlio as io
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyimzml.ImzMLParser as moduleparser
import pyimzml.ImzMLWriter as modulewriter
import warnings
import os
import sys
from shutil import copyfile

class ImzMLParserCrop(moduleparser.ImzMLParser):
    def __init__(self, filename, to_crop=[], parse_lib=None, ibd_file=moduleparser.INFER_IBD_FROM_IMZML, include_spectra_metadata=None):
        self.to_crop = to_crop
        super().__init__(filename, parse_lib, ibd_file, include_spectra_metadata)

    def _ImzMLParser__iter_read_spectrum_meta(self, include_spectra_metadata):
        mz_group = int_group = None
        slist = None
        self.elem_iterator = self.iterparse(self.filename, events=("start", "end"))


        if sys.version_info > (3,):
            _, self.root = next(self.elem_iterator)
        else:
            _, self.root = self.elem_iterator.next()

        is_first_spectrum = True

        for event, elem in self.elem_iterator:
            if elem.tag == self.sl + "spectrumList" and event == "start":
                self._ImzMLParser__process_metadata()
                slist = elem
            elif elem.tag == self.sl + "spectrum" and event == "end":
                to_remove = self._ImzMLParser__process_spectrum(elem, include_spectra_metadata)
                if is_first_spectrum:
                    self._ImzMLParser__read_polarity(elem)
                    is_first_spectrum = False
                if to_remove:
                    slist.remove(elem)
        self._ImzMLParser__fix_offsets()


    def _ImzMLParser__process_spectrum(self, elem, include_spectra_metadata):
        super()._ImzMLParser__process_spectrum(elem, include_spectra_metadata)
        scan_elem_parent = elem.find('%sscanList' % self.sl)
        scan_elem = elem.find('%sscanList/%sscan' % (self.sl, self.sl))
        node_y = scan_elem.find('%s%scvParam[@name="%s"]' % ('', moduleparser.XMLNS_PREFIX, "position y"))
        y = int(node_y.get("value"))
        offset = np.count_nonzero(y > self.to_crop)
        node_y.set("value",  str(y - offset))
        if (y-1) in self.to_crop:
            return True
        return False

    def __exit__(self, exc_t, exc_v, trace):
        import xml
        import xml.etree.ElementTree as ET
        ET.register_namespace("", "http://psi.hupo.org/ms/mzml")
        tree = ET.ElementTree(self.elem_iterator.root)
        tree.write(self.filename, encoding="ISO-8859-1", default_namespace=None, xml_declaration=True)
        super().__exit__(exc_t, exc_v, trace)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input imzML")
parser.add_argument("-o", "--output", help="Output imzML")
parser.add_argument("-l", "--lines", help="Lines to remove", nargs="+", default=[], type=int)

args = parser.parse_args()

input_name = args.input
output_name = args.output
lines = args.lines

lines = np.array(lines)

input_ibd = os.path.splitext(input_name)[0] + ".ibd"
output_ibd = os.path.splitext(output_name)[0] + ".ibd"

copyfile(input_name, output_name)
copyfile(input_ibd, output_ibd)

with  ImzMLParserCrop(output_name, to_crop=lines) as imzml:
    pass
