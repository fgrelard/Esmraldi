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
    def __init__(self, filename, to_crop_x=[], to_crop_y=[], parse_lib=None, ibd_file=moduleparser.INFER_IBD_FROM_IMZML):
        self.to_crop_x = to_crop_x
        self.to_crop_y = to_crop_y
        super().__init__(filename, parse_lib, ibd_file)

    def _ImzMLParser__iter_read_spectrum_meta(self):
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
                slist = elem
            elif elem.tag == self.sl + "spectrum" and event == "end":
                to_remove = self._ImzMLParser__process_spectrum(elem)
                if is_first_spectrum:
                    is_first_spectrum = False
                if to_remove:
                    slist.remove(elem)
            elif elem.tag == self.sl + "referenceableParamGroup" and event == "end":
                for param in elem:
                    if param.attrib["name"] == "m/z array":
                        self.mzGroupId = elem.attrib['id']
                        mz_group = elem
                    elif param.attrib["name"] == "intensity array":
                        self.intGroupId = elem.attrib['id']
                        int_group = elem
        self._ImzMLParser__assign_precision(int_group, mz_group)
        self._ImzMLParser__fix_offsets()


    def _ImzMLParser__process_spectrum(self, elem):
        super()._ImzMLParser__process_spectrum(elem)
        scan_elem_parent = elem.find('%sscanList' % self.sl)
        scan_elem = elem.find('%sscanList/%sscan' % (self.sl, self.sl))
        node_x = scan_elem.find('%s%scvParam[@name="%s"]' % ('', self.sl, "position x"))
        node_y = scan_elem.find('%s%scvParam[@name="%s"]' % ('', self.sl, "position y"))
        x = int(node_x.get("value"))
        y = int(node_y.get("value"))
        offset_x = np.count_nonzero(x > self.to_crop_x)
        offset_y = np.count_nonzero(y > self.to_crop_y)
        node_x.set("value",  str(x - offset_x))
        node_y.set("value",  str(y - offset_y))
        if (x-1) in self.to_crop_x:
            return True
        if (y-1) in self.to_crop_y:
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
parser.add_argument("-l", "--lines", help="Lines to remove (start1,end1), (start2, end2)", nargs="+", type=int, action="append")
parser.add_argument("-c", "--columns", help="Colulns to remove (start1,end1), (start2, end2)", nargs="+", type=int, action="append")

args = parser.parse_args()

input_name = args.input
output_name = args.output
lines = args.lines
cols = args.columns

new_lines = []
for l in lines:
    new_lines += list(range(*l))

new_lines = np.array(new_lines)


new_cols = []
for l in cols:
    new_cols += list(range(*l))

new_cols = np.array(new_cols)
print(new_lines)
print(new_cols)


input_ibd = os.path.splitext(input_name)[0] + ".ibd"
output_ibd = os.path.splitext(output_name)[0] + ".ibd"

copyfile(input_name, output_name)
copyfile(input_ibd, output_ibd)

with  ImzMLParserCrop(output_name, to_crop_x=new_cols, to_crop_y=new_lines) as imzml:
    pass
