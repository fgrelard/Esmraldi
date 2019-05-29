import pyimzml
import argparse
import src.spectraprocessing
import pyimzml.ImzMLParser as imzmlparser

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI (.imzML)")
parser.add_argument("-o", "--output", help="Output")
args = parser.parse_args()

inputname = args.input
outname = args.output
p = imzmlparser.ImzMLParser(inputname)
print(p.getspectrum(1))
