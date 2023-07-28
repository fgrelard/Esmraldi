import esmraldi.imzmlio as io
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input imzML")
parser.add_argument("-o", "--output", help="Output imzML")
parser.add_argument("-l", "--lines", help="Lines to remove (start1,end1), (start2, end2)", nargs="+", type=int, action="append")

args = parser.parse_args()

input_name = args.input
output_name = args.output
lines = args.lines

lines = np.array(lines)
new_lines=[]
for l in lines:
    new_lines += list(range(*l))

new_lines = np.array(new_lines)
print(new_lines)

imzml = io.open_imzml(input_name)

coordinates = imzml.coordinates

max_x = max(coordinates, key=lambda item:item[0])[0]
max_y = max(coordinates, key=lambda item:item[1])[1]

line_indices = [(i+1, l+1, 1) for l in new_lines for i in range(max_x)]

keep_indices = np.arange(max_x*max_y).reshape((max_y, max_x), order="C")
keep_indices = keep_indices.flatten().tolist()
keep_coordinates = sorted(coordinates, key=lambda x: x[1])

print("coordinates")
for i, l in enumerate(new_lines):
    start = (l-i) * max_x
    end = start + max_x
    print(start,end)
    del keep_indices[start:end]
    del keep_coordinates[start:end]


new_coordinates = []
for c in keep_coordinates:
    y = c[1]
    offset = np.count_nonzero(y > new_lines)
    new_coord = (c[0], c[1] - offset, c[2])
    new_coordinates.append(new_coord)

keep_coordinates = new_coordinates

# keep_indices = []
# keep_coordinates = []
# for i, c in enumerate(coordinates):
#     if c in line_indices:
#         print(c)
#         continue
#     keep_indices.append(i)
#     x = c[1]
#     offset = np.count_nonzero(x > lines)
#     new_coord = (c[0], c[1] - offset, c[2])
#     keep_coordinates.append(new_coord)

# print(keep_coordinates, keep_indices)

print("get spectra")
spectra = io.get_spectra(imzml, keep_indices)
mz, I = spectra[:, 0], spectra[:, 1]

keep_coordinates = sorted(keep_coordinates, key=lambda x: x[0])
print(mz.shape, I.shape)
io.write_imzml(mz, I, keep_coordinates, output_name)
