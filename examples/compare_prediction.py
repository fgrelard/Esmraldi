import SimpleITK as sitk
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
import collections

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def find_parameter_dirs(input_dir, parameters):
    curr_dirs = []
    for d in sorted_alphanumeric(os.listdir(input_dir)):
        if any([d.endswith(p) for p in parameters]):
            curr_dirs.append(input_dir + d)
    return curr_dirs

def find_file_dir(directory, key, search_term):
    visit = None
    for root, dirs, files in os.walk(directory):
        if search_term is not None:
            if search_term in dirs:
                visit = search_term
        if visit is not None and visit not in root:
            continue
        for f in files:
            if f == key + ".png":
                image = sitk.GetArrayFromImage(sitk.ReadImage(root + os.path.sep + f))
                return image
    return None

def find_all_files(directories, key, search_term=None):
    images = {}
    for directory in directories:
        image = find_file_dir(directory, key, search_term)
        title = directory
        title = os.path.basename(title).split("_")[-1]
        if image is not None:
            images[title] = image
    return images


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input directory")
parser.add_argument("-s", "--subtree", help="Subtree folders to analyze")
parser.add_argument("-p", "--parameters", help="Parameters to analyze", nargs="+")
parser.add_argument("-k", "--keys", help="Images to analyze", nargs="+")
parser.add_argument("--search", help="Alternative path containing term")

args = parser.parse_args()
input_dir = args.input
keys = args.keys
parameters = args.parameters
search = args.search

if os.path.isdir(input_dir):
    dirs = find_parameter_dirs(input_dir, parameters)
    for key in keys:
        images = find_all_files(dirs, key, search)
        ordered = collections.OrderedDict(sorted(images.items(), key=lambda k: float(k[0])))
        n = len(images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n*1.0/cols))
        fig, ax = plt.subplots(rows, cols)
        fig.suptitle(key)
        print(n, cols, rows)
        for i, (k, im) in enumerate(ordered.items()):
            if cols != 1:
                if rows == 1:
                    ind = (i,)
                else:
                    ind_x = i//cols
                    ind_y = i%cols
                    ind = (ind_x, ind_y)
                ax_ind = ax[ind]
            else:
                ax_ind = ax
            title = dirs[i]
            title = os.path.basename(title).split("_")[-1]
            ax_ind.imshow(im)
            ax_ind.set_title(k)
            ax_ind.set_axis_off()
        plt.show()
