"""
Registration example

Rigid registration
Transform=rigid, centers initialized with moments
Metric=Mattes mutual information
Optimization=Gradient descent
"""
import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import numpy as np
import esmraldi.imzmlio as imzmlio
import os
import esmraldi.registration as reg
import esmraldi.segmentation as segmentation

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.slices, rows, cols, colors = X.shape
        self.ind = 0

        self.im = ax.imshow(self.X[self.ind, :, :, :])
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :, :])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def command_iteration(method) :
    """
    Callback called after each registration iteration

    Parameters
    ----------
    method: sitk.ImageRegistrationMethod
        the registration method

    """
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))



def resample(image, transform):
    """
    Applies a transform to an image


    Parameters
    ----------
    image: sitk.Image
        input image
    transform: sitk.Transform
        a transform

    Returns
    ----------
    sitk.Image
        transformed image

    """
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)




def registration_number_bins(outputname, min_bins, max_bins):
    """
    Registration over a range of bins
    for the sampling strategy

    Parameters
    ----------
    outputname: str
        output image name
    min_bins: int
        minimum number of bins
    max_bins: int
        maximum number of bins

    """
    for i in range(min_bins, max_bins):
        outname = os.path.splitext(outputname)[0] + "_bins" + str(i) +".png"
        try:
            resampler = reg.register(fixed, moving, i)
            out = resampler.Execute(moving)
            out = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
            sitk.WriteImage(out, outname)
        except Exception as e:
            pass

def extract_slice_itk(image, index):
    extract = sitk.ExtractImageFilter()
    size = list(image.GetSize())
    size[2] = 0
    extract.SetSize(size)
    extract.SetIndex([0, 0, index])
    return extract.Execute(image)


def register2D(fixed, moving, numberOfBins, is_best_rotation=False, array_moving=None, flipped=False, sampling_percentage=0.1, learning_rate=1.1, min_step=0.001, relaxation_factor=0.8):
    to_flip = False
    # Flip axis and choose best fit during registration
    if flipped:
        # Construct a 3D image
        # 2 slices = original + flipped
        flipped_moving = sitk.Flip(moving, (True, False))
        moving_and_flipped = np.zeros((2, moving.GetSize()[1], moving.GetSize()[0]), dtype=np.float32)
        moving_and_flipped[0, ...] = sitk.GetArrayFromImage(moving)
        moving_and_flipped[1, ...] = sitk.GetArrayFromImage(flipped_moving)
        array_flipped = moving_and_flipped
        #determines if symmetry of image is a better match
        best_resampler, index = reg.best_fit(fixed, array_flipped, numberOfBins, sampling_percentage, find_best_rotation=is_best_rotation, learning_rate=learning_rate, min_step=min_step, relaxation_factor=relaxation_factor)
        if index == 1:
            to_flip = True
    else:
        #normal registration
        best_resampler = reg.register(fixed, moving, numberOfBins, sampling_percentage, find_best_rotation=is_best_rotation, learning_rate=learning_rate, min_step=min_step, relaxation_factor=relaxation_factor)
    return best_resampler, to_flip


def apply_registration(image, best_resampler, to_flip):
    try:
        if to_flip:
            print("Flipped!")
            image = sitk.Flip(image, (True, False))
            image = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
        out = best_resampler.Execute(image)
    except Exception as e:
        print("Problem with best_resampler")
        print(e)
        return None
    return out

def apply_registration_imzml(image, best_resampler, to_flip):
    size = register.GetSize()
    pixel_type = register.GetPixelID()
    fixed_size = fixed.GetSize()

    outRegister = sitk.Image(fixed_size[0], fixed_size[1], size[2], pixel_type )
    for i in range(size[2]):
        slice = register[:,:,i]
        slice.SetSpacing([1, 1])

        outSlice = best_resampler.Execute(slice)
        outSlice = sitk.JoinSeries(outSlice)
        outRegister = sitk.Paste(outRegister, outSlice, outSlice.GetSize(), destinationIndex=[0,0,i])
    return outRegister

def read_image_to_register(registername, is_imzml):
    mzs = None
    if is_imzml:
        imzml = imzmlio.open_imzml(registername)
        array = imzmlio.to_image_array(imzml).T
        register = sitk.GetImageFromArray(array)
        mz, _ = imzml.getspectrum(0)
    else:
        register = sitk.ReadImage(registername, sitk.sitkFloat32)
    if to_flip:
        dim = register.GetDimension()
        flip = (True, False) + ((False,) if dim > 2 else ())
        register = sitk.Flip(register, flip)
        register = sitk.GetImageFromArray(sitk.GetArrayFromImage(register))

    dim = register.GetDimension()
    register.SetSpacing([1 for i in range(dim)])

    if is_resize:
        new_size = moving.GetSize() + ((register.GetSize()[2],) if dim > 2 else ())
        register = segmentation.resize(register, new_size)

    identity = np.identity(dim).tolist()
    flat_list = [item for sublist in identity for item in sublist]
    direction = tuple(flat_list)
    register.SetDirection(flat_list)

    return register, mz


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Fixed image")
parser.add_argument("-m", "--moving", help="Moving image")
parser.add_argument("-r", "--register", help="Registration image or directory containing images (same number as in moving and fixed)")
parser.add_argument("-p", "--pattern", help="Selects registration images fitting this regexp pattern (default=*) if parameter register is a directory", default="*")
parser.add_argument("-o", "--output", help="Output")
parser.add_argument("-b", "--bins", help="number per bins", default=10)
parser.add_argument("-s", "--symmetry", help="best fit with flipped image", action="store_true", default=False)
parser.add_argument("--resize", help="Resize the moving image to match the fixed image size", action="store_true")
parser.add_argument("--best_rotation", help="Initialize registration by finding the best rotation angle between the two images", action="store_true")
parser.add_argument("--learning_rate", help="Learning rate", default=1.1)
parser.add_argument("--relaxation_factor", help="Relaxation factor", default=0.9)
parser.add_argument("--sampling_percentage", help="Sampling percentage", default=0.1)
parser.add_argument("--min_step", help="Minimum step for gradient descent", default=0.001)
args = parser.parse_args()

fixedname = args.fixed
movingname = args.moving
outputname = args.output
registername = args.register
bins = int(args.bins)
flipped = args.symmetry
is_resize = bool(args.resize)
is_best_rotation = bool(args.best_rotation)

learning_rate = float(args.learning_rate)
relaxation_factor = float(args.relaxation_factor)
sampling_percentage = float(args.sampling_percentage)
min_step = float(args.min_step)

fixed = sitk.ReadImage(fixedname, sitk.sitkFloat32)
moving = sitk.ReadImage(movingname, sitk.sitkFloat32)

#Resizing
dim_moving = moving.GetDimension()
if is_resize:
    fixed_size = fixed.GetSize()
    moving_size = (fixed_size[0], int(moving.GetSize()[1]*fixed_size[0]/moving.GetSize()[0]))
    if dim_moving == 3:
        moving_size += (fixed_size[2], )
    print(moving_size)
    moving = segmentation.resize(moving, moving_size)
    moving = sitk.Cast(moving, sitk.sitkFloat32)


fixed.SetSpacing([1 for i in range(dim_moving)])
moving.SetSpacing([1 for i in range(dim_moving)])

array_moving = sitk.GetArrayFromImage(moving)

width = fixed.GetWidth()
height = fixed.GetHeight()
numberOfBins = int(math.sqrt(height * width / bins))

best_resamplers = []
flips = []
if dim_moving == 2:
    best_resampler, to_flip = register2D(fixed, moving, numberOfBins, is_best_rotation, array_moving, flipped, sampling_percentage, learning_rate, min_step, relaxation_factor)
    out = apply_registration(moving, best_resampler, to_flip)
    best_resamplers.append(best_resampler)
    flips.append(to_flip)


if dim_moving == 3:
    size = fixed.GetSize()
    pixel_type = fixed.GetPixelID()
    out = sitk.Image(size[0], size[1], size[2], pixel_type)
    for i in range(array_moving.shape[0]):
        print("Slice ", i)
        best_resampler, to_flip = register2D(fixed[:,:,i], moving[:,:,i], numberOfBins, is_best_rotation, array_moving, flipped, sampling_percentage, learning_rate, min_step, relaxation_factor)

        out2D = apply_registration(moving[:,:,i], best_resampler, to_flip)
        out2D = sitk.JoinSeries(out2D)
        out = sitk.Paste(out, out2D, out2D.GetSize(), destinationIndex=[0,0,i])
        best_resamplers.append(best_resampler)
        flips.append(to_flip)

if out != None:
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)

    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1//3.+simg2//1.5)

    if dim_moving == 2:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sitk.GetArrayFromImage(moving))
        ax[1].imshow(sitk.GetArrayFromImage(cimg))
    elif dim_moving == 3:
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, sitk.GetArrayFromImage(cimg))
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()
    plt.show()


# print("-------")
# print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
# print(" Iteration: {0}".format(R.GetOptimizerIteration()))
# print(" Metric value: {0}".format(R.GetMetricValue()))

if registername:
    if os.path.isdir(registername):
        list_image_names = []
        for root, dirs, files in os.walk(inputname):
            for f in files:
                if re_pattern.match(f):
                    list_image_names.append(os.path.join(root, f))
    else:
        list_image_names = [registername]

    is_different_resampler = False
    if len(list_image_names) == len(best_resamplers):
        is_different_resampler = True

    for i in range(len(list_image_names)):
        if is_different_resampler:
            best_resampler = best_resamplers[i]
            to_flip = flips[i]
        else:
            best_resampler = best_resamplers[0]
            to_flip = flips[0]

        current_name = list_image_names[i]

        is_imzml = current_name.lower().endswith(".imzml")
        register, mz = read_image_to_register(current_name, is_imzml)

        if is_imzml:
            outRegister = apply_registration_imzml(register, best_resampler, to_flip)
            intensities, coordinates = imzmlio.get_spectra_from_images(sitk.GetArrayFromImage(outRegister).T)
            coordinates = coordinates + ((i,) for i in range(len(coordinates)))
            mzs = [mz] * len(coordinates)
            imzmlio.write_imzml(mzs, intensities, coordinates, outputname)
        else:
            outRegister = apply_registration(register, best_resampler, to_flip)
            outRegister = sitk.Cast(outRegister, sitk.sitkUInt8)
            sitk.WriteImage(outRegister, outputname)
