import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import argparse

def precision(im1, im2):
    tp = np.count_nonzero((im2 + im1) == 2)
    allp = np.count_nonzero(im2 == 1)
    return tp * 1.0 / allp

def recall(im1, im2):
    tp = np.count_nonzero((im2 + im1) == 2)
    allr = np.count_nonzero(im1 == 1)
    return tp * 1.0 / allr

def quality_registration(imRef, imRegistered):
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    imRef_bin = otsu_filter.Execute(imRef)
    imRegistered_bin = otsu_filter.Execute(imRegistered)
    p = precision(imRef_bin, imRegistered_bin)
    r = recall(imRef_bin, imRegistered_bin)
    return p, r


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Fixed image")
parser.add_argument("-r", "--registered", help="Moving image")
parser.add_argument("-b", "--bins", help="number of bins", default=5)

args = parser.parse_args()
fixedname = args.fixed
registeredname = args.registered

args = parser.parse_args()
fixed = sitk.ReadImage(fixedname, sitk.sitkFloat32)
registered = sitk.ReadImage(registeredname, sitk.sitkFloat32)

simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(registered), sitk.sitkUInt8)
p, r = quality_registration(fixed, registered)
