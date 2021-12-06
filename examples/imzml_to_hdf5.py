import matplotlib.pyplot as plt
import sys
import esmraldi.imzmlio as io
import h5py
import datetime
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input imzML")
parser.add_argument("-o", "--output", help="Output HDF5")

args = parser.parse_args()
inputname = args.input
outputname = args.output

sample_name = 'finger print'
sample_source = 'Rappez'
sample_preparation = 'direct on glass'
maldi_matrix = 'DHB'
matrix_application = 'sublimation'
f_in = io.open_imzml(inputname)
print(f_in.metadata.pretty()["instrument_configurations"])
f_out = h5py.File(outputname,'w')
spectral_data = f_out.create_group('spectral_data')
spatial_data = f_out.create_group('spatial_data')
shared_data = f_out.create_group('shared_data')

# parameters
instrument_parameters_1 = shared_data.create_group('instrument_parameters/001')
instrument_parameters_1.attrs['instrument name'] = 'QExactivePlus'
instrument_parameters_1.attrs['analyser type'] = 'Qrbitrap'
instrument_parameters_1.attrs['data conversion'] = 'imzML->hdf5:'+str(datetime.datetime.now())
# m/z axis
    #will centroid data so this doesn't exist
# ROIs
    #todo - determine and propagate all ROIs
roi_1 = shared_data.create_group('regions_of_interest/001')
roi_1.attrs['name'] = 'root region'
roi_1.attrs['parent'] = ''
# Sample
    #todo - not write empty properties
sample_1 = shared_data.create_group('samples/001')
sample_1.attrs['name'] = sample_name
sample_1.attrs['source'] = sample_source
sample_1.attrs['preparation'] = sample_preparation
sample_1.attrs['MALDI matrix'] = maldi_matrix
sample_1.attrs['MALDI matrix application'] = matrix_application

n=0;
for i,coords in enumerate(f_in.coordinates):
    key=str(i)
    ## make new spectrum
    mzs,ints = f_in.getspectrum(i)
    mzs_list,intensity_list = mzs,ints
    # add intensities
    this_spectrum = spectral_data.create_group(key)
    this_intensities = this_spectrum.create_dataset('intensities',data=np.float32(intensity_list),compression="gzip",compression_opts=9)
    # add coordinates
    if len(coords)==2:
        coords = (coords[0],coords[1],1)
    this_coordinates = this_spectrum.create_dataset('coordinates',data=(coords[0],coords[1],coords[2]))
    ## link to shared parameters
    # mzs
    this_mzs = this_spectrum.create_dataset('mzs',data=np.float64(mzs_list),compression="gzip",compression_opts=9)

    ###
    # ROI
    this_spectrum['ROIs/001'] = h5py.SoftLink('/shared_data/regions_of_interest/001')
    # Sample
    this_spectrum['samples/001'] = h5py.SoftLink('/shared_data/samples/001')
    # Instrument config
    this_spectrum['instrument_parameters'] = h5py.SoftLink('/shared_data/instrument_parameters/001')
    n+=1

f_out.close()
