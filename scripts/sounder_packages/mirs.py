#!/usr/bin/env python
# encoding: utf-8
"""
MIRS.py

Purpose: Provide read methods for various datasets associated with the
Microwave Integrated Retrieval System (MiRS) Package.

Created by Geoff Cureton <geoff.cureton@ssec.wisc.edu> on 2015-04-07.
Copyright (c) 2012-2016 University of Wisconsin Regents. All rights reserved.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
import logging
import traceback
from datetime import datetime

import numpy as np
from numpy import ma

from ql_common import get_pressure_index, get_pressure_level
from ql_common import Datafile_NetCDF
from thermo import dewhum, rh_to_mr

from .sounder_packages import Sounder_Packages

# every module should have a LOG object
LOG = logging.getLogger(__file__)

# Dimension names
dimension_name = {}
dimension_name['nrows'] = ['Scanline']
dimension_name['ncols'] = ['Field_of_view']
dimension_name['nlevels'] = ['P_Layer']

# Files dataset names for each dataset type
dset_name = {}
dset_name['pres'] = 'Player'
dset_name['pres_array'] = None
dset_name['lat'] = 'Latitude'
dset_name['lon'] = 'Longitude'
dset_name['ctp'] = None
dset_name['ctt'] = None
dset_name['temp'] = 'PTemp'
dset_name['2temp'] = None
dset_name['cold_air_aloft'] = None
dset_name['dwpt'] = None
dset_name['relh'] = None
dset_name['wvap'] = 'PVapor'

# Dataset types (surface/top; pressure level...)
dset_type = {}
dset_type['pres'] = 'profile'
dset_type['pres_array'] = 'level'
dset_type['lat'] = 'single'
dset_type['lon'] = 'single'
dset_type['ctp'] = 'level'
dset_type['ctt'] = 'level'
dset_type['temp'] = 'level'
dset_type['2temp'] = 'level'
dset_type['cold_air_aloft'] = 'level'
dset_type['dwpt'] = 'level'
dset_type['relh'] = 'level'
dset_type['wvap'] = 'level'

# Dataset dependencies for each dataset

dset_deps = {}
dset_deps['pres'] = []
dset_deps['pres_array'] = []
dset_deps['lat'] = []
dset_deps['lon'] = []
dset_deps['ctp'] = []
dset_deps['ctt'] = []
dset_deps['temp'] = []
dset_deps['2temp'] = ['temp']
dset_deps['cold_air_aloft'] = ['temp', 'wvap', 'temp_gdas', 'relh_gdas']
dset_deps['dwpt'] = ['pres_array', 'temp', 'wvap']
dset_deps['wvap'] = []
dset_deps['relh'] = ['pres_array', 'temp', 'wvap']

# The class method used to read/calculate each dataset.
dset_method = {}
dset_method['pres'] = []
dset_method['lat'] = []
dset_method['lon'] = []
dset_method['ctp'] = []
dset_method['ctt'] = []
dset_method['temp'] = []
dset_method['2temp'] = []
dset_method['cold_air_aloft'] = []
dset_method['dwpt'] = []
dset_method['relh'] = []
dset_method['wvap'] = []


class Mirs(Sounder_Packages):
    '''
    Player: Pressure for each layer in mb (hPa)
    PTemp: Temperature profile in K
    PVapor: Water vapor profile (mixing ratio) in g/kg
    '''
    '''
    float Latitude(Scanline, Field_of_view) ;
            Latitude:long_name = "Latitude of the view (-90,90)" ;

    float Longitude(Scanline,Field_of_view) ;
            Longitude:long_name = "Longitude of the view (-180,180)" ;

    float Player(P_Layer) ;
            Player:description = "Pressure for each layer in mb" ;

    float PTemp(Scanline, Field_of_view, P_Layer) ;
            PTemp:long_name = "Temperature profile in K" ;
            PTemp:scale = 1. ;

    float PVapor(Scanline, Field_of_view, P_Layer) ;
            PVapor:long_name = "Water vapor profile in g/kg" ;
            PVapor:scale = 1. ;
    '''
    def __init__(self, *args, **kwargs):
        Sounder_Packages.__init__(self, *args, **kwargs)

        file_list, data_name, plot_type = args

        self.pkg_name = "Microwave Integrated Retrieval System (MiRS)"
        pres_0 = kwargs['pres_0']

        self.file_list = file_list
        self.plot_type = plot_type
        self.have_geo = False
        self.dimensions = {}
        self.datasets = {}

        # Construct a list of the required datasets...
        dsets_to_read = [data_name] + self.get_dset_deps(data_name)
        LOG.info("Initial reverse list of datasets to read : {}".format(
            dsets_to_read))
        dsets_to_read.reverse()
        LOG.info("Dupe Datasets to read : {}".format(dsets_to_read))

        self.dsets_to_read = []
        for dset in dsets_to_read:
            if dset not in self.dsets_to_read:
                self.dsets_to_read.append(dset)
            else:
                pass

        self.dsets_to_read = ['lat', 'lon'] + self.dsets_to_read
        LOG.info("Final Datasets to read : {}".format(self.dsets_to_read))

        # Define any special methods for computing the required dataset
        dset_method['relh'] = self.relative_humidity
        dset_method['dwpt'] = self.dewpoint_temperature
        dset_method['pres_array'] = self.pressure_array

        # Read in the pressure dataset
        LOG.info(">>> Reading in the pressure dataset...")
        dfile_obj = Datafile_NetCDF(file_list[0])
        data_obj = dfile_obj.Dataset(dfile_obj, dset_name['pres'])
        self.pressure = data_obj.dset[:]
        LOG.debug(f"self.pressure.shape = {self.pressure.shape}")

        for key in dimension_name:
            found_dim = False
            for key_var in dimension_name[key]:
                try:
                    self.dimensions[key] = dfile_obj.dimensions[key_var]
                    found_dim = True
                    break
                except Exception as err:
                    LOG.warn(
                        f"\tThere was a problem reading dimensions '{key}'")
                    LOG.warn("\t{}".format(err))
                    LOG.debug(traceback.format_exc())
            if not found_dim:
                LOG.error(
                    f"Failed to find dimension match for '{key}' for {os.path.basename(file_list[0])}, is this a valid {type(self).__name__.upper()} file?"
                )
                dfile_obj.close()
                sys.exit(1)
            LOG.debug("dimension {} = {}".format(key, self.dimensions[key]))

        dfile_obj.close()

        if plot_type == 'image':

            LOG.info("Preparing a 'level' plot...")

            if True:
                # Determine the level closest to the required pressure
                self.level, self.pres_0 = get_pressure_level(
                    self.pressure, pres_0)
                # Contruct a pass of the required datasets at the desired pressure.
                self.construct_level_pass(file_list, self.level, None, None,
                                          None)

            LOG.debug("\n\n>>> Intermediate dataset manifest...\n")
            LOG.debug(self.print_dataset_manifest(self))

        # Rationalise the satellite name...
        satellite_names = {
            b'_ma1_': u'Metop-B',
            b'_ma2_': u'Metop-A',
            b'.M1.': u'Metop-B',
            b'.M2.': u'Metop-A',
            b'.NN.': u'NOAA-15',
            b'.NP.': u'NOAA-16',
            b'_n18_': u'NOAA-18',
            b'_n19_': u'NOAA-19',
            b'_n20_': u'NOAA-20',
            b'_npp_': u'Suomi-NPP',
            b'_TRP1_': u'TRP1',
            b'_TRP2_': u'TRP2',
            b'_TRP3_': u'TRP3',
            b'_TRP4_': u'TRP4',
            b'_TRP5_': u'TRP5',
            b'_TRP6_': u'TRP6'
        }
        for filename in self.datasets['file_attrs'].keys():
            for key in satellite_names.keys():
                if key.decode('utf-8') in filename:
                    self.datasets['file_attrs'][filename][
                        'Satellite_Name'] = satellite_names[key]
                    break

        LOG.debug("\n\n>>> Final dataset manifest...\n")
        LOG.debug(self.print_dataset_manifest(self))

    def construct_level_pass(self, file_list, level, elev_idx, row, col):
        '''
        Read in each of the desired datasets. Each of the desired datasets may
        be either a "base" dataset, which is read directly from the file, or a
        "derived" dataset, which is constructed from previously read datasets.

        For each granule, all base and derived datasets should be completed, to
        ensure that any dataset interdependencies to not fail due to differing array
        shapes.
        '''

        LOG.info("Contructing a LEVEL pass...")
        LOG.info("(level,row,col)  = ({}, {}, {})".format(level, row, col))

        self.datasets['file_attrs'] = {}
        for dset in self.dsets_to_read:
            self.datasets[dset] = {}

        this_granule_data = {}
        this_granule_mask = {}

        # Loop through each of the granules...
        for grans in np.arange(len(file_list)):

            file_name = file_list[grans]

            try:

                LOG.info("")
                LOG.info("\tReading in granule {} : {}".format(
                    grans, os.path.basename(file_name)))

                dfile_obj = Datafile_NetCDF(file_name)
                self.datasets['file_attrs'][os.path.basename(file_name)] \
                        = dfile_obj.attrs
                self.datasets['file_attrs'][os.path.basename(file_name)]['dt_obj'] \
                        = self.get_granule_dt(file_name)

                # Loop through each of the desired datasets
                for dset in self.dsets_to_read:
                    LOG.info("")
                    LOG.info("\t\tFor dataset {}".format(dset))

                    # Choose the correct "get" method for the dataset
                    if dset_method[dset] == []:
                        get_data = self.get_data
                    else:
                        get_data = dset_method[dset]

                    data = get_data(dfile_obj, dset, level, row, col,
                                    this_granule_data, this_granule_mask)
                    this_granule_data[dset] = data

                    LOG.info("\t\tthis_granule_data['{}'].shape = {}".format(
                        dset, this_granule_data[dset].shape))

                    # Determine the missing/fill value
                    missing_value = None
                    for fill_val_key in dfile_obj.fill_val_keys:
                        if fill_val_key in self.datasets[dset]['attrs'].keys():
                            missing_value = float(
                                self.datasets[dset]['attrs'][fill_val_key])
                            LOG.debug(
                                "Setting missing_value to {} from {} dataset attributes"
                                .format(missing_value, dset))
                    if missing_value is None:
                        missing_value = dfile_obj.fill_value
                    LOG.info("\t\tMissing value = {}".format(missing_value))

                    data_mask = ma.masked_equal(data, missing_value).mask
                    if data_mask.shape == ():
                        data_mask = np.zeros(data.shape, dtype='bool')
                    this_granule_mask[dset] = data_mask

                    try:
                        self.datasets[dset]['data'] = \
                                np.vstack((self.datasets[dset]['data'], this_granule_data[dset]))
                        self.datasets[dset]['data_mask'] = \
                                np.vstack((self.datasets[dset]['data_mask'], this_granule_mask[dset]))

                        LOG.info("\t\tsubsequent arrays...")

                    except KeyError as err:

                        LOG.info("\t\tFirst arrays...")
                        LOG.info(
                            "\t\tCreating new data array for {}".format(dset))

                        self.datasets[dset]['data'] = this_granule_data[dset]
                        self.datasets[dset]['data_mask'] = this_granule_mask[
                            dset]

                    LOG.info("\t\tIntermediate {} shape = {}".format(
                        dset, self.datasets[dset]['data'].shape))
                    #LOG.info("\tIntermediate {} mask shape = {}".format(
                    #dset,self.datasets[dset]['data_mask'].shape))

                #LOG.info("\tClosing file {}".format(file_name))
                dfile_obj.close()

            except Exception as err:
                LOG.warn("\tThere was a problem, closing {}".format(file_name))
                LOG.warn("\t{}".format(err))
                LOG.debug(traceback.format_exc())
                LOG.info("\tClosing file {}".format(file_name))
                dfile_obj.close()
                LOG.info("\tExiting...")
                sys.exit(1)

    def get_data(self, dfile_obj, data_name, level, row, col,
                 this_granule_data, this_granule_mask):
        '''
        This method reads a single granule of the required dataset, and returns
        a single array (which may be a surface/top, pressure level, or slice dataset.
        '''

        LOG.info("\t\t(row,col,level)  = ({}, {}, {})".format(row, col, level))

        level = slice(level) if level is None else level
        row = slice(row) if row is None else row
        col = slice(col) if col is None else col

        data_obj = dfile_obj.Dataset(dfile_obj, dset_name[data_name])
        LOG.info("\t\tdset.shape(row,col,level) = {}".format(
            data_obj.dset.shape))

        if dset_type[data_name] == 'single':
            LOG.info("\t\tgetting a 'single' dataset...")
            dset = data_obj.dset[row, col].squeeze()
        elif dset_type[data_name] == 'level':
            LOG.info("\t\tgetting a 'level' dataset...")
            dset = data_obj.dset[row, col, level].squeeze()

        LOG.info("\t\tDataset {} has shape {}".format(data_name, dset.shape))

        self.datasets[data_name]['attrs'] = dict(data_obj.attrs)

        return dset

    def pressure_array(self, dfile_obj, data_name, level, row, col,
                       this_granule_data, this_granule_mask):
        '''
        Custom method to return an array of the pressure, whether it be at a single
        pressure level, or a vertical slice.
        '''

        LOG.info("\t\tComputing {}".format(data_name))

        # The dataset dimensions from the file dimension block.
        nrows = self.dimensions['nrows']
        ncols = self.dimensions['ncols']
        nlevels = self.dimensions['nlevels']
        LOG.info("\t\t(nrows,ncols,nlevels)  = ({}, {}, {})".format(
            nrows, ncols, nlevels))

        LOG.info("\t\t(row,col,level)  = ({}, {}, {})".format(row, col, level))

        level = slice(level) if level is None else level
        row = slice(row) if row is None else row
        col = slice(col) if col is None else col

        LOG.info("\t\t(row,col,level) slice objects = ({}, {}, {})".format(
            row, col, level))

        LOG.info("\t\ttemp has shape {}".format(
            this_granule_data['temp'].shape))

        LOG.info("\t\tplot type is {}".format(self.plot_type))

        # Create a pressure cube of dimensions (nrows,ncols,nlevels)
        self.pressure_array = np.broadcast_to(
            np.broadcast_to(self.pressure, (ncols, nlevels)),
            (nrows, ncols, nlevels))
        LOG.info("\t\tpressure_array.shape = {}".format(
            self.pressure_array.shape))

        # Determine whether this is an level or slice plot, and get the correct array slice of the
        # pressure cube...
        if self.plot_type == 'image':
            LOG.info("\t\t> getting a pressure level...")
            dset = self.pressure_array[row, col, level].squeeze()
        else:
            LOG.info("\t\t> getting a pressure slice...")
            dset = self.pressure_array[row, col, level].squeeze().T

        LOG.info("\t\tdset.shape = {}".format(dset.shape))

        dset_mask = np.zeros(dset.shape, dtype='bool')

        dset = ma.array(dset, mask=dset_mask)

        self.datasets[data_name]['attrs'] = self.datasets['lat']['attrs']

        return dset

    def pressure_array_old(self, dfile_obj, data_name, level, row, col,
                           this_granule_data, this_granule_mask):
        '''
        Custom method to return an array of the pressure, whether it be at a single
        pressure level, or a vertical slice.
        '''

        LOG.info("\t\t(level,row,col)  = ({}, {}, {})".format(level, row, col))

        level = slice(level) if level is None else level
        row = slice(row) if row is None else row
        col = slice(col) if col is None else col

        LOG.info("\t\t(level,row,col)  = ({}, {}, {})".format(level, row, col))

        LOG.info("\t\tComputing {}".format(data_name))

        LOG.info("\t\trelh_gdas has shape {}".format(
            this_granule_data['relh_gdas'].shape))

        # Determine whether this is an level or slice, and determine the size
        # of the relh_gdas array...
        nlevels = len(self.pressure)
        if this_granule_data['relh_gdas'].shape[0] == nlevels:
            LOG.info("\t\t> getting a pressure slice...")
            nrows = this_granule_data['relh_gdas'].shape[1]
            ncols = None
            #self.pressure_array = np.broadcast_to(np.broadcast_to(self.pressure,(nrows,nlevels)),(ncols,nrows,nlevels)).T
        else:
            LOG.info("\t\t> getting a pressure level...")
            nrows = this_granule_data['relh_gdas'].shape[0]
            ncols = this_granule_data['relh_gdas'].shape[1]
            #self.pressure_array = np.broadcast_to(np.broadcast_to(self.pressure,(nrows,nlevels)),(ncols,nrows,nlevels)).T

        # Contruct the pressure array.
        nrows = this_granule_data['relh_gdas'].shape[0]
        ncols = this_granule_data['relh_gdas'].shape[1]
        LOG.info("\t\t(nlevels,nrows,ncols)  = ({}, {}, {})".format(
            nlevels, nrows, ncols))
        self.pressure_array = np.broadcast_to(
            np.broadcast_to(self.pressure, (nrows, nlevels)),
            (ncols, nrows, nlevels)).T

        LOG.info("\t\tpressure_array.shape = {}".format(
            self.pressure_array.shape))

        LOG.info("\t\tgetting a 'level' dataset...")
        dset = self.pressure_array[level, row, col].squeeze()
        LOG.info("\t\tdset.shape = {}".format(dset.shape))
        dset_mask = np.zeros(dset.shape, dtype='bool')

        dset = ma.array(dset, mask=dset_mask)

        self.datasets[data_name]['attrs'] = self.datasets['lat']['attrs']

        return dset

    def cold_air_aloft(self, dfile_obj, data_name, level, row, col,
                       this_granule_data, this_granule_mask):
        '''
        Custom method to return the temperature, binned into three categories:

        temp < -65 degC             --> 0
        -65 degC < temp < -60 degC  --> 1
        temp > -60 degC             --> 2

        The resulting product is known as the "cold-air-aloft" and is of interest
        to aviation.
        '''

        LOG.info("\t\t(level,row,col)  = ({}, {}, {})".format(level, row, col))

        level = slice(level) if level is None else level
        row = slice(row) if row is None else row
        col = slice(col) if col is None else col

        LOG.info("\t\tComputing {}".format(data_name))
        dset_mask = this_granule_mask['temp']

        LOG.debug("\t\tTemp has shape {}".format(
            this_granule_data['temp'].shape))

        dset = this_granule_data['temp'][:, :].squeeze() - 273.16

        low_idx = np.where(dset < -65.)
        mid_idx = np.where((dset > -65.) * (dset < -60.))
        hi_idx = np.where(dset > -60.)
        dset[low_idx] = 0.
        dset[mid_idx] = 1.
        dset[hi_idx] = 2.

        dset = ma.array(dset, mask=dset_mask)

        self.datasets[data_name]['attrs'] = self.datasets['temp']['attrs']

        return dset

    def temp_times_two(self, dfile_obj, data_name, level, row, col,
                       this_granule_data, this_granule_mask):
        '''
        Custom method to compute two times the temperature.
        '''
        LOG.info("\t\tComputing {}".format(data_name))
        dset = 2. * self.datasets['temp']['data'] + 0.01 * self.pres_0

        self.datasets[data_name]['attrs'] = self.datasets['temp']['attrs']

        return dset

    def dewpoint_temperature(self, dfile_obj, data_name, level, row, col,
                             this_granule_data, this_granule_mask):
        '''
        Custom method to compute the dewpoint temperature.
        '''
        LOG.info("\t\tComputing {}".format(data_name))

        # The dataset dimensions from the file dimension block.
        nrows = self.dimensions['nrows']
        ncols = self.dimensions['ncols']
        nlevels = self.dimensions['nlevels']
        LOG.info("\t\t(nrows,ncols,nlevels)  = ({}, {}, {})".format(
            nrows, ncols, nlevels))

        LOG.info("\t\t(row,col,level)  = ({}, {}, {})".format(row, col, level))

        level = slice(level) if level is None else level
        row = slice(row) if row is None else row
        col = slice(col) if col is None else col

        LOG.info("\t\t(row,col,level) slice objects = ({}, {}, {})".format(
            row, col, level))

        LOG.info("\t\ttemp has shape {}".format(
            this_granule_data['temp'].shape))

        LOG.info("\t\tplot type is {}".format(self.plot_type))

        wvap = this_granule_data['wvap']
        temp = this_granule_data['temp']
        pres = this_granule_data['pres_array']

        dewhum_vec = np.vectorize(dewhum)
        dwpt, _, _ = dewhum_vec(pres, temp, wvap)

        dset = dwpt[:]
        LOG.info("\t\tdset.shape = {}".format(dset.shape))

        # Apply the temperature mask to the dewpoint temperature...
        mask = this_granule_mask['temp'] + ma.masked_less(dset, 100.).mask
        fill_value = self.datasets['temp']['attrs'][
            '_FillValue'] if '_FillValue' in self.datasets['temp'][
                'attrs'].keys() else -99999
        dset = ma.array(dset,
                        mask=this_granule_mask['temp'],
                        fill_value=fill_value)

        self.datasets[data_name]['attrs'] = self.datasets['temp']['attrs']

        return dset

    def relative_humidity(self, dfile_obj, data_name, level, row, col,
                          this_granule_data, this_granule_mask):
        '''
        Custom method to compute the relative humidity.
        '''
        LOG.info("\t\tComputing {}".format(data_name))

        # The dataset dimensions from the file dimension block.
        nrows = self.dimensions['nrows']
        ncols = self.dimensions['ncols']
        nlevels = self.dimensions['nlevels']
        LOG.info("\t\t(nrows,ncols,nlevels)  = ({}, {}, {})".format(
            nrows, ncols, nlevels))

        LOG.info("\t\t(row,col,level)  = ({}, {}, {})".format(row, col, level))

        level = slice(level) if level is None else level
        row = slice(row) if row is None else row
        col = slice(col) if col is None else col

        LOG.info("\t\t(row,col,level) slice objects = ({}, {}, {})".format(
            row, col, level))

        LOG.info("\t\ttemp has shape {}".format(
            this_granule_data['temp'].shape))

        LOG.info("\t\tplot type is {}".format(self.plot_type))

        wvap = this_granule_data['wvap']
        temp = this_granule_data['temp']
        pres = this_granule_data['pres_array']

        dewhum_vec = np.vectorize(dewhum)
        _, rh, _ = dewhum_vec(pres, temp, wvap)

        dset = rh[:]

        # Apply the temperature mask to the relative humidity...
        mask = this_granule_mask['temp'] + ma.masked_less(dset, 0.).mask
        fill_value = self.datasets['temp']['attrs'][
            '_FillValue'] if '_FillValue' in self.datasets['temp'][
                'attrs'].keys() else -99999
        dset = ma.array(dset,
                        mask=this_granule_mask['temp'],
                        fill_value=fill_value)

        self.datasets[data_name]['attrs'] = self.datasets['temp']['attrs']

        return dset

    def print_dataset_manifest(self, pkg_obj):
        dataset_names = list(pkg_obj.datasets.keys())
        dataset_names.remove('file_attrs')
        LOG.debug("datasets: {}".format(dataset_names))

        for key in dataset_names:
            LOG.debug("For dataset {}:".format(key))
            LOG.debug("\tdatasets['{}']['attrs'] = {}".format(
                key, pkg_obj.datasets[key]['attrs']))
            LOG.debug("\tdatasets['{}']['data'].shape = {}".format(
                key, pkg_obj.datasets[key]['data'].shape))
        print("")

    def get_dset_deps(self, dset):
        '''
        For a particular dataset, returns the prerequisite datasets. Works
        recursively to find multiple levels of dependencies.
        Note: Currently there is nothing in place to detect circular dependencies,
        so be careful of how you construct the dependencies.
        '''
        deps = dset_deps[dset]
        for dset in deps:
            deps = deps + self.get_dset_deps(dset)
        return deps

    def get_granule_dt(self, file_name):
        '''
        Computes a datetime object based on the filename.
        Filenames for this package seem to fall into two types...

        SND_SX.M1.D15091.S1516.E1528.B0000001.WE.LR.ORB.nc
        NPR-MIRS-SND_v11r3_ma1_s201807260320000_e201807260332000_c201808071530560.nc
        '''
        file_name = os.path.basename(file_name)

        if 'SND_SX' in file_name:
            image_date_time = '.'.join(file_name.split(".")[2:4])
            dt_image_date = datetime.strptime(image_date_time, 'D%y%j.S%H%M')
        elif 'NPR-MIRS-SND' in file_name:
            image_date_time = file_name.split("_")[3]
            dt_image_date = datetime.strptime(image_date_time,
                                              's%Y%m%d%H%M%S%f')

        return dt_image_date


def gphite(press, temp, wvap, z_sfc, n_levels, i_dir):
    '''
    version of 18.05.00

    PURPOSE:

     Routine to compute geopotential height given the atmospheric state.
       Includes virtual temperature adjustment.

    CREATED:

     19-Sep-1996 Received from Hal Woolf, recoded by Paul van Delst
     18-May-2000 Logic error related to z_sfc corrected by Hal Woolf

     ARGUMENTS:

        Input
       --------
       press    - REAL*4 pressure array (mb)

       temp     - REAL*4 temperature profile array (K)

       wvap     - REAL*4 water vapour profile array (g/kg)

       z_sfc    - REAL*4 surface height (m).  0.0 if not known.

       n_levels - INT*4 number of elements used in passed arrays

        i_dir   - INT*4 direction of increasing layer number

                    i_dir = +1, Level[0] == p[top]         } satellite/AC
                                Level[n_levels-1] == p[sfc]  }    case

                    i_dir = -1, Level[0] == p[sfc]         } ground-based
                                Level[n_levels-1] == p[top]  }    case

        Output
       --------
          z     - REAL*4 pressure level height array (m)

    COMMENTS:

      Dimension of height array may not not be the same as that of the
        input profile data.

    ======================================================================
    python version  of gphite.f
    ======================================================================
    '''

    # -- Parameters

    rog = 29.2898
    fac = 0.5 * rog

    # Determine the level of the lowest valid temp and wvap values in the
    # profile...

    profile_mask = temp.mask + wvap.mask

    valid_idx = np.where(profile_mask is not True)[0]

    lowest_valid_idx = valid_idx[-1]
    lowest_valid_temp = temp[lowest_valid_idx]
    lowest_valid_wvap = wvap[lowest_valid_idx]
    lowest_valid_pressure = press[lowest_valid_idx]

    # Reset number of levels, and truncate profile arrays to exclude missing
    # data at the bottom of the profile.
    n_levels = lowest_valid_idx + 1
    t = temp[:lowest_valid_idx + 1]
    w = wvap[:lowest_valid_idx + 1]
    p = press[:lowest_valid_idx + 1]

    #-----------------------------------------------------------------------
    #  -- Calculate virtual temperature adjustment and exponential       --
    #  -- pressure height for level above surface.  Also set integration --
    #  -- loop bounds                                                    --
    #-----------------------------------------------------------------------

    z = np.ones(n_levels) * (-999.)

    if (i_dir > 0):

        # Data stored top down

        v_lower = t[-1] * (1.0 + (0.00061 * w[-1]))

        algp_lower = np.log(p[-1])

        i_start = n_levels - 1
        i_end = 0

    else:

        # Data stored bottom up

        v_lower = t[0] * (1.0 + (0.00061 * w[0]))

        algp_lower = np.log(p[0])

        i_start = 1
        i_end = n_levels - 1

    #-----------------------------------------------------------------------
    #                     -- Assign surface height --
    #-----------------------------------------------------------------------

    hgt = z_sfc

    # .. Following added 18 May 2000 ... previously, z(n_levels) for downward
    #       (usual) case was not defined!

    if (i_dir > 0):
        z[-1] = z_sfc
    else:
        z[0] = z_sfc

    # .. End of addition

    #-----------------------------------------------------------------------
    #             -- Loop over layers always from sf% -> top --
    #-----------------------------------------------------------------------

    level_idx = np.arange(n_levels)

    # Looping from ground level to TOA.

    for l in level_idx[::-1 * i_dir]:

        # Apply virtual temperature adjustment for upper level

        v_upper = t[l]

        if (p[l] >= 300.0):
            v_upper = v_upper * (1.0 + (0.00061 * w[l]))

        # Calculate exponential pressure height for upper layer

        algp_upper = np.log(p[l])

        # Calculate height

        hgt = hgt + (fac * (v_upper + v_lower) * (algp_lower - algp_upper))

        # Overwrite values for next layer

        v_lower = v_upper
        algp_lower = algp_upper

        # Store heights in same direction as other data

        z[l] = hgt  # geopotential height

    return z


def pressure_to_height(p, t, w, z_sfc=0.):

    gc = 0.98

    z1 = gphite(p, t, w, z_sfc, 101, 1) * gc
    z = z1 * 3.28  # meters to feet

    return z


def get_elevation(pressure, temp, relh):
    '''
    Given cubes of the pressure (mb), temperature (K) and relative humidity (%),
    computes the elevation for each element of the cube.
    '''

    np.set_printoptions(precision=3)
    cube_mask = np.zeros(pressure.shape, dtype='bool')

    pressure = ma.array(pressure, mask=cube_mask)
    temp = ma.array(temp, mask=cube_mask)
    relh = ma.array(relh, mask=cube_mask)

    rh_to_mr_vec = np.vectorize(rh_to_mr)

    wvap = rh_to_mr_vec(relh, pressure, temp)

    level = -4
    LOG.debug("\npressure[{}] = \n{}\n".format(level, pressure[level]))
    LOG.debug("\ntemp[{}] = \n{}\n".format(level, temp[level]))
    LOG.debug("\nrelh[{}] = \n{}\n".format(level, relh[level]))
    LOG.debug("\nwvap[{}] = \n{}\n".format(level, wvap[level]))

    elevation = np.zeros(temp.shape, dtype='float')

    (nlevels, nrows, ncols) = elevation.shape

    for row in np.arange(nrows):
        for col in np.arange(ncols):
            try:
                elevation[:, row, col] = pressure_to_height(pressure[:, row,
                                                                     col],
                                                            temp[:, row, col],
                                                            wvap[:, row, col],
                                                            z_sfc=0.)
            except Exception as err:
                elevation[:, row, col] = -9999.

    return elevation


def get_level_indices(data, match_val):
    '''
    Given cubes of the pressure (mb), temperature (K) and relative humidity (%),
    computes the elevation for each element of the cube.
    '''

    (nlevels, nrows, ncols) = data.shape

    # Determine the level which corresponds with the required elevation...

    data_level_idx = -9999 * np.ones(data[0].shape, dtype='int')

    # Calculate the profile index corresponding to the smallest residual beteween
    # match_val and the profile values (for each row and col)
    for row in np.arange(nrows):
        for col in np.arange(ncols):
            try:
                # Get the data profile and indices for this row/col, masked with
                # the fill values.
                data_profile = ma.masked_equal(data[:, row, col], -9999.)
                data_profile_idx = ma.array(np.arange(len(data_profile)),
                                            mask=data_profile.mask)

                # Compress the profile and mask to remove fill values.
                data_profile = data_profile.compressed()
                data_profile_idx = data_profile_idx.compressed()

                if len(data_profile) == 0:
                    raise ValueError('Entire profile is masked.')

                # Compute the absolute residual of each element of this profile
                # from the match_val.
                data_profile = np.abs(data_profile - match_val)

                # Compute the minimum value of the residuals, and determine its
                # index in the original profile.
                data_profile_min = data_profile.min()
                data_profile_min_idx = np.where(
                    data_profile == data_profile_min)[0][0]

                # Assign the index of the minimum residual for this row/col
                data_level_idx[row, col] = data_profile_min_idx

            except Exception as err:
                data_level_idx[row, col] = -9999

    data_level_idx = ma.masked_equal(data_level_idx, -9999)
    LOG.debug("\ndata_level_idx = \n{}".format(data_level_idx))

    # Create a template index list for a single pressure level
    cube_idx = list(np.where(np.ones(data[0].shape, dtype=int) == 1))
    cube_idx = [
        np.zeros(cube_idx[0].shape, dtype=int), cube_idx[0], cube_idx[1]
    ]
    LOG.debug("\ncube_idx = {}".format(cube_idx))
    LOG.debug("cube_idx[0] = {} has shape {}".format(cube_idx[0],
                                                     cube_idx[0].shape))
    LOG.debug("cube_idx[1] = {} has shape {}".format(cube_idx[1],
                                                     cube_idx[1].shape))
    LOG.debug("cube_idx[2] = {} has shape {}".format(cube_idx[2],
                                                     cube_idx[2].shape))

    # Assign the "level" index list to the
    cube_mask = ma.masked_equal(np.ravel(data_level_idx), -9999).mask
    cube_idx[0] = ma.array(data_level_idx, mask=cube_mask).compressed()
    cube_idx[1] = ma.array(cube_idx[1], mask=cube_mask).compressed()
    cube_idx[2] = ma.array(cube_idx[2], mask=cube_mask).compressed()
    cube_idx = tuple(cube_idx)

    LOG.debug("\ncompressed cube_idx = {}".format(cube_idx))
    LOG.debug("compressed cube_idx[0] = {} has shape {}".format(
        cube_idx[0], cube_idx[0].shape))
    LOG.debug("compressed cube_idx[1] = {} has shape {}".format(
        cube_idx[1], cube_idx[1].shape))
    LOG.debug("compressed cube_idx[2] = {} has shape {}".format(
        cube_idx[2], cube_idx[2].shape))

    return cube_idx
