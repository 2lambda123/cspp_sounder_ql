#!/usr/bin/env python
# encoding: utf-8
"""
sounder_profile.py

Purpose: Create a profile plot from a range of input data. Supports outputs from
         the following packages...

         * International ATOVS Processing Package (IAPP)
         * Microwave Integrated Retrieval System (MIRS)
         * CSPP Hyperspectral Retrieval (Dual Regression) Package
         * NOAA Unique CrIS/ATMS Product System (NUCAPS)

Preconditions:
    * matplotlib (with basemap)
    * netCDF4 python module
    * h5py python module

Optional:
    *

Minimum commandline:

    python sounder_profile.py INPUTFILE DATATYPE PRODUCT

where...

    INPUTFILES: The fully qualified path to the input files. May be a
    directory or a file glob.

    DATATYPE: One of 'IAPP','MIRS', 'DR' or 'NUCAPS'.

    PRODUCT: One of 'temp','dwpt','wvap','relh',


Created by Geoff Cureton <geoff.cureton@ssec.wisc.edu> on 2014-05-10.
Copyright (c) 2014 University of Wisconsin Regents. All rights reserved.

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

import os, 
import sys
import logging
import traceback
from os import path, uname, environ
import string
import re
import uuid
from shutil import rmtree, copyfile
from glob import glob
from time import time
from datetime import datetime, timedelta

import numpy as np
from numpy import ma
import copy

from scipy import vectorize

import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

# This must come *after* the backend is specified.
import matplotlib.pyplot as ppl

from mpl_toolkits.basemap import Basemap

import scipy.spatial as ss

from netCDF4 import Dataset
from netCDF4 import num2date
import h5py

import skewt as skT
from thermo import mr_to_rh, rh_to_mr, dewpoint_magnus, dewpoint_AB
from thermo import MixR2VaporPress, DewPoint

# every module should have a LOG object
LOG = logging.getLogger(__file__)


def uwyoming():
    sounding = SkewT.Sounding(filename="sounding.txt")

    sounding_keys = ['mixr',
                     'temp',
                     'thtv',
                     'SoundingDate',
                     'hght',
                     'StationNumber',
                     'dwpt',
                     'thte',
                     'sknt',
                     'relh',
                     'drct',
                     'thta',
                     'pres']

    sounding_inputs = {}

    for key in sounding_keys:
        sounding_inputs[key] = sounding[key]

    return sounding_inputs


def get_geo_indices(lat,lon,lat_0=None,lon_0=None):
    '''
    Get the indices in the lat and lon arrays closes to the desired
    latitude and longitude. If no lat and lon are specified, try to select
    the central indices. Uses kd-tree.
    '''

    try:
        geo_shape = lat.shape

        nrows, ncols = geo_shape[0],geo_shape[1]
        LOG.debug("nrows,ncols= ({},{})".format(nrows,ncols))

        if (lat_0==None) and (lon_0==None):
            # Non lat/lon pair given, use the central unmasked values.
            row_idx = int(np.floor(nrows/2.))
            col_idx = int(np.floor(ncols/2.))
            lat_0 = lat[row_idx,col_idx]
            lon_0 = lon[row_idx,col_idx]
            LOG.info("No lat/lon pair given, using ({:4.2f},{:4.2f})".
                    format(lat_0,lon_0))

        lat_mask = lat.mask if ma.is_masked(lat) else np.zeros(lat.shape,dtype='bool')
        lon_mask = lon.mask if ma.is_masked(lon) else np.zeros(lon.shape,dtype='bool')
        geo_mask = lat_mask + lon_mask

        lat_masked = ma.array(lat,mask=geo_mask).ravel()
        lon_masked = ma.array(lon,mask=geo_mask).ravel()
        lat_masked = lat_masked.compressed()
        lon_masked = lon_masked.compressed()

        rows, cols = np.mgrid[0:nrows, 0:ncols]
        cols_masked = ma.array(cols,mask=geo_mask).ravel()
        rows_masked = ma.array(rows,mask=geo_mask).ravel()
        cols_masked = cols_masked.compressed()
        rows_masked = rows_masked.compressed()

        LOG.debug("There are {} masked latitude values".format(np.sum(lat_mask)))
        LOG.debug("There are {} masked longitude values".format(np.sum(lon_mask)))

        LOG.info("Searching for lat,lon coordinate ({:4.2f},{:4.2f})".\
                format(lat_0,lon_0))

        # Construct the kdtree
        points = zip(lon_masked, lat_masked)
        tree = ss.KDTree(points)

        # get index of each point which distance from (lon_0,lat_0) is under 1
        a = tree.query_ball_point([lon_0, lat_0], 1.0)
        if a == []:
            LOG.error("Specified coordinates ({:4.2f},{:4.2f}) not found...".\
                    format(lat_0,lon_0))
            return None,None

        row = [rows_masked[i] for i in a][0]
        col = [cols_masked[i] for i in a][0]

        LOG.debug("Row index is {}".format(row))
        LOG.debug("Col index is {}".format(col))
        
        LOG.debug("Retrieved latitude is {:4.2f}".format(lat[row,col].squeeze()))
        LOG.debug("Retrieved longitude is {:4.2f}".format(lon[row,col].squeeze()))

    except Exception, err :
        LOG.warn("{}".format(err))
        LOG.debug(traceback.format_exc())

    return row,col


def iapp_sounder(iapp_file,lat_0=None,lon_0=None):
    '''
    Pressure_Levels: Pressure for each level in mb (hPa)
    Temperature_Retrieval: Temperature profile in K
    WaterVapor_Retrieval: Water vapor profile (mixing ratio) in g/kg
    Dew_Point_Temp_Retrieval: Dew Point Temperature Profile in K
    '''

    '''
    float Latitude(Along_Track, Across_Track) ;
            Latitude:long_name = "Geodetic Latitude" ;
            Latitude:units = "degrees_north" ;
            Latitude:scale_factor = 1. ;
            Latitude:add_offset = 0. ;
            Latitude:Parameter_Type = "IAPP Output" ;
            Latitude:valid_range = -90.f, 90.f ;
            Latitude:_FillValue = -999.f ;

    float Longitude(Along_Track, Across_Track) ;
            Longitude:long_name = "Geodetic Longitude" ;
            Longitude:units = "degrees_east" ;

    float Pressure_Levels(Pres_Levels) ;
            Pressure_Levels:long_name = "Pressure Levels at which the retrieved 
                                         values are calculated" ;
            Pressure_Levels:units = "hPa" ;

    float Temperature_Retrieval(Along_Track, Across_Track, Pres_Levels) ;
            Temperature_Retrieval:long_name = "Temperature Retrieval for 
                                               the IAPP" ;
            Temperature_Retrieval:units = "degrees Kelvin" ;

    float WaterVapor_Retrieval(Along_Track, Across_Track, Pres_Levels) ;
            WaterVapor_Retrieval:long_name = "Water Vapor Retrieval for 
                                             the IAPP" ;
            WaterVapor_Retrieval:units = "g/kg" ;

    float Dew_Point_Temp_Retrieval(Along_Track, Across_Track, Pres_Levels) ;
            Dew_Point_Temp_Retrieval:long_name = "Dew Point Temperature Retrieval 
                                                  for the IAPP" ;
            Dew_Point_Temp_Retrieval:units = "degrees Kelvin" ;                
    '''

    data_labels = [
                    'pres',
                    'temp',
                    'dwpt',
                    'wvap',
                    'relh',
                    'lat',
                    'lon'
                    ]

    sounding_inputs = {}
    for label in data_labels:
        sounding_inputs[label] = {}

    sounding_inputs['pres']['dset_name'] = 'Pressure_Levels'
    sounding_inputs['temp']['dset_name'] = 'Temperature_Retrieval'
    sounding_inputs['dwpt']['dset_name'] = 'Dew_Point_Temp_Retrieval'
    sounding_inputs['wvap']['dset_name'] = 'WaterVapor_Retrieval'
    sounding_inputs['lat']['dset_name']  = 'Latitude'
    sounding_inputs['lon']['dset_name']  = 'Longitude'
    sounding_inputs['relh'] = None


    # Open the file object
    fileObj = Dataset(iapp_file)

    try:

        sounding_inputs['pres']['node'] = fileObj.variables['Pressure_Levels']
        sounding_inputs['temp']['node'] = fileObj.variables['Temperature_Retrieval']
        sounding_inputs['dwpt']['node'] = fileObj.variables['Dew_Point_Temp_Retrieval']
        sounding_inputs['wvap']['node'] = fileObj.variables['WaterVapor_Retrieval']
        sounding_inputs['lat']['node'] = fileObj.variables['Latitude']
        sounding_inputs['lon']['node'] = fileObj.variables['Longitude']

        lat = sounding_inputs['lat']['node'][:,:]
        lon = sounding_inputs['lon']['node'][:,:]

        for label in data_labels:
            if sounding_inputs[label] != None:
                LOG.info(">>> Processing {} ...".format(sounding_inputs[label]['dset_name']))
                for attr_name in sounding_inputs[label]['node'].ncattrs(): 
                    attr = getattr(sounding_inputs[label]['node'],attr_name)
                    LOG.debug("{} = {}".format(attr_name,attr))
                    sounding_inputs[label][attr_name] = attr

        row,col = get_geo_indices(lat,lon,lat_0=lat_0,lon_0=lon_0)

        if row==None or col==None:
            raise Exception("No suitable lat/lon coordinates found, aborting...")
             
        LOG.debug("Retrieved row,col = {},{}".format(row,col))
        sounding_inputs['lat_0'] = lat[row,col]
        sounding_inputs['lon_0'] = lon[row,col]
        #row,col = 10,10

        sounding_inputs['pres']['data'] = sounding_inputs['pres']['node'][:]
        sounding_inputs['temp']['data'] = sounding_inputs['temp']['node'][row,col,:]
        sounding_inputs['dwpt']['data'] = sounding_inputs['dwpt']['node'][row,col,:]
        sounding_inputs['wvap']['data'] = sounding_inputs['wvap']['node'][row,col,:]

        LOG.info("Closing {}".format(iapp_file))
        fileObj.close()

    except Exception, err :
        LOG.warn("There was a problem, closing {}".format(iapp_file))
        LOG.warn("{}".format(err))
        LOG.debug(traceback.format_exc())
        fileObj.close()
        LOG.info("Exiting...")
        sys.exit(1)

    # Construct the masks of the various datasets, and mask the data accordingly
    for label in ['temp','dwpt','wvap']:
        if sounding_inputs[label] != None:
            LOG.debug(">>> Processing {} ..."
                    .format(sounding_inputs[label]['dset_name']))
            fill_value = sounding_inputs[label]['_FillValue']
            data = ma.masked_equal(sounding_inputs[label]['data'],fill_value)
            LOG.debug("ma.is_masked({}) = {}".format(label,ma.is_masked(data)))

            if ma.is_masked(data):
                if data.mask.shape == ():
                    data_mask = np.ones(data.shape,dtype='bool')
                else:
                    data_mask = data.mask
            else: 
                data_mask = np.zeros(data.shape,dtype='bool')

            LOG.debug("There are {}/{} masked values in {}".\
                    format(np.sum(data_mask),data.size,label))

            sounding_inputs[label]['mask'] = data_mask
            sounding_inputs[label]['data'] = ma.array(data,mask=data_mask)

    # Computing the relative humidity
    sounding_inputs['relh'] = {}
    mr_to_rh_vec = np.vectorize(mr_to_rh)
    rh = mr_to_rh_vec(
            sounding_inputs['wvap']['data'],
            sounding_inputs['pres']['data'],
            sounding_inputs['temp']['data']
            )
    sounding_inputs['relh']['data'] = rh
    sounding_inputs['relh']['units'] = '%'
    sounding_inputs['relh']['long_name'] = 'Relative Humidity'

    return sounding_inputs


def mirs_sounder(mirs_file,lat_0=None,lon_0=None):
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

    float PClw(Scanline, Field_of_view, P_Layer) ;
            PClw:long_name = "Cloud liquid water profile in g/kg" ;
            PClw:scale = 1. ;            
    '''         

    data_labels = [
                    'pres',
                    'temp',
                    'dwpt',
                    'wvap',
                    'relh',
                    'lat',
                    'lon'
                    ]

    sounding_inputs = {}
    for label in data_labels:
        sounding_inputs[label] = {}

    sounding_inputs['pres']['dset_name'] = 'Player'
    sounding_inputs['temp']['dset_name'] = 'PTemp'
    sounding_inputs['wvap']['dset_name'] = 'PVapor'
    sounding_inputs['lat']['dset_name']  = 'Latitude'
    sounding_inputs['lon']['dset_name']  = 'Longitude'
    sounding_inputs['dwpt'] = None
    sounding_inputs['relh'] = None

    # Open the file object
    fileObj = Dataset(mirs_file)

    try:

        sounding_inputs['pres']['node'] = fileObj.variables['Player']
        sounding_inputs['temp']['node'] = fileObj.variables['PTemp']
        sounding_inputs['wvap']['node'] = fileObj.variables['PVapor']
        sounding_inputs['lat']['node'] = fileObj.variables['Latitude']
        sounding_inputs['lon']['node'] = fileObj.variables['Longitude']

        lat = sounding_inputs['lat']['node'][:,:]
        lon = sounding_inputs['lon']['node'][:,:]

        # Fill in some missing attributes
        fill_value = getattr(fileObj,'missing_value')
        sounding_inputs['pres']['units'] = 'hPa'
        sounding_inputs['temp']['units'] = 'K'
        sounding_inputs['wvap']['units'] = 'g/kg'

        for label in data_labels:
            if sounding_inputs[label] != None:
                LOG.info(">>> Processing {} ...".\
                        format(sounding_inputs[label]['dset_name']))
                sounding_inputs[label]['_FillValue'] = fill_value
                for attr_name in sounding_inputs[label]['node'].ncattrs(): 
                    attr = getattr(sounding_inputs[label]['node'],attr_name)
                    LOG.debug("{} = {}".format(attr_name,attr))
                    sounding_inputs[label][attr_name] = attr

        row,col = get_geo_indices(lat,lon,lat_0=lat_0,lon_0=lon_0)
        if row==None or col==None:
            #LOG.error("Specified coordinates not found, aborting...")
            raise Exception("No suitable lat/lon coordinates found, aborting...")
             
        LOG.debug("Retrieved row,col = {},{}".format(row,col))
        sounding_inputs['lat_0'] = lat[row,col]
        sounding_inputs['lon_0'] = lon[row,col]
        #row,col = 10,10

        sounding_inputs['pres']['data'] = sounding_inputs['pres']['node'][:]

        sounding_inputs['temp']['data'] = sounding_inputs['temp']['node'][row,col,:]
        sounding_inputs['wvap']['data'] = sounding_inputs['wvap']['node'][row,col,:]

        LOG.info("Closing {}".format(mirs_file))
        fileObj.close()

    except Exception, err :
        LOG.warn("There was a problem, closing {}".format(mirs_file))
        LOG.warn("{}".format(err))
        LOG.debug(traceback.format_exc())
        fileObj.close()
        LOG.info("Exiting...")
        sys.exit(1)

    # Construct the masks of the various datasets, and mask the data accordingly
    for label in ['temp','wvap']:
        if sounding_inputs[label] != None:
            LOG.debug(">>> Processing {} ..."
                    .format(sounding_inputs[label]['dset_name']))
            data = ma.masked_equal(sounding_inputs[label]['data'],fill_value)
            LOG.debug("ma.is_masked({}) = {}".format(label,ma.is_masked(data)))

            if ma.is_masked(data):
                if data.mask.shape == ():
                    data_mask = np.ones(data.shape,dtype='bool')
                else:
                    data_mask = data.mask
            else: 
                data_mask = np.zeros(data.shape,dtype='bool')

            LOG.debug("There are {}/{} masked values in {}".\
                    format(np.sum(data_mask),data.size,label))

            sounding_inputs[label]['mask'] = data_mask
            sounding_inputs[label]['data'] = ma.array(data,mask=data_mask)

    # Computing the relative humidity
    sounding_inputs['relh'] = {}
    mr_to_rh_vec = np.vectorize(mr_to_rh)
    rh = mr_to_rh_vec(
            sounding_inputs['wvap']['data'],
            sounding_inputs['pres']['data'],
            sounding_inputs['temp']['data']
            )
    sounding_inputs['relh']['data'] = rh
    sounding_inputs['relh']['units'] = '%'
    sounding_inputs['relh']['long_name'] = 'Relative Humidity'

    dewpoint_AB_vec = np.vectorize(dewpoint_AB)
    sounding_inputs['dwpt'] = {}
    dwpt = dewpoint_AB_vec(
            sounding_inputs['temp']['data'],
            sounding_inputs['relh']['data']
            ) + 273.15

    sounding_inputs['dwpt']['data'] = dwpt
    sounding_inputs['dwpt']['units'] = 'K'
    sounding_inputs['dwpt']['long_name'] = 'Dew-point Temperature profile in K'

    return sounding_inputs


def dual_regression_sounder(dr_file,lat_0=None,lon_0=None):
    '''
    Plevs: Pressure for each level in mb (hPa)
    TAir: Retrieved temperature profile at 101 levels (K)
    Dewpnt: Dew point temperature profile at 101 levels (K)
    H2OMMR : Retrieved humidity (water vapor mixing ratio) profile at 101 levels (g/kg)

    /Latitude
    /Longitude
    /Plevs
    /TAir
    /Dewpnt
    /H2OMMR
    '''

    data_labels = [
                    'pres',
                    'temp',
                    'dwpt',
                    'wvap',
                    'relh',
                    'lat',
                    'lon'
                    ]

    sounding_inputs = {}
    for label in data_labels:
        sounding_inputs[label] = {}

    sounding_inputs['pres']['dset_name'] = 'Plevs'
    sounding_inputs['temp']['dset_name'] = 'TAir'
    sounding_inputs['dwpt']['dset_name'] = 'Dewpnt'
    sounding_inputs['relh']['dset_name'] = 'RelHum'
    sounding_inputs['lat']['dset_name']  = 'Latitude'
    sounding_inputs['lon']['dset_name']  = 'Longitude'
    sounding_inputs['wvap'] = None


    # Open the file object
    if h5py.is_hdf5(dr_file):
        fileObj = h5py.File(dr_file,'r')
    else:
        LOG.error("{} does not exist or is not a valid HDF5 file, aborting...".\
                format(dr_file))
        sys.exit(1)

    try:

        sounding_inputs['pres']['node'] = fileObj['Plevs']
        sounding_inputs['temp']['node'] = fileObj['TAir']
        sounding_inputs['dwpt']['node'] = fileObj['Dewpnt']
        sounding_inputs['relh']['node'] = fileObj['RelHum']
        sounding_inputs['lat']['node'] = fileObj['Latitude']
        sounding_inputs['lon']['node'] = fileObj['Longitude']

        lat = sounding_inputs['lat']['node'][:,:]
        lon = sounding_inputs['lon']['node'][:,:]

        #FIXME: All attributes for the DR datasets appear to be strings, 
        #       whether the attribute value is a string or not. So we can't 
        #       determine the attribute type from the file, we just have 
        #       *know* it. Why bother with self-descibing file formats then?
        for label in data_labels:
            if sounding_inputs[label] != None:
                LOG.info(">>> Processing {} ...".\
                        format(sounding_inputs[label]['dset_name']))
                for attr_name in sounding_inputs[label]['node'].attrs.keys(): 
                    attr = sounding_inputs[label]['node'].attrs[attr_name]
                    LOG.debug("{} = {}".format(attr_name,np.squeeze(attr)))
                    sounding_inputs[label][attr_name] = attr

        row,col = get_geo_indices(lat,lon,lat_0=lat_0,lon_0=lon_0)

        if row==None or col==None:
            raise Exception("No suitable lat/lon coordinates found, aborting...")
             
        LOG.debug("Retrieved row,col = {},{}".format(row,col))
        sounding_inputs['lat_0'] = lat[row,col]
        sounding_inputs['lon_0'] = lon[row,col]
        #row,col = 10,10

        sounding_inputs['pres']['data'] = sounding_inputs['pres']['node'][:]
        sounding_inputs['temp']['data'] = \
                sounding_inputs['temp']['node'][:,row,col]
        sounding_inputs['dwpt']['data'] = \
                sounding_inputs['dwpt']['node'][:,row,col]
        sounding_inputs['relh']['data'] = \
                sounding_inputs['relh']['node'][:,row,col]

        LOG.info("Closing {}".format(dr_file))
        fileObj.close()

    except Exception, err :
        LOG.warn("There was a problem, closing {}".format(dr_file))
        LOG.warn("{}".format(err))
        LOG.debug(traceback.format_exc())
        fileObj.close()
        LOG.info("Exiting...")
        sys.exit(1)

    # Construct the masks of the various datasets, and mask the data accordingly
    for label in ['temp','dwpt','relh']:
        if sounding_inputs[label] != None:
            LOG.debug(">>> Processing {} ..."
                    .format(sounding_inputs[label]['dset_name']))
            fill_value = float(sounding_inputs[label]['missing_value'][0])
            data = ma.masked_equal(sounding_inputs[label]['data'],fill_value)
            LOG.debug("ma.is_masked({}) = {}".format(label,ma.is_masked(data)))

            if ma.is_masked(data):
                if data.mask.shape == ():
                    data_mask = np.ones(data.shape,dtype='bool')
                else:
                    data_mask = data.mask
            else: 
                data_mask = np.zeros(data.shape,dtype='bool')

            LOG.debug("There are {}/{} masked values in {}".\
                    format(np.sum(data_mask),data.size,label))

            sounding_inputs[label]['mask'] = data_mask
            sounding_inputs[label]['data'] = ma.array(data,mask=data_mask)


    return sounding_inputs


def nucaps_sounder(nucaps_file,lat_0=None,lon_0=None):
    '''
    Pressure: Pressure for each layer in mb (hPa)
    Temperature: Temperature profile in K
    H2O_MR: Water vapor profile (mixing ratio) in g/g
    '''

    '''
    float Latitude(Number_of_CrIS_FORs) ;
            Latitude:long_name = "Retrieval latitude values for each CrIS FOR" ;
            Latitude:standard_name = "latitude" ;
            Latitude:units = "degrees_north" ;
            Latitude:parameter_type = "NUCAPS data" ;
            Latitude:valid_range = -90.f, 90.f ;
            Latitude:_FillValue = -9999.f ;

    float Longitude(Number_of_CrIS_FORs) ;
            Longitude:long_name = "Retrieval longitude values for each CrIS FOR" ;
            Longitude:standard_name = "longitude" ;
            Longitude:units = "degrees_east" ;
            Longitude:parameter_type = "NUCAPS data" ;
            Longitude:valid_range = -180.f, 180.f ;
            Longitude:_FillValue = -9999.f ;

    float Pressure(Number_of_CrIS_FORs, Number_of_P_Levels) ;
            Pressure:long_name = "Pressure" ;
            Pressure:units = "mb" ;
            Pressure:parameter_type = "NUCAPS data" ;
            Pressure:coordinates = "Time Latitude Longitude" ;
            Pressure:valid_range = 0.f, 2000.f ;
            Pressure:_FillValue = -9999.f ;

    float Temperature(Number_of_CrIS_FORs, Number_of_P_Levels) ;
            Temperature:long_name = "Temperature" ;
            Temperature:standard_name = "air_temperature" ;
            Temperature:units = "Kelvin" ;
            Temperature:parameter_type = "NUCAPS data" ;
            Temperature:coordinates = "Time Latitude Longitude Pressure" ;
            Temperature:valid_range = 0.f, 1000.f ;
            Temperature:_FillValue = -9999.f ;

    float H2O_MR(Number_of_CrIS_FORs, Number_of_P_Levels) ;
            H2O_MR:long_name = "water vapor mixing ratio" ;
            H2O_MR:units = "g/g" ;
            H2O_MR:parameter_type = "NUCAPS data" ;
            H2O_MR:coordinates = "Time Latitude Longitude Effective_Pressure" ;
            H2O_MR:valid_range = 0.f, 1.e+08f ;
            H2O_MR:_FillValue = -9999.f ;
    '''         

    data_labels = [
                    'pres',
                    'temp',
                    'dwpt',
                    'wvap',
                    'relh',
                    'lat',
                    'lon'
                    ]

    sounding_inputs = {}
    for label in data_labels:
        sounding_inputs[label] = {}

    sounding_inputs['pres']['dset_name'] = 'Pressure'
    sounding_inputs['temp']['dset_name'] = 'Temperature'
    sounding_inputs['wvap']['dset_name'] = 'H2O_MR'
    sounding_inputs['lat']['dset_name']  = 'Latitude'
    sounding_inputs['lon']['dset_name']  = 'Longitude'
    sounding_inputs['dwpt'] = None
    sounding_inputs['relh'] = None

    # Open the file object
    fileObj = Dataset(nucaps_file)

    try:

        sounding_inputs['pres']['node'] = fileObj.variables['Pressure']
        sounding_inputs['temp']['node'] = fileObj.variables['Temperature']
        sounding_inputs['wvap']['node'] = fileObj.variables['H2O_MR']
        sounding_inputs['lat']['node'] = fileObj.variables['Latitude']
        sounding_inputs['lon']['node'] = fileObj.variables['Longitude']

        lat = sounding_inputs['lat']['node'][:].reshape(4,30)
        lon = sounding_inputs['lon']['node'][:].reshape(4,30)

        # Fill in some missing attributes
        sounding_inputs['pres']['units'] = 'hPa'
        sounding_inputs['temp']['units'] = 'K'
        sounding_inputs['wvap']['units'] = 'g/kg'

        for label in data_labels:
            if sounding_inputs[label] != None:
                LOG.info(">>> Processing {} ...".\
                        format(sounding_inputs[label]['dset_name']))
                for attr_name in sounding_inputs[label]['node'].ncattrs(): 
                    attr = getattr(sounding_inputs[label]['node'],attr_name)
                    LOG.debug("{} = {}".format(attr_name,attr))
                    sounding_inputs[label][attr_name] = attr

        row,col = get_geo_indices(lat,lon,lat_0=lat_0,lon_0=lon_0)
        if row==None or col==None:
            #LOG.error("Specified coordinates not found, aborting...")
            raise Exception("No suitable lat/lon coordinates found, aborting...")
             
        LOG.debug("Retrieved row,col = {},{}".format(row,col))
        sounding_inputs['lat_0'] = lat[row,col]
        sounding_inputs['lon_0'] = lon[row,col]
        #row,col = 10,10

        sounding_inputs['pres']['data'] = sounding_inputs['pres']['node'][0,:]

        temp = sounding_inputs['temp']['node'][:,:].reshape(4,30,100)
        # Convert water vapor from g/g to g/kg
        wvap = 1000.*sounding_inputs['wvap']['node'][:,:].reshape(4,30,100)

        sounding_inputs['temp']['data'] = temp[row,col,:]
        sounding_inputs['wvap']['data'] = wvap[row,col,:]

        LOG.info("Closing {}".format(nucaps_file))
        fileObj.close()
        #sys.exit(1)

    except Exception, err :
        LOG.warn("There was a problem, closing {}".format(nucaps_file))
        LOG.warn("{}".format(err))
        LOG.debug(traceback.format_exc())
        fileObj.close()
        LOG.info("Exiting...")
        sys.exit(1)

    # Construct the masks of the various datasets, and mask the data accordingly
    for label in ['temp','wvap']:
        if sounding_inputs[label] != None:
            LOG.debug(">>> Processing {} ..."
                    .format(sounding_inputs[label]['dset_name']))
            fill_value = sounding_inputs[label]['_FillValue']
            data = ma.masked_equal(sounding_inputs[label]['data'],fill_value)
            LOG.debug("ma.is_masked({}) = {}".format(label,ma.is_masked(data)))

            if ma.is_masked(data):
                if data.mask.shape == ():
                    data_mask = np.ones(data.shape,dtype='bool')
                else:
                    data_mask = data.mask
            else: 
                data_mask = np.zeros(data.shape,dtype='bool')

            LOG.debug("There are {}/{} masked values in {}".\
                    format(np.sum(data_mask),data.size,label))

            sounding_inputs[label]['mask'] = data_mask
            sounding_inputs[label]['data'] = ma.array(data,mask=data_mask)

    # Computing the relative humidity
    sounding_inputs['relh'] = {}
    mr_to_rh_vec = np.vectorize(mr_to_rh)
    rh = mr_to_rh_vec(
            sounding_inputs['wvap']['data'],
            sounding_inputs['pres']['data'],
            sounding_inputs['temp']['data']
            )
    sounding_inputs['relh']['data'] = rh
    sounding_inputs['relh']['units'] = '%'
    sounding_inputs['relh']['long_name'] = 'Relative Humidity'

    # Computing the water vapor pressure in Pa
    #sounding_inputs['ewat'] = {}
    #MixR2VaporPress_vec = np.vectorize(MixR2VaporPress)
    #vapor_pressure_water = MixR2VaporPress_vec(
            #sounding_inputs['wvap']['data'] * 0.001,
            #sounding_inputs['pres']['data'] * 100.
            #)
    #sounding_inputs['ewat']['data'] = vapor_pressure_water
    #sounding_inputs['ewat']['units'] = 'Pa'
    #sounding_inputs['ewat']['long_name'] = 'Water vapor pressure in Pa'
    
    #DewPoint_vec = np.vectorize(DewPoint)
    #sounding_inputs['dwpt'] = {}
    #dwpt = DewPoint_vec(vapor_pressure_water) + 273.15

    #dewpoint_magnus_vec = np.vectorize(dewpoint_magnus)
    #sounding_inputs['dwpt'] = {}
    #dwpt = dewpoint_magnus_vec(
            #sounding_inputs['temp']['data'],
            #sounding_inputs['relh']['data']
            #)

    dewpoint_AB_vec = np.vectorize(dewpoint_AB)
    sounding_inputs['dwpt'] = {}
    dwpt = dewpoint_AB_vec(
            sounding_inputs['temp']['data'],
            sounding_inputs['relh']['data']
            ) + 273.15

    sounding_inputs['dwpt']['data'] = dwpt
    sounding_inputs['dwpt']['units'] = 'K'
    sounding_inputs['dwpt']['long_name'] = 'Dew-point Temperature profile in K'

    return sounding_inputs



def plot_dict(sounding_inputs,png_name='skewT_plot.png',dpi=200, **plot_options):


    # Determine the row and column from the input lat and lon and the 
    # supplied geolocation


    p = sounding_inputs['pres']['data'][::-1]
    rh = sounding_inputs['relh']['data'][::-1]
    t = sounding_inputs['temp']['data'][::-1] - 273.16
    td = sounding_inputs['dwpt']['data'][::-1] - 273.16

    '''
    p = pressure
    rh = relative humidity
    t = temperature
    td = dewpoint
    fig = optional matplotlib figure
    '''
    fig,ax,canvas = skT.plot_skewt_OO(p, rh, t, td, **plot_options)

    ax.set_xlabel(plot_options['taxis_label'],fontsize=10)
    ax.set_ylabel(plot_options['paxis_label'],fontsize=10)
    ax.set_title(plot_options['title'],fontsize=10)

    # Redraw the figure
    canvas.draw()
    canvas.print_figure(png_name,dpi=dpi)

    return fig,ax


def _argparse():
    '''
    Method to encapsulate the option parsing and various setup tasks.
    '''

    import argparse

    dataChoices=['IAPP','MIRS','DR','NUCAPS']

    defaults = {
                'input_file':None,
                'datatype':None,
                'temp_min':-40,
                'temp_max':40,
                'lat_0':None,
                'lat_0':None,
                'output_file':None,
                'outputFilePrefix' : None,
                'dpi':200
                }

    description = '''Create a Skew-T plot from input sounder data.'''

    usage = "usage: %prog [mandatory args] [options]"
    version = __version__

    parser = argparse.ArgumentParser(
                                     version=version,
                                     description=description
                                     )

    # Mandatory/positional arguments
    
    parser.add_argument(
                      action='store',
                      dest='input_file',
                      type=str,
                      help='''The fully qualified path to a single level-1d input file.'''
                      )

    parser.add_argument(
                      action="store",
                      dest="datatype",
                      default=defaults["datatype"],
                      type=str,
                      choices=dataChoices,
                      help='''The type of the input sounder data file.\n\n
                              Possible values are...
                              {}.
                           '''.format(dataChoices.__str__()[1:-1])
                      )

    # Optional arguments 

    parser.add_argument('--temp_min',
                      action="store",
                      dest="temp_min",
                      type=float,
                      help='''Minimum temperature value to plot.
                      [default: {} deg C]'''.format(defaults["temp_min"])
                      )

    parser.add_argument('--temp_max',
                      action="store",
                      dest="temp_max",
                      type=float,
                      help='''Maximum temperature value to plot.
                      [default: {} deg C]'''.format(defaults["temp_max"])
                      )

    parser.add_argument('-d','--dpi',
                      action="store",
                      dest="dpi",
                      default='200.',
                      type=float,
                      help='''The resolution in dots per inch of the output 
                      png file. 
                      [default: {}]'''.format(defaults["dpi"])
                      )

    parser.add_argument('--lat_0',
                      action="store",
                      dest="lat_0",
                      type=float,
                      help="Plot the sounder footprint closest to lat_0."
                      )

    parser.add_argument('--lon_0',
                      action="store",
                      dest="lon_0",
                      type=float,
                      help="Plot the sounder footprint closest to lon_0."
                      )

    parser.add_argument('-o','--output_file',
                      action="store",
                      dest="output_file",
                      default=defaults["output_file"],
                      type=str,
                      help='''The filename of the output Skew-T png file. 
                      [default: {}]'''.format(defaults["output_file"])
                      )

    parser.add_argument('-O','--output_file_prefix',
                      action="store",
                      dest="outputFilePrefix",
                      default=defaults["outputFilePrefix"],
                      type=str,
                      help="""String to prefix to the automatically generated 
                      png names, which are of the form 
                      <N_Collection_Short_Name>_<N_Granule_ID>.png. 
                      [default: {}]""".format(defaults["outputFilePrefix"])
                      )

    parser.add_argument("-z", "--verbose",
                      dest='verbosity',
                      action="count", 
                      default=0,
                      help='''each occurrence increases verbosity 1 level from 
                      ERROR: -z=WARNING -zz=INFO -zzz=DEBUG'''
                      )


    args = parser.parse_args()

    # Set up the logging
    console_logFormat = '%(asctime)s : (%(levelname)s):%(filename)s:%(funcName)s:%(lineno)d:  %(message)s'
    dateFormat = '%Y-%m-%d %H:%M:%S'
    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level = levels[args.verbosity], 
            format = console_logFormat, 
            datefmt = dateFormat)

    return args


def main():
    '''
    The main method.
    '''

    # Read in the options
    options = _argparse()

    input_file = options.input_file
    datatype = options.datatype
    temp_min = options.temp_min
    temp_max = options.temp_max
    lat_0  = options.lat_0
    lon_0  = options.lon_0
    output_file  = options.output_file
    outputFilePrefix  = options.outputFilePrefix
    dpi = options.dpi

    # Read in the input file, and return a doctionary containing the required
    # data

    dataChoices=['IAPP','MIRS','DR','NUCAPS']

    if datatype == 'IAPP' :
        sounding_inputs = iapp_sounder(input_file,lat_0=lat_0,lon_0=lon_0)
    elif datatype == 'MIRS' :
        sounding_inputs = mirs_sounder(input_file,lat_0=lat_0,lon_0=lon_0)
    elif datatype == 'DR' :
        sounding_inputs = dual_regression_sounder(input_file,lat_0=lat_0,lon_0=lon_0)
    elif datatype == 'NUCAPS' :
        sounding_inputs = nucaps_sounder(input_file,lat_0=lat_0,lon_0=lon_0)
    else:
        pass

    # Determine the filename
    file_suffix = "{}_SkewT".format(datatype)

    if output_file==None and outputFilePrefix==None :
        output_file = "{}.{}.png".format(input_file,file_suffix)
    if output_file!=None and outputFilePrefix==None :
        pass
    if output_file==None and outputFilePrefix!=None :
        output_file = "{}_{}.png".format(outputFilePrefix,file_suffix)
    if output_file!=None and outputFilePrefix!=None :
        output_file = "{}_{}.png".format(outputFilePrefix,file_suffix)


    lat = sounding_inputs['lat_0']
    lon = sounding_inputs['lon_0']
    input_file = path.basename(input_file)
    plot_options = {}

    # Default plot labels
    plot_options['title'] = "{}\n{:4.2f}$^{{\circ}}$N, {:4.2f}$^{{\circ}}$W".\
            format(input_file,lat,lon)
    plot_options['paxis_label'] = 'Pressure (mbar)'
    plot_options['taxis_label'] = 'Temperature ($^{\circ}\mathrm{C}$)'
    plot_options['T_legend'] = 'temperature'
    plot_options['Td_legend'] = 'dew point temperature'
    plot_options['dpi'] = dpi

    # Create the plot
    LOG.info("Creating the skew-T plot {}".format(output_file))
    plot_dict(sounding_inputs,png_name=output_file,**plot_options)

    return 0


if __name__=='__main__':
    sys.exit(main())  
