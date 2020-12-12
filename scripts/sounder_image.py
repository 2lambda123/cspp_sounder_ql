#!/usr/bin/env python
# encoding: utf-8
"""
sounder_image.py

Purpose: Create a plot of temperature, dewpoint or relative humidity,
         at a particular pressure level. Supports outputs from the following
         packages...

         * International ATOVS Processing Package (IAPP)
         * Microwave Integrated Retrieval System (MiRS)
         * CSPP Hyperspectral Retrieval (HSRTV) Package
         * NOAA Unique CrIS/ATMS Product System (NUCAPS)
         * Hyper-Spectral Enterprise Algorithm Package (HEAP NUCAPS)


Preconditions:
    * matplotlib (with basemap)
    * netCDF4 python module
    * h5py python module

Optional:
    *

Minimum commandline:

    python sounder_image.py  INPUTFILE DATATYPE

where...

    INPUTFILES: The fully qualified path to the input files. May be a
    file glob.

    DATATYPE: One of 'iapp','mirs', 'hsrtv', 'nucaps' OR 'heap'.


Created by Geoff Cureton <geoff.cureton@ssec.wisc.edu> on 2014-05-10.
Copyright (c) 2018 University of Wisconsin Regents.
Licensed under GNU GPLv3.
"""

import os
import sys
import logging
import traceback
from cffi import FFI

from args import argument_parser_image

import numpy as np
from numpy import ma
import copy

from scipy import vectorize

import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from netCDF4 import Dataset
from netCDF4 import num2date
import h5py

import sounder_image_data
from thermo import dewhum
from ql_common import granuleFiles
from ql_common import (
    plotMapDataContinuous,
    plotMapDataContinuous_cartopy,
    plotMapDataDiscrete,
)
from ql_common import set_plot_styles

import sounder_packages

# every module should have a LOG object
LOG = logging.getLogger(__file__)


def main():
    """
    The main method.
    """

    # Read in the command line options
    args, work_dir = argument_parser_image()

    input_files = args.inputs
    datatype = args.datatype
    dataset = args.dataset
    stride = args.stride
    pressure = args.pressure
    lat_0 = args.lat_0
    lon_0 = args.lon_0
    bounding_lat = args.bounding_lat
    extent = args.extent
    plotMin = args.plotMin
    plotMax = args.plotMax
    plot_type = args.plot_type
    doScatterPlot = args.doScatterPlot
    pointSize = args.pointSize
    proj = args.proj
    map_res = args.map_res
    output_file = args.output_file
    outputFilePrefix = args.outputFilePrefix
    dpi = args.dpi

    # Obtain matched lists of the geolocation and product files
    LOG.info("Input file(s):\n\t{}".format("\n\t".join(input_files)))
    input_file_list = granuleFiles(input_files)

    if input_file_list == []:
        LOG.warn(
            'No files match the input file glob "{}", aborting.'.format(input_files)
        )
        sys.exit(1)

    LOG.debug("Input file(s):\n\t{}".format("\n\t".join(input_file_list)))

    # Read in the input file, and return a dictionary containing the required
    # data

    LOG.info("Input pressure = {}".format(pressure))

    # Get all of the required command line options for the required souder package...
    LOG.info("datatype = {}".format(datatype))
    LOG.info("dataset = {}".format(dataset))
    sounder_args = (input_file_list, dataset, plot_type)
    sounder_kwargs = {
        "pres_0": pressure,
        "lat_0": None,
        "lon_0": None,
    }

    # Get a reference to the desired sounder package class, and instantiate
    sounder_package_ref = sounder_packages.sounder_package_ref[datatype]
    sounder_obj = sounder_package_ref(*sounder_args, **sounder_kwargs)

    LOG.debug("sounder_obj.pres_0 = {}".format(sounder_obj.pres_0))

    pres_0 = sounder_obj.pres_0
    lats = sounder_obj.datasets["lat"]["data"]
    lons = sounder_obj.datasets["lon"]["data"]
    data = sounder_obj.datasets[dataset]["data"]
    data_mask = sounder_obj.datasets[dataset]["data_mask"]

    input_file = os.path.basename(input_file_list[0])

    # Get the dataset options
    try:
        dataset_options = sounder_image_data.Dataset_Options.data[dataset]
    except KeyError:
        dataset_options = sounder_image_data.Dataset_Options.data["unknown"]
        dataset_options["name"] = dataset

    for key in dataset_options.keys():
        LOG.debug("dataset_options['{}'] = {}".format(key, dataset_options[key]))

    # Determine the filename
    if dataset == "ctp" or dataset == "ctt" or plot_type == "slice":
        LOG.info("pres_0 = {}".format(pres_0))
        LOG.info("elev_0 = {}".format(elev_0))
        vertical_type = "elev" if elev_0 != None else "pres"
        LOG.info("vertical_type = {}".format(vertical_type))
        file_suffix = "{}_{}_{}".format(datatype, dataset, vertical_type)
    else:
        if pres_0 != None:
            file_suffix = "{}_{}_{}mb".format(datatype, dataset, int(pres_0))
        else:
            file_suffix = "{}_{}_{}ft".format(datatype, dataset, int(elev_0))

    if output_file == None and outputFilePrefix == None:
        output_file = "{}.{}.png".format(input_file, file_suffix)
    if output_file != None and outputFilePrefix == None:
        pass
    if output_file == None and outputFilePrefix != None:
        output_file = "{}_{}.png".format(outputFilePrefix, file_suffix)
    if output_file != None and outputFilePrefix != None:
        output_file = "{}_{}.png".format(outputFilePrefix, file_suffix)

    dataset = args.dataset

    # Set the plot styles
    plot_style_options = set_plot_styles(sounder_obj, dataset, dataset_options, args)

    # Get pointers to the desired plotting routines
    plot_image = plot_style_options["plot_image"]
    plot_map = plot_style_options["plot_map"]
    plot_slice = plot_style_options["plot_slice"]

    plot_options = {}
    plot_options["lat_0"] = lat_0
    plot_options["lon_0"] = lon_0
    plot_options["bounding_lat"] = bounding_lat
    plot_options["extent"] = extent
    plot_options["proj"] = proj
    plot_options["pkg_name"] = sounder_obj.pkg_name

    # Create the plot
    if plot_type == "image":
        retval = plot_map(
            lats,
            lons,
            data,
            data_mask,
            output_file,
            dataset_options,
            plot_style_options,
            plot_options,
        )

    print("")
    return retval


if __name__ == "__main__":
    sys.exit(main())
