#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to process HST/COS data.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import costools
import os
import glob
import calcos
import shutil
import multiprocessing
import warnings

from calcos.x1d import concatenateSegments
from astropy.io import fits
from IPython import get_ipython

__all__ = ["timetag_split", ]


# Turn off parallelization if Jupyter notebook is being used
def _is_running_in_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter Notebook or JupyterLab
        elif shell == 'TerminalInteractiveShell':
            return False  # IPython terminal
        else:
            return False  # Other environment
    except NameError:
        return False      # Standard Python interpreter'
if _is_running_in_jupyter():
    warnings.warn("Jupyter environment detected: Parallelized COS data "
                  "reduction is turned off.")
    _multiprocess = False
    __N_PROCESSES = 1
else:
    __N_PROCESSES = multiprocessing.cpu_count()
    print('{} processing units available for parallelized COS data '
          'reduction.'.format(__N_PROCESSES))
    _multiprocess = True


# Divide exposures into sub-exposures for TIME-TAG data and process them
def timetag_split(dataset, prefix, output_dir, n_subexposures=10,
                  temporal_resolution=None, clean_intermediate_steps=True,
                  overwrite=False, output_file_name=None,
                  multiprocess=_multiprocess, n_cpus=__N_PROCESSES):
    """
    Creates a new time-series of x1d fits files of an HST/COS dataset.

    Parameters
    ----------
    dataset : ``str``
        Dataset name (example: ``ld9m17d3q``).

    prefix : ``str``
        Fixed path to dataset directory.

    output_dir : ``str``
        Path to output directory.

    n_subexposures : ``int``
        Number of subexposures to produce in the time series. This value is
        overridden if the ``temporal_resolution`` is defined. Default is 10.

    temporal_resolution : ``float``, ``int`` or ``None``, optional
        Temporal resolution in unit of seconds for the time series. If ``None``,
        then the temporal resolution will be defined by ``n_subexposures``.
        If not ``None``, then the number of subexposures is overridden by the
        temporal resolution. Default is ``None``.

    clean_intermediate_steps : ``bool``, optional
        Sets whether intermediate steps should be cleaned up after each run.
        Default is ``True``.

    overwrite : ``bool``, optional
        Overwrite the output file if it already exists. Default is ``False``.

    output_file_name : ``str`` or ``None``, optional
        Sets the name of the output file. If set, it must contain the extension
        ``.fits``. If ``None``, then the default output file name is
        ``[dataset]_ts_x1d.fits``. Default is ``None``.

    multiprocess : ``bool``, optional
        Sets whether multiprocessing should be used. If using Jupyter notebooks,
        it is recommended to set this to ``False``. Default is ``True``.

    n_cpus : ``int``, optional
        Number of CPU cores to use in data reduction. Default is the maximum
        number of available units in your system.
    """
    # Initial checks
    if output_dir == prefix:
        raise ValueError('The output directory must be different '
                                    'from the prefix.')
    if output_file_name is None:
        output_file_name = dataset + '_ts_x1d.fits'
        output_file = str(output_dir) + '/' + output_file_name
    elif output_file_name[:-5] != '.fits':
        raise ValueError('The extension of the output file must be .fits.')
    else:
        output_file = str(output_dir) + '/' + output_file_name

    # Test if output file exists, and if it does, delete it if overwrite is True
    if os.path.isfile(output_file):
        if overwrite is False:
            raise IOError('Time-tag split output file already exists.')
        else:
            os.remove(output_file)
    else:
        pass

    x1d_filename = dataset + '_x1d.fits'

    x1d_header_0 = fits.getheader(str(prefix) + '/' + x1d_filename, 0)
    x1d_header_1 = fits.getheader(str(prefix) + '/' + x1d_filename, 1)

    # Perform some other checks
    if x1d_header_0['OBSTYPE'] != 'SPECTROSCOPIC':
        raise ValueError('Observation type must be SPECTROSCOPIC.')
    if x1d_header_0['OBSMODE'] != 'TIME-TAG':
        raise ValueError('Observing mode must be TIME-TAG.')

    # Extracting some useful information
    exp_time = x1d_header_1['EXPTIME']

    # Define the number of sub-exposures
    if temporal_resolution is not None:
        n_subexposures = int(round(exp_time / temporal_resolution))
    else:
        pass

    # The CalCOS pipeline requires time bins to perform the time-tag split
    time_bins = np.linspace(0, exp_time, n_subexposures + 1)

    # We will use splittag to break down the full exposure into subexposures
    tag_filename_a = prefix + dataset + '_corrtag_a.fits'
    tag_filename_b = prefix + dataset + '_corrtag_b.fits'
    time_list = ""
    for time in time_bins:
        time_list += str(time) + ', '

    costools.splittag.splittag(infiles=tag_filename_a,
                               outroot=output_dir + dataset,
                               time_list=time_list)
    costools.splittag.splittag(infiles=tag_filename_b,
                               outroot=output_dir + dataset,
                               time_list=time_list)

    # Run the tag split data in the CalCOS pipeline
    # Some hack necessary to avoid IO error when using x1dcorr
    split_list = glob.glob(output_dir + dataset + '_?_corrtag_?.fits')
    for item in split_list:
        char_list = list(item)
        char_list.insert(-13, char_list.pop(-6))
        char_list.insert(-12, char_list.pop(-6))
        link = ""
        new_item = link.join(char_list)
        os.rename(item, new_item)

    # Extract the tag-split spectra
    split_list = glob.glob(output_dir + dataset + '_?_?_corrtag.fits')

    if multiprocess is True:
        with multiprocessing.Pool(processes=n_cpus) as pool:
            _ = pool.starmap(
                calcos.calcos, [(subexposure, output_dir + 'temp/')
                                for subexposure in split_list])
    else:
        for subexposure in split_list:
            _ = calcos.calcos(subexposure, output_dir  + 'temp/')

    # Move x1ds to output folder
    split_list = glob.glob(output_dir + 'temp/' + dataset + '*_x1d.fits')
    for subexposure in split_list:
        shutil.move(subexposure, output_dir + dataset + subexposure[-13:])

    # Clean the intermediate steps files
    if clean_intermediate_steps is True:
        shutil.rmtree(output_dir + 'temp/')
    else:
        pass

    # Return the filenames back to normal
    split_list = glob.glob(output_dir + dataset + '*_corrtag.fits')
    for subexposure in split_list:
        char_list = list(subexposure)
        char_list.insert(-5, char_list.pop(-15))
        char_list.insert(-5, char_list.pop(-15))
        link = ""
        new_subexposure = link.join(char_list)
        os.rename(subexposure, new_subexposure)
    split_list = glob.glob(output_dir + dataset + '*_x1d.fits')
    for subexposure in split_list:
        char_list = list(subexposure)
        char_list.insert(-5, char_list.pop(-9))
        char_list.insert(-5, char_list.pop(-10))
        link = ""
        new_subexposure = link.join(char_list)
        os.rename(subexposure, new_subexposure)

    # Concatenate segments `a` and `b` of the detector
    for i in range(n_subexposures):
        x1d_list = glob.glob(output_dir + dataset + '_%i_x1d_?.fits' % (i + 1))
        concatenateSegments(x1d_list, output_dir + dataset +
                            '_%i' % (i + 1) + '_x1d.fits')

    # Remove more intermediate steps
    if clean_intermediate_steps is True:
        remove_list = glob.glob(output_dir + dataset + '_?_x1d_?.fits')
        for item in remove_list:
            os.remove(item)

    # Merge the splits into a single time-series fits file, like it's done in
    # the STIS code
    # new_fits_filename = dataset + '_ts_x1d.fits'
    with fits.open(output_dir + dataset + '_1_x1d.fits') as hdu:
        primary_header = hdu[0].header
        primary_data = hdu[0].data
        bintable_header = hdu[1].header
        bintable_data = hdu[1].data
        primary_header['FILENAME'] = output_file_name

    new_primary_hdu = fits.PrimaryHDU(header=primary_header, data=primary_data)
    new_bintable_hdu = fits.BinTableHDU(header=bintable_header,
                                        data=bintable_data)
    hdu_list = [new_primary_hdu, new_bintable_hdu]

    for i in range(n_subexposures - 1):
        with fits.open(output_dir + dataset + '_{}_x1d.fits'.format(
                str(i + 1))) as hdu:
            next_bintable_header = hdu[1].header
            next_bintable_data = hdu[1].data
        next_bintable_hdu = fits.BinTableHDU(header=next_bintable_header,
                                             data=next_bintable_data)
        hdu_list.append(next_bintable_hdu)

    # Write time series to a new fits file
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(output_file)

    # Remove last intermediate steps
    if clean_intermediate_steps is True:
        remove_list = glob.glob(output_dir + dataset + '_?_x1d.fits')
        for item in remove_list:
            os.remove(item)
        remove_list = glob.glob(output_dir + dataset + '_?_corrtag_*.fits')
        for item in remove_list:
            os.remove(item)
