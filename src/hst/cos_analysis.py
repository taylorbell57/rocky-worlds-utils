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

from calcos.x1d import concatenateSegments
from astropy.io import fits


# Change the following if you do not want to use all CPUs
__N_PROCESSES = multiprocessing.cpu_count()

__all__ = ["timetag_split", ]


# Divide exposures into sub-exposures for TIME-TAG data and process them
def timetag_split(dataset, prefix, output_dir, target_snr,
                  max_n_subexposures, clean_intermediate_steps=True):
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

    target_snr : ``float``
        Minimum signal-to-noise ratio (SNR) that each sub-exposure in the time
        series should have. This SNR is calculated by integrating the counts
        in the entire spectrum and then taking the square root of this value.

    max_n_subexposures : ``int``
        Maximum number of subexposures to produce in the time series. This is
        useful for avoiding large file sizes.

    clean_intermediate_steps : ``bool``, optional
        Sets whether intermediate steps should be cleaned up after each run.
        Default is ``True``.

    Returns
    -------
    n_subexposures : ``int``
        Number of subexposures in the time series.
    """
    # Checks whether output directory is different from prefix
    if output_dir == prefix:
        raise ValueError('The output directory must be different '
                                    'from the prefix.')

    x1d_filename = dataset + '_x1d.fits'

    x1d_header_0 = fits.getdata(str(prefix) + '/' + x1d_filename, 0)
    x1d_header_1 = fits.getheader(str(prefix) + '/' + x1d_filename, 1)
    x1d_data = fits.getdata(str(prefix) + '/' + x1d_filename)

    # Perform some other checks
    if x1d_header_0['OBSTYPE'] != 'SPECTROSCOPIC':
        raise ValueError('Observation type must be SPECTROSCOPIC.')
    if x1d_header_0['OBSMODE'] != 'TIME-TAG':
        raise ValueError('Observing mode must be TIME-TAG.')

    # Extracting some useful information
    exp_time = x1d_header_1['EXPTIME']
    net_counts = x1d_data['NET'][0] * exp_time

    # Calculate integrated counts in the spectrum
    int_counts = np.sum(net_counts)
    rough_noise_level = np.sqrt(int_counts)
    rough_snr_full_exposure = int_counts / rough_noise_level

    # If the SNR of the full exposure is too low in the first place,
    # then simply do 5 subexposures as a standard
    if rough_snr_full_exposure < 10:
        n_subexposures = 5
    else:
        target_counts_per_subexposure = target_snr ** 2
        n_subexposures = round(int_counts / target_counts_per_subexposure)

    if n_subexposures > max_n_subexposures:
        n_subexposures = max_n_subexposures
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

    with multiprocessing.Pool(processes=__N_PROCESSES) as pool:
        _ = pool.starmap(
            calcos.calcos, [(subexposure, output_dir + 'temp/')
                            for subexposure in split_list])

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
    new_fits_filename = dataset + '_ts_x1d.fits'
    with fits.open(output_dir + dataset + '_1_x1d.fits') as hdu:
        primary_header = hdu[0].header
        primary_data = hdu[0].data
        bintable_header = hdu[1].header
        bintable_data = hdu[1].data
        primary_header['FILENAME'] = new_fits_filename

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
    hdul.writeto(output_dir + new_fits_filename)

    # Remove last intermediate steps
    if clean_intermediate_steps is True:
        remove_list = glob.glob(output_dir + dataset + '_?_x1d.fits')
        for item in remove_list:
            os.remove(item)
        remove_list = glob.glob(output_dir + dataset + '_?_corrtag_*.fits')
        for item in remove_list:
            os.remove(item)

    return n_subexposures