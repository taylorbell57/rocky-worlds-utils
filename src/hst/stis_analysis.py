#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to process HST/STIS data.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import stistools
import os

from astropy.io import fits

__all__ = ["timetag_split", ]


# Divide exposures into sub-exposures for TIME-TAG data and process them
def timetag_split(dataset, prefix, output_dir, target_snr,
                       max_n_subexposures, clean_intermediate_steps=True):
    """
    Creates a new time-series of x1d fits files of an HST/STIS dataset.

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
    x1d_filename = dataset + '_x1d.fits'

    x1d_header_1 = fits.getheader(str(prefix) + '/' + x1d_filename, 1)
    x1d_data = fits.getdata(str(prefix) + '/' + x1d_filename)

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

    # We will use inttag to break down the full exposure into subexposures
    # XXX WARNING: Dependending on the number of subexposures, the file can be
    # very large! XXX
    tag_filename = dataset + '_tag.fits'

    stistools.inttag.inttag(tagfile=str(prefix) + '/' + tag_filename,
                            output=str(
                                output_dir) + '/' + dataset + '_ts_raw.fits',
                            rcount=n_subexposures)

    # And now we run the raw file through calstis, more specifically basic2d,
    # which does not extract the spectrum yet
    stistools.basic2d.basic2d(
        str(output_dir) + '/' + dataset + '_ts_raw.fits',
        str(output_dir) + '/' + dataset + '_ts_flt.fits')

    stistools.wavecal.wavecal(
        str(output_dir) + '/' + dataset + '_ts_flt.fits',
        wavecal=str(prefix) + '/' + dataset + '_wav.fits')

    # Now we extract the spectrum in the same location as the original full
    # exposure and turn off the automatic trace finding (set by the maxsrch
    # parameter)
    extract_yloc = x1d_data['A2CENTER'][0]
    extract_size = x1d_data['EXTRSIZE'][0]
    extract_bk1_size = x1d_data['BK1SIZE'][0]
    extract_bk2_size = x1d_data['BK2SIZE'][0]
    extract_bk1_offset = x1d_data['BK1OFFST'][0]
    extract_bk2_offset = x1d_data['BK2OFFST'][0]

    stistools.x1d.x1d(str(output_dir) + '/' + dataset + '_ts_flt.fits',
                      output=str(output_dir) + '/' + dataset + '_ts_x1d.fits',
                      maxsrch=0,
                      # Setting this to zero ensures that code will not look for
                      # trace location
                      a2center=extract_yloc,
                      extrsize=extract_size,
                      backcorr='perform',
                      bk1size=extract_bk1_size,
                      bk2size=extract_bk2_size,
                      bk1offst=extract_bk1_offset,
                      bk2offst=extract_bk2_offset
                      )

    # Clean intermediate steps if requested
    if clean_intermediate_steps is True:
        os.remove(str(output_dir) + '/' + dataset + '_ts_flt.fits')
        os.remove(str(output_dir) + '/' + dataset + '_ts_raw.fits')
    else:
        pass

    return n_subexposures
