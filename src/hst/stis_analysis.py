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

import stistools
import os

from astropy.io import fits

__all__ = ["timetag_split", ]


# Divide exposures into sub-exposures for TIME-TAG data and process them
def timetag_split(dataset, prefix, output_dir, n_subexposures=10,
                  temporal_resolution=None, clean_intermediate_steps=True,
                  overwrite=False, output_file_name=None):
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
    """
    # Initial checks
    if output_dir == prefix:
        raise ValueError('The output directory must be different '
                                    'from the prefix.')
    if output_file_name is None:
        output_file = str(output_dir) + '/' + dataset + '_ts_x1d.fits'
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
    x1d_data = fits.getdata(str(prefix) + '/' + x1d_filename)

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

    # Process the time series
    stistools.x1d.x1d(str(output_dir) + '/' + dataset + '_ts_flt.fits',
                      output=output_file,
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
