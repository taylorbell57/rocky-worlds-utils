#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to process HST/STIS data.
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import stistools
import os

from astropy.io import fits
from astropy.stats import poisson_conf_interval
from scipy.integrate import simpson

__all__ = ["integrate_flux", "timetag_split"]


# This function integrates the flux within a wavelength range for given arrays
# for wavelength and flux
def integrate_flux(wavelength_range, wavelength_list, flux_list, gross_list,
                   net_list, exposure_time):
    """

    Parameters
    ----------
    wavelength_range
    wavelength_list
    flux_list
    gross_list
    net_list
    exposure_time

    Returns
    -------
    integrated_flux
    integrated_error
    """
    # Since the pixels may not range exactly in the interval above,
    # we will need to deal with fractional pixels. But first, let's
    # integrate the pixels that are fully inside the range
    full_indexes = np.where((wavelength_list > wavelength_range[0]) & (
                wavelength_list < wavelength_range[1]))[0]
    full_pixel_flux = simpson(flux_list[full_indexes],
                              wavelength_list[full_indexes])

    # And now we deal with the flux in the fractional pixels
    index_left = full_indexes[0]
    index_right = full_indexes[-1]
    pixel_width_left = wavelength_list[index_left] - wavelength_list[
        index_left - 1]
    fraction_left = 1 - (wavelength_range[0] - wavelength_list[
        index_left - 1]) / pixel_width_left
    pixel_width_right = wavelength_list[index_right + 1] - wavelength_list[
        index_right]
    fraction_right = 1 - (wavelength_list[index_right + 1] - wavelength_range[
        1]) / pixel_width_right
    fractional_flux_left = flux_list[index_left - 1] * fraction_left
    fractional_flux_right = flux_list[index_right + 1] * fraction_right

    # There is a bug in stistools that overestimates errors of time-tag split
    # subexposures. So we will need to calculate them manually here
    # based on the raw counts in the extracted spectra. Here we go.
    full_pixel_gross = np.sum(gross_list[full_indexes])
    fractional_gross_left = gross_list[index_left - 1] * fraction_left
    fractional_gross_right = gross_list[index_right + 1] * fraction_right
    integrated_gross = (full_pixel_gross + fractional_gross_left +
                        fractional_gross_right)
    sensitivity = flux_list[full_indexes] / net_list[full_indexes]
    mean_sensitivity = np.mean(sensitivity)
    gross_error = (poisson_conf_interval(integrated_gross,
                                        interval='sherpagehrels') -
                   integrated_gross)
    integrated_error = (-gross_error[0] + gross_error[
        1]) / 2 * mean_sensitivity / exposure_time  # Take the average error for
    # simplicity

    integrated_flux = (full_pixel_flux + fractional_flux_left +
                       fractional_flux_right)
    return integrated_flux, integrated_error


# Divide exposures into sub-exposures for TIME-TAG data and process them
def timetag_split(dataset, prefix, output_dir, target_snr,
                       max_n_subexposures, clean_intermediate_steps=True):
    """

    Parameters
    ----------
    dataset
    prefix
    output_dir
    target_snr
    max_n_subexposures
    clean_intermediate_steps

    Returns
    -------

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
