#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to time series from HST STIS and COS spectra.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np

from astropy.stats import poisson_conf_interval
from scipy.integrate import simpson

__all__ = ["integrate_flux", ]


# This function integrates the flux within a wavelength range for given arrays
# for wavelength and flux
def integrate_flux(wavelength_range, wavelength_list, flux_list, gross_list,
                   net_list, exposure_time):
    """
    Integrate fluxes from HST STIS and COS spectra within a range of
    wavelengths. This code takes into account fractional pixels and correctly
    estimates uncertainties in the Poisson counting regime.

    Parameters
    ----------
    wavelength_range : array-like
        List, array or tuple of two floats containing the start and end of the
        wavelength range to be integrated.

    wavelength_list : ``numpy.ndarray``
        Array containing the wavelengths of the spectrum.

    flux_list : ``numpy.ndarray``
        Array containing the flux values of the spectrum.

    gross_list : ``numpy.ndarray``
        Array containing the gross counts of the spectrum.

    net_list : ``numpy.ndarray``
        Array containing the net count rates of the spectrum.

    exposure_time : ``float``
        Exposure time in seconds.

    Returns
    -------
    integrated_flux : ``float``
        Integrated flux.

    integrated_error : ``float``
        Uncertainty of the integrated flux.
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