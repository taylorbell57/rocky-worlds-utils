#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to facilitate numerical calculations.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
- Mees B Fix <<mfix@stsci.edu>>
"""

from astroquery.mast import MastMissions
from crds import assign_bestrefs
import glob
import numpy as np


__all__ = [
    "nearest_index",
]


def nearest_index(array, target_value):
    """
    Finds the index of a value in ``array`` that is closest to ``target_value``.

    Parameters
    ----------
    array : ``numpy.array``
        Target array.
    target_value : ``float``
        Target value.

    Returns
    -------
    index : ``int``
        Index of the value in ``array`` that is closest to ``target_value``.
    """
    index = array.searchsorted(target_value)
    index = np.clip(index, 1, len(array) - 1)
    left = array[index - 1]
    right = array[index]
    index -= target_value - left < right - target_value
    return index


def assign_reffiles(files, task_verbosity=0):
    """Assign and download reference files

    files : list
        List of absolute path for files to pull refs for

    task_verbosity : int
        Level of output from assign_bestrefs [default: 0]
    """
    assign_bestrefs(files, sync_references=True, verbosity=task_verbosity)


def download_cos_data():
    hst = MastMissions(mission="hst")

    query = hst.query_criteria(sci_data_set_name="lcil2a010")
    products = hst.get_product_list(query)
    products = products.filled()
    products = products[
        [
            filename.endswith(".fits")
            and suffix
            in {
                "X1D",
                "CORRTAG",
                "CORRTAG_A",
                "CORRTAG_B",
                "RAWTAG",
                "RAWTAG_A",
                "RAWTAG_B",
            }
            for suffix, filename in zip(products["file_suffix"], products["filename"])
        ]
    ]

    hst.download_products(products, flat=True)

    files = glob.glob("lcil2ajnq*")
    assign_reffiles(files)


def download_stis_data():
    hst = MastMissions(mission="hst")

    query = hst.query_criteria(sci_data_set_name="oev303010")
    products = hst.get_product_list(query)
    products = products.filled()
    products = products[
        [
            filename.endswith(".fits")
            and suffix
            in {
                "X1D",
                "WAV",
                "TAG",
                "FLT",
            }
            for suffix, filename in zip(products["file_suffix"], products["filename"])
        ]
    ]

    hst.download_products(products, flat=True)

    files = glob.glob("oev303010*")
    assign_reffiles(files)
