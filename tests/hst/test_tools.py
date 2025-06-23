#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for `tools` module.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""

import sys
import numpy as np

sys.path.insert(1, 'rocky-worlds-utils/hst')
import tools


# Test the nearest_index function
def test_nearest_index():
    array = np.linspace(1, 10, 10)
    ind = tools.nearest_index(array, np.pi)
    assert ind == 2
