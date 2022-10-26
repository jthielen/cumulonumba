# Copyright (c) 2022 Cumulonumba Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Utility routines for numerical operations.

Module contains functions reused with modification from MetPy (Copyright (c) 2008-2020, MetPy
Developers) under the terms of the BSD 3-Clause License (reproduced below).

---

BSD 3-Clause License

Copyright (c) 2008-2020, MetPy Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numba
import numpy as np


@numba.njit
def _isclose(a, b, rtol=1.e-5, atol=1.e-8):
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


@numba.njit
def _nan_mask(*inputs):
    """Find NaNs in arrays that cause issues with calculations.

    Takes a variable number of arguments and returns a single mask covering all inputs.

    (Reused with modification from MetPy, see module header.)
    """
    input_iter = iter(inputs)
    mask = np.isnan(next(input_iter))
    for v in input_iter:
        mask |= np.isnan(v)
    return mask


@numba.njit
def _greater_or_close(a, value):
    r"""Compare values for greater or close to boolean masks.

    Returns a boolean mask for values greater than or equal to a target within a specified
    absolute or relative tolerance (as in :func:`numpy.isclose`).

    (Reused with modification from MetPy, see module header.)

    Parameters
    ----------
    a : array-like
        Array of values to be compared
    value : float
        Comparison value

    Returns
    -------
    array-like
        Boolean array where values are greater than or nearly equal to value.

    """
    return (a > value) | _isclose(a, value)


@numba.njit
def _less_or_close(a, value):
    r"""Compare values for less or close to boolean masks.

    Returns a boolean mask for values less than or equal to a target within a specified
    absolute or relative tolerance (as in :func:`numpy.isclose`).

    (Reused with modification from MetPy, see module header.)

    Parameters
    ----------
    a : array-like
        Array of values to be compared
    value : float
        Comparison value

    Returns
    -------
    array-like
        Boolean array where values are less than or nearly equal to value

    """
    return (a < value) | _isclose(a, value)


@numba.njit
def _nearest_intersection_idx(a, b):
    """Determine the index of the point just before two lines with common x values.

    (Reused with modification from MetPy, see module header.)

    Parameters
    ----------
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2

    Returns
    -------
        An array of indexes representing the index of the values
        just before the intersection(s) of the two lines.

    """
    # Difference in the two y-value sets
    difference = a - b

    # Determine the point just before the intersection of the lines
    # Will return multiple points for multiple intersections
    sign_change_idx, = np.nonzero(np.diff(np.sign(difference)))

    return sign_change_idx


@numba.njit
def _find_intersections(x, a, b, direction=0, log_x=False):
    """Calculate the best estimate of intersection.

    Calculates the best estimates of the intersection of two y-value
    data sets that share a common x-value set.

    (Reused with modification from MetPy, see module header.)

    Parameters
    ----------
    x : array-like
        1-dimensional array of numeric x-values
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2
    direction : int, optional
        specifies direction of crossing. 0, 1 (a becoming greater than b), or -1 (b becoming
        greater than a). Defaults to 0.
    log_x : bool, optional
        Use logarithmic interpolation along the `x` axis (i.e. for finding intersections
        in pressure coordinates). Default is False.

    Returns
    -------
        Stacked (x, y) array-like with the x and y coordinates of the
        intersections of the lines.

    """
    # Change x to logarithmic if log_x=True
    if log_x is True:
        x = np.log(x)

    # Find the index of the points just before the intersection(s)
    nearest_idx = _nearest_intersection_idx(a, b)
    next_idx = nearest_idx + 1

    # Determine the sign of the change
    sign_change = np.sign(a[next_idx] - b[next_idx])

    # x-values around each intersection
    x0 = x[nearest_idx]
    x1 = x[next_idx]

    # y-values around each intersection for the first line
    a0 = a[nearest_idx]
    a1 = a[next_idx]

    # y-values around each intersection for the second line
    b0 = b[nearest_idx]
    b1 = b[next_idx]

    # Calculate the x-intersection. This comes from finding the equations of the two lines,
    # one through (x0, a0) and (x1, a1) and the other through (x0, b0) and (x1, b1),
    # finding their intersection, and reducing with a bunch of algebra.
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1
    intersect_x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)

    # Calculate the y-intersection of the lines. Just plug the x above into the equation
    # for the line through the a points. One could solve for y like x above, but this
    # causes weirder unit behavior and seems a little less good numerically.
    intersect_y = ((intersect_x - x0) / (x1 - x0)) * (a1 - a0) + a0

    # If there's no intersections, return
    if len(intersect_x) == 0:
        return np.stack((intersect_x, intersect_y))

    # Return x to linear if log_x is True
    if log_x is True:
        intersect_x = np.exp(intersect_x)

    # Check for duplicates
    duplicate_mask = (np.ediff1d(intersect_x, to_end=1) != 0)

    # Make a mask based on the direction of sign change desired
    if direction == 1:
        mask = sign_change > 0
    elif direction == -1:
        mask = sign_change < 0
    elif direction == 0:
        return np.stack((intersect_x[duplicate_mask], intersect_y[duplicate_mask]))
    else:
        raise ValueError('Direction must be 1, 0, or -1.')

    return np.stack((intersect_x[mask & duplicate_mask], intersect_y[mask & duplicate_mask]))


@numba.njit
def _find_append_zero_crossings(x, y):
    r"""
    Find and interpolate zero crossings.

    Estimate the zero crossings of an x,y series and add estimated crossings to series,
    returning a sorted array with no duplicate values.

    (Reused with modification from MetPy, see module header.)
    """
    crossings = _find_intersections(x[1:], y[1:], np.zeros_like(y[1:]), log_x=True)
    x = np.concatenate((x, crossings[0]))
    y = np.concatenate((y, crossings[1]))

    # Resort so that data are in order
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Remove duplicate data points if there are any
    keep_idx = np.ediff1d(x, to_end=[1]) > 1e-6
    x = x[keep_idx]
    y = y[keep_idx]
    return np.stack((x, y))


@numba.njit
def _steffensens_method(func, x0, constants, xtol=1e-08, maxiter=500):
    """Steffensen's Method for accelerated fixed point interation.

    Direct implementation of Algorithm 2.6 from Burden and Faires "Numerical Analysis", 9th
    edition, pg. 89. API signature borrowed from SciPy (scipy.optimize.fixed_point).
    """
    p0 = x0
    for i in range(1, maxiter + 1):
        p1 = func(p0, constants)
        p2 = func(p1, constants)
        p = p0 - (p1 - p0)**2 / (p2 - 2 * p1 + p0)
        if np.abs(p - p0) < xtol:
            return p  # convergence
        elif np.isnan(p) or np.isinf(p):
            return p2  # early stop on delta-square == 0
        else:
            p0 = p
    return p0
