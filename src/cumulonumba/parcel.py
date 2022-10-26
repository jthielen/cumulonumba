# Copyright (c) 2022 Cumulonumba Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Calculations for atmospheric parcels.

Reused with modification from MetPy (Copyright (c) 2008-2020, MetPy Developers) under the
terms of the BSD 3-Clause License (reproduced below).

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

from enum import Enum

import numba
import numpy as np

from .constants import Rd
from .thermo import _lcl_scalar, _moist_lapse_1d, dry_lapse
from .util import (
    _find_append_zero_crossings,
    _find_intersections,
    _greater_or_close,
    _isclose,
    _less_or_close,
    _nan_mask
)


class IntersectionChoice(Enum):
    TOP = 1
    BOTTOM = 2
    WIDE = 3
    MOST_CAPE = 4


class IntersectionType(Enum):
    LFC = 1
    EL = 2


@numba.njit
def _lfc(
    pressure, temperature, dewpoint, parcel_temperature_profile, which=IntersectionChoice.TOP
):
    r"""Calculate the level of free convection (LFC).

    This works by finding the first intersection of the ideal parcel path and
    the measured parcel temperature. If this intersection occurs below the LCL,
    the LFC is determined to be the same as the LCL, based upon the conditions
    set forth in [USAF1990]_, pg 4-14, where a parcel must be lifted dry adiabatically
    to saturation before it can freely rise.

    Parameters
    ----------
    pressure : 1D array-like
        Atmospheric pressure, in Pa.

    temperature : 1D array-like
        Temperature at the levels given by `pressure`, in K.

    dewpoint : 1D array-like
        Dewpoint at the levels given by `pressure`, in K.

    parcel_temperature_profile: 1D array-like
        The parcel's temperature profile from which to calculate the LFC, in K.

    which: IntersectionChoice, optional
        Pick which LFC to return. Options are TOP, BOTTOM, WIDE, MOST_CAPE;
        TOP returns the lowest-pressure LFC (default),
        BOTTOM returns the highest-pressure LFC,
        WIDE returns the LFC whose corresponding EL is farthest away,
        MOST_CAPE returns the LFC that results in the most CAPE in the profile.

    Returns
    -------
    numpy.ndarray
        Length 2 array containing LFC pressure and temperature, in Pa and K, respectively.

    Notes
    -----
    TODO: original metpy implementation worked independently of CAPE/CIN (and so included nan
    masking). decide how you wish to handle that? also had dewpoint_start argument, which was
    unused by CAPE/CIN, so dropped.

    """
    # Set initial dewpoint
    dewpoint_start = dewpoint[0]

    # The parcel profile and data may have the same first data point.
    # If that is the case, ignore that point to get the real first
    # intersection for the LFC calculation. Use logarithmic interpolation.
    if _isclose(parcel_temperature_profile[0], temperature[0]):
        x, y = _find_intersections(
            pressure[1:],
            parcel_temperature_profile[1:],
            temperature[1:],
            direction=1,
            log_x=True
        )
    else:
        x, y = _find_intersections(
            pressure, parcel_temperature_profile, temperature, direction=1, log_x=True
        )

    # Compute LCL for this parcel for future comparisons
    this_lcl = _lcl_scalar(pressure[0], parcel_temperature_profile[0], dewpoint_start)

    # The LFC could:
    # 1) Not exist
    # 2) Exist but be equal to the LCL
    # 3) Exist and be above the LCL

    # LFC does not exist or is LCL
    if len(x) == 0:
        # Is there any positive area above the LCL?
        mask = pressure < this_lcl[0]
        if np.all(_less_or_close(parcel_temperature_profile[mask], temperature[mask])):
            return np.array((np.nan, np.nan))
        else:  # LFC = LCL
            return this_lcl
    # LFC exists. Make sure it is no lower than the LCL
    else:
        idx = x < this_lcl[0]
        # LFC height < LCL height, so set LFC = LCL
        if not np.any(idx):
            el_pressure = _find_intersections(
                pressure[1:],
                parcel_temperature_profile[1:],
                temperature[1:],
                direction=-1,
                log_x=True
            )[0]
            if np.min(el_pressure) > this_lcl[0]:
                return np.array((np.nan, np.nan))
            else:
                return this_lcl
        # Otherwise, find all LFCs that exist above the LCL
        # What is returned depends on which flag as described in the docstring
        else:
            return _multiple_el_lfc_options(
                x,
                y,
                idx,
                which,
                pressure,
                parcel_temperature_profile,
                temperature,
                dewpoint,
                intersect_type=IntersectionType.LFC
            )


@numba.njit
def _multiple_el_lfc_options(
    intersect_pressures,
    intersect_temperatures,
    valid_x,
    which,
    pressure,
    parcel_temperature_profile,
    temperature,
    dewpoint,
    intersect_type
):
    """Choose which ELs and LFCs to return from a sounding."""
    p_list = intersect_pressures[valid_x]
    t_list = intersect_temperatures[valid_x]
    if which == IntersectionChoice.BOTTOM:
        xy = np.array((p_list[0], t_list[0]))
    elif which == IntersectionChoice.TOP:
        xy = np.array((p_list[-1], t_list[-1]))
    elif which == IntersectionChoice.WIDE:
        xy = _wide_option(
            intersect_type, p_list, t_list, pressure, parcel_temperature_profile, temperature
        )
    elif which == IntersectionChoice.MOST_CAPE:
        xy = _most_cape_option(
            intersect_type,
            p_list,
            t_list,
            pressure,
            temperature,
            dewpoint,
            parcel_temperature_profile
        )
    else:
        raise ValueError('Invalid option for "which". Valid options are "TOP", "BOTTOM", '
                         '"WIDE", and "MOST_CAPE".')
    return xy


@numba.njit
def _wide_option(
    intersect_type, p_list, t_list, pressure, parcel_temperature_profile, temperature
):
    """Calculate the LFC or EL that produces the greatest distance between these points."""
    # zip the LFC and EL lists together and find greatest difference
    if intersect_type == IntersectionType.LFC:
        # Find EL intersection pressure values
        lfc_p_list = p_list
        el_p_list = _find_intersections(
            pressure[1:], parcel_temperature_profile[1:], temperature[1:], direction=-1, log_x=True
        )[0]
    else:  # intersect_type == 'EL'
        el_p_list = p_list
        # Find LFC intersection pressure values
        lfc_p_list = _find_intersections(
            pressure, parcel_temperature_profile, temperature, direction=1, log_x=True
        )[0]
    max_diff_idx = 0
    max_diff = 0
    for i, (lfc_p, el_p) in enumerate(zip(lfc_p_list, el_p_list)):
        diff = lfc_p - el_p
        if diff > max_diff:
            max_diff = diff
            max_diff_idx = i
    return np.array((p_list[max_diff_idx], t_list[max_diff_idx]))


@numba.njit
def _most_cape_option(
    intersect_type, p_list, t_list, pressure, temperature, dewpoint, parcel_temperature_profile
):
    """Calculate the LFC or EL that produces the most CAPE in the profile."""
    # Need to loop through all possible combinations of cape, find greatest cape profile
    max_cape = 0.
    lfc_chosen = IntersectionChoice.TOP
    el_chosen = IntersectionChoice.TOP
    for which_lfc in [IntersectionChoice.TOP, IntersectionChoice.BOTTOM]:
        for which_el in [IntersectionChoice.TOP, IntersectionChoice.BOTTOM]:
            cape = _cape_cin_single_profile(
                pressure, temperature, dewpoint, parcel_temperature_profile, which_lfc=which_lfc, which_el=which_el
            )[0]
            if cape > max_cape:
                lfc_chosen = which_lfc
                el_chosen = which_el

    if intersect_type == IntersectionType.LFC:
        if lfc_chosen == IntersectionChoice.TOP:
            xy = np.array((p_list[-1], t_list[-1]))
        else:  # BOTTOM is returned
            xy = np.array((p_list[0], t_list[0]))
    else:  # EL is returned
        if el_chosen == IntersectionChoice.TOP:
            xy = np.array((p_list[-1], t_list[-1]))
        else:
            xy = np.array((p_list[0], t_list[0]))
    return xy


@numba.njit
def _el(pressure, temperature, dewpoint, parcel_temperature_profile, which=IntersectionChoice.TOP):
    r"""Calculate the equilibrium level.

    This works by finding the last intersection of the ideal parcel path and
    the measured environmental temperature. If there is one or fewer intersections, there is
    no equilibrium level.

    Parameters
    ----------
    pressure : 1D array-like
        Atmospheric pressure profile, in Pa

    temperature : 1D array-like
        Temperature at the levels given by `pressure`, in K.

    dewpoint : 1D array-like
        Dewpoint at the levels given by `pressure`, in K.

    parcel_temperature_profile: 1D array-like
        The parcel's temperature profile from which to calculate the EL, in K.

    which: IntersectionChoice, optional
        Pick which EL to return. Options are TOP, BOTTOM, WIDE, MOST_CAPE;
        TOP returns the lowest-pressure EL (default),
        BOTTOM returns the highest-pressure EL,
        WIDE returns the EL whose corresponding EL is farthest away,
        MOST_CAPE returns the EL that results in the most CAPE in the profile.

    Returns
    -------
    numpy.ndarray
        Length 2 array containing EL pressure and temperature, in Pa and K, respectively.

    Notes
    -----
    TODO: original metpy implementation worked independently of CAPE/CIN (and so included nan
    masking). decide how you wish to handle that? also had dewpoint_start argument, which was
    unused by CAPE/CIN, so dropped.

    """
    # If the top of the sounding parcel is warmer than the environment, there is no EL
    if parcel_temperature_profile[-1] > temperature[-1]:
        return np.array((np.nan, np.nan))

    # Interpolate in log space to find the appropriate pressure - units have to be stripped
    # and reassigned to allow np.log() to function properly.
    x, y = _find_intersections(
        pressure[1:], parcel_temperature_profile[1:], temperature[1:], direction=-1, log_x=True
    )
    lcl_p = _lcl_scalar(pressure[0], temperature[0], dewpoint[0])[0]
    if len(x) > 0 and x[-1] < lcl_p:
        idx = x < lcl_p
        return _multiple_el_lfc_options(
            x, y, idx, which, pressure, parcel_temperature_profile, temperature, dewpoint, intersect_type=IntersectionType.EL
        )
    else:
        return np.array((np.nan, np.nan))


@numba.njit
def _cape_cin_single_profile(
    pressure,
    temperature,
    dewpoint,
    parcel_temperature_profile,
    which_lfc=IntersectionChoice.BOTTOM,
    which_el=IntersectionChoice.TOP
):
    r"""Base routine for calculating CAPE and CIN.

    Calculate the convective available potential energy (CAPE) and convective inhibition (CIN)
    of a given upper air profile and parcel path. CIN is integrated between the surface and
    LFC, CAPE is integrated between the LFC and EL (or top of sounding). Intersection points
    of the measured temperature profile and parcel profile are logarithmically interpolated.

    Parameters
    ----------
    pressure : 1D array-like
        Atmospheric pressure level(s) of interest, in order from highest to
        lowest pressure, in Pa.
    temperature : 1D array-like
        Atmospheric temperature corresponding to pressure, in K.
    dewpoint : 1D array-like
        Atmospheric dewpoint corresponding to pressure, in K.
    parcel_temperature_profile : 1D array-like
        Temperature profile of the parcel, in K.
    which_lfc : cumulonumba.IntersectionChoice
        Choose which LFC to integrate from. Valid options are TOP, BOTTOM, WIDE, and
        MOST_CAPE. Default is BOTTOM.
    which_el : cumulonumba.IntersectionChoice
        Choose which EL to integrate to. Valid options are TOP, BOTTOM, WIDE, and MOST_CAPE.
        Default is BOTTOM.

    Returns
    -------
    numpy.ndarray
        Length 2 array containing CAPE and CIN, in J kg-1.
    """
    # Clean nans
    mask = ~_nan_mask(pressure, temperature, dewpoint, parcel_temperature_profile)
    pressure = pressure[mask]
    temperature = temperature[mask]
    dewpoint = dewpoint[mask]
    parcel_temperature_profile = parcel_temperature_profile[mask]

    # Calculate LFC limit of integration
    lfc_pressure = _lfc(
        pressure,
        temperature,
        dewpoint,
        parcel_temperature_profile=parcel_temperature_profile,
        which=which_lfc
    )[0]

    # If there is no LFC, no need to proceed.
    if np.isnan(lfc_pressure):
        return np.array((0., 0.))

    # Calculate the EL limit of integration
    el_pressure = _el(
        pressure,
        temperature,
        dewpoint,
        parcel_temperature_profile=parcel_temperature_profile,
        which=which_el
    )[0]

    # No EL and we use the top reading of the sounding.
    if np.isnan(el_pressure):
        el_pressure = pressure[-1]

    # Difference between the parcel path and measured temperature profiles
    y = parcel_temperature_profile - temperature

    # Estimate zero crossings
    x, y = _find_append_zero_crossings(np.copy(pressure), y)

    # CAPE
    # Only use data between the LFC and EL for calculation
    p_mask = _less_or_close(x, lfc_pressure) & _greater_or_close(x, el_pressure)
    x_clipped = x[p_mask]
    y_clipped = y[p_mask]
    cape = Rd * np.trapz(y_clipped, np.log(x_clipped))

    # CIN
    # Only use data between the surface and LFC for calculation
    p_mask = _greater_or_close(x, lfc_pressure)
    x_clipped = x[p_mask]
    y_clipped = y[p_mask]
    cin = Rd * np.trapz(y_clipped, np.log(x_clipped))

    # Set CIN to 0 if it's returned as a positive value (Unidata/MetPy#1190)
    if cin > 0:
        cin = 0
    return np.array((cape, cin))


@numba.njit
def parcel_profile(pressure, temperature, dewpoint):
    r"""Calculate the profile a parcel takes through the atmosphere.

    The parcel starts at `temperature`, and `dewpoint`, lifted up
    dry adiabatically to the LCL, and then moist adiabatically from there.
    `pressure` specifies the pressure levels for the profile.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure level(s) of interest. This array must be from
        high to low pressure.

    temperature : `pint.Quantity`
        Starting temperature

    dewpoint : `pint.Quantity`
        Starting dewpoint

    Returns
    -------
    `pint.Quantity`
        The parcel's temperatures at the specified pressure levels

    """
    _, _, _, t_l, _, t_u = _parcel_profile_helper(pressure, temperature, dewpoint)
    return np.concatenate((t_l, t_u))


# def parcel_profile_with_lcl(pressure, temperature, dewpoint):
#     r"""Calculate the profile a parcel takes through the atmosphere.

#     The parcel starts at `temperature`, and `dewpoint`, lifted up
#     dry adiabatically to the LCL, and then moist adiabatically from there.
#     `pressure` specifies the pressure levels for the profile. This function returns
#     a profile that includes the LCL.

#     Parameters
#     ----------
#     pressure : `pint.Quantity`
#         Atmospheric pressure level(s) of interest. This array must be from
#         high to low pressure.

#     temperature : `pint.Quantity`
#         Atmospheric temperature at the levels in `pressure`. The first entry should be at
#         the same level as the first `pressure` data point.

#     dewpoint : `pint.Quantity`
#         Atmospheric dewpoint at the levels in `pressure`. The first entry should be at
#         the same level as the first `pressure` data point.

#     Returns
#     -------
#     pressure : `pint.Quantity`
#         The parcel profile pressures, which includes the specified levels and the LCL

#     ambient_temperature : `pint.Quantity`
#         Atmospheric temperature values, including the value interpolated to the LCL level

#     ambient_dew_point : `pint.Quantity`
#         Atmospheric dewpoint values, including the value interpolated to the LCL level

#     profile_temperature : `pint.Quantity`
#         The parcel profile temperatures at all of the levels in the returned pressures array,
#         including the LCL
#     """
#     p_l, p_lcl, p_u, t_l, t_lcl, t_u = _parcel_profile_helper(pressure, temperature[0],
#                                                               dewpoint[0])
#     new_press = concatenate((p_l, p_lcl, p_u))
#     prof_temp = concatenate((t_l, t_lcl, t_u))
#     new_temp = _insert_lcl_level(pressure, temperature, p_lcl)
#     new_dewp = _insert_lcl_level(pressure, dewpoint, p_lcl)
#     return new_press, new_temp, new_dewp, prof_temp
#### TODO modify /\ ####


@numba.njit
def _parcel_profile_helper(pressure, temperature, dewpoint):
    """Help calculate parcel profiles.

    Returns the temperature and pressure, above, below, and including the LCL. The
    other calculation functions decide what to do with the pieces.

    """
    # Check that pressure does not increase.
    if not np.all(pressure[:-1] >= pressure[1:]):
        raise ValueError(
            "Pressure increases between at least two points in your sounding. Using "
            "scipy.signal.medfilt may fix this."
        )

    # Find the LCL
    press_lcl, temp_lcl = _lcl_scalar(pressure[0], temperature, dewpoint)
    press_lcl = press_lcl

    # Find the dry adiabatic profile, *including* the LCL. We need >= the LCL in case the
    # LCL is included in the levels. It's slightly redundant in that case, but simplifies
    # the logic for removing it later.
    press_lower = np.concatenate((pressure[pressure >= press_lcl], np.array([press_lcl])))
    temp_lower = dry_lapse(press_lower, temperature)

    # If the pressure profile doesn't make it to the lcl, we can stop here
    if _greater_or_close(np.nanmin(pressure), press_lcl):
        return (
            press_lower[:-1],
            np.array([press_lcl]),
            np.empty((0,), dtype=pressure.dtype),
            temp_lower[:-1],
            temp_lcl,
            np.empty((0,), dtype=pressure.dtype)
        )

    # Establish profile above LCL
    press_upper = np.concatenate((np.array([press_lcl]), pressure[pressure < press_lcl]))

    # Find moist pseudo-adiabatic profile starting at the LCL
    temp_upper = _moist_lapse_1d(press_upper, temp_lower[-1], press_upper[0])

    # Return profile pieces
    return (
        press_lower[:-1],
        np.array([press_lcl]),
        press_upper[1:],
        temp_lower[:-1],
        temp_lcl,
        temp_upper[1:]
    )


##### TODO modify \/ ####
#def _insert_lcl_level(pressure, temperature, lcl_pressure):
#    """Insert the LCL pressure into the profile."""
#    interp_temp = interpolate_1d(lcl_pressure, pressure, temperature)
#
#    # Pressure needs to be increasing for searchsorted, so flip it and then convert
#    # the index back to the original array
#    loc = pressure.size - pressure[::-1].searchsorted(lcl_pressure)
#    return units.Quantity(np.insert(temperature.m, loc, interp_temp.m), temperature.units)
##### TODO modify /\ ####
